# GTC TTL Rollout Notes

## Why This Change Was Made

The live bagged-lasso bot was showing a very high zero-fill IOC cancel rate even though EC2-to-Kalshi latency was healthy. The main issues we identified were:

- mixed stale YES/NO quote state in the collector
- blind IOC submission without a last-moment executable-book check
- no visibility into top-of-book size from the live feed
- the opening probe tranche was too dependent on instant taker liquidity

The goal of this patch set is to preserve the strong prediction structure from the winning shadow configuration while making live execution more realistic and more diagnosable.

## What Changed

### 1. Mixed-quote protection

We patched the live collector so it no longer keeps stale opposite-side quotes alive when a fresh same-side quote arrives. That fixes the earlier failure mode where YES fields were current but NO fields were old.

We also added a quote-consistency guard in signal evaluation so internally inconsistent books are blocked instead of traded.

### 2. Pre-submit live orderbook check

Before every live submit, the execution engine now calls Kalshi's orderbook endpoint and reconstructs the current executable ask from the opposite top bid.

For immediate orders, the engine blocks the trade locally if:

- the live executable ask has moved above the bot's limit
- the top-of-book size is smaller than the requested contracts
- the orderbook check itself fails

This prevents a chunk of bad submits from ever reaching the exchange.

For passive probe orders that use short-TTL `good_till_canceled`, the orderbook check is still logged but it is no longer treated as a hard submit gate. That matters because a resting probe is allowed to sit below the current ask; it should not be rejected just because it is not immediately marketable.

### 3. Top-of-book size is now parsed from the feed

The collector now parses top-of-book size fields from Kalshi's feed:

- `yes_bid_size`
- `yes_ask_size`

Because Kalshi's binary book is complementary, those are mirrored into:

- `no_ask_size` from `yes_bid_size`
- `no_bid_size` from `yes_ask_size`

These sizes are now carried through state and archived into market-event rows so live analysis can compare:

- feed top-book size
- REST orderbook top-book size
- actual fill outcome

### 4. Probe orders now use short-TTL GTC

The opening probe tranche is no longer forced to use IOC.

The execution engine now detects the probe order by tranche metadata:

- `tranche_index = 0`
- `tranche_window = 10m`
- `tranche_reason = opened`
- `lifecycle_state = probe_pending`

That probe now submits as:

- `time_in_force = good_till_canceled`
- with a short `expiration_ts`

This gives the opening order a brief chance to rest instead of demanding an immediate match.

All later add tranches still submit as IOC.

### 5. New tranche schedule

The layering schedule is now:

- `10m`
- `5m`
- `4m`
- `3m`

The old `1m` final tranche is gone. The new final tranche is `3m`.

This was done because the `2m/1m` area was too competitive and often too thin for the live bot.

### 6. YES and NO are both still live

We did not force a NO-only profile.

The live runner still allows both YES and NO so we can collect proper live-side execution data and compare behavior across both directions.

## New Runtime Defaults

These are the important execution defaults after the patch:

- pre-submit orderbook check: enabled
- pre-submit orderbook depth: `1`
- probe order TIF: `good_till_canceled`
- probe order TTL: `2.0` seconds

The probe behavior is configurable through env vars:

- `KALSHI_PROD_EXECUTION_PROBE_ORDER_TIME_IN_FORCE`
- `KALSHI_PROD_EXECUTION_PROBE_ORDER_TTL_SECONDS`

The pre-submit orderbook check is configurable through:

- `KALSHI_PROD_EXECUTION_ENABLE_PRE_SUBMIT_ORDERBOOK_CHECK`
- `KALSHI_PROD_EXECUTION_PRE_SUBMIT_ORDERBOOK_DEPTH`

## Logging Added

The live submit logs now include both feed-side and REST-side execution context.

New useful fields include:

- `ticker_update_to_submit_requested_ms`
- `received_at_to_submit_requested_ms`
- `top_book_side`
- `top_book_price_cents`
- `top_book_contracts`
- `current_executable_ask_cents`
- `feed_top_book_side`
- `feed_top_book_price_cents`
- `feed_top_book_contracts`
- `feed_executable_ask_cents`
- `time_in_force`
- `expiration_ts`
- `is_probe_order`

This should make it much easier to separate:

- stale-feed problems
- book-moved-away problems
- no-size problems
- strategy-selection problems

## How To Run This Version

Use the dedicated bagged-lasso live runner with:

- `--enable-layering`
- `--allow-stacking`
- `--disable-regime-control`
- `--signal-profile research-parity`
- `--edge-threshold-cents 2`

There is no CLI flag for the tranche schedule. The code now already uses `10m/5m/4m/3m`.

## Validation

Targeted tests passed after the patch set:

- `tests/test_kalshi_execution.py`
- `tests/test_kalshi_live_collector.py`
- `tests/test_kalshi_layering.py`
- `tests/test_kalshi_feature_engine.py`
- `tests/test_kalshi_live_archive.py`
- `tests/test_kalshi_signal_risk.py`

Result:

- `95 passed`
