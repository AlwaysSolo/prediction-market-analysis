# `spot_v1` Tutorial

## What `spot_v1` Is

`spot_v1` is the repo’s first external-BTC-spot feature schema for `KXBTC15M`.

It adds a live dual-venue BTC spot proxy and new spot-aware model features on top of the existing Kalshi-only feature set. The goal is not to build a generic BTC mid; the goal is to reduce basis versus Kalshi’s BTC settlement reference as much as possible with free live feeds.

At a high level, `spot_v1` adds:

- live Coinbase + Kraken external spot ingestion
- a consolidated BTC spot price and 60-second TWAP
- true log-moneyness features
- external-spot returns and realized-volatility features
- Black-Scholes-style proxy pricing features
- Kalshi-versus-BS divergence features
- strict live gating: if a `spot_v1` artifact is selected and the spot feed is unavailable, the runner refuses to score

## Current Status

What is implemented now:

- live spot feed module: `src/live/data/btc_spot_feed.py`
- spot feature logic: `src/live/kalshi/spot_features.py`
- shared strike extractor: `src/live/kalshi/strike.py`
- live feature-engine integration
- archive schema bump to `kalshi_feature_row_v2`
- `spot_v1` support in:
  - `scripts/train_kxbtc15m_bagged_lasso.py`
  - `scripts/train_kxbtc15m_lightgbm.py`
  - `scripts/run_kalshi_bagged_lasso_live.py`
  - `scripts/run_kalshi_research_multi_model.py`
  - `scripts/run_kalshi_multi_model_execution_engine.py`

What is not fully implemented yet:

- a repo-native historical BTC spot backfill generator
- paid L1 historical quote ingestion for Phase 2
- a full live smoke test against Coinbase + Kraken in this repo snapshot
- the full operational hardening pass for sequence-gap resync and drift monitoring

Because of that, the offline side of `spot_v1` currently assumes you already have historical external spot parquet data available.

## Files You Will Use

Core source files:

- `src/live/data/btc_spot_feed.py`
- `src/live/kalshi/spot_features.py`
- `src/live/kalshi/strike.py`
- `src/live/kalshi/feature_engine.py`
- `src/live/kalshi/offline_training.py`

Training CLIs:

- `scripts/train_kxbtc15m_bagged_lasso.py`
- `scripts/train_kxbtc15m_lightgbm.py`

Runtime CLIs:

- `scripts/run_kalshi_bagged_lasso_live.py`
- `scripts/run_kalshi_research_multi_model.py`
- `scripts/run_kalshi_multi_model_execution_engine.py`

Related reporting:

- `scripts/parity_report.py`
- `docs/PARITY_REPORT_TUTORIAL.md`

## What `spot_v1` Adds

Diagnostics written into feature rows:

- `btc_spot_price`
- `btc_spot_twap_60s`
- `btc_spot_age_ms`
- `btc_spot_is_fresh`
- `btc_spot_venues_fresh`
- `btc_spot_venue_divergence_bps`
- `btc_vol_effective_sample_size`
- `btc_spot_source`

Model features added by `spot_v1`:

- moneyness:
  - `btc_log_moneyness`
  - `btc_log_moneyness_twap60`
  - `btc_log_moneyness_per_minute`
- returns:
  - `btc_spot_return_30s`
  - `btc_spot_return_120s`
  - `btc_spot_return_300s`
  - `btc_spot_return_900s`
- realized vol:
  - `btc_spot_vol_120s`
  - `btc_spot_vol_300s`
  - `btc_spot_vol_900s`
  - `btc_spot_vol_1800s`
  - `btc_spot_vol_ewma_hl300`
- BS-style proxy probabilities:
  - `btc_bs_yes_prob_120s`
  - `btc_bs_yes_prob_300s`
  - `btc_bs_yes_prob_900s`
  - `btc_bs_yes_prob_1800s`
  - `btc_bs_yes_prob_ewma`
- Kalshi minus BS divergence families:
  - raw probability gap
  - log-odds gap
  - detrended gap
- `btc_kalshi_implied_vol`

## PowerShell Notes

If you use PowerShell, use the backtick for multiline commands:

```powershell
python scripts/train_kxbtc15m_bagged_lasso.py `
  --help
```

Do not use `^` in PowerShell. That is for `cmd.exe`.

The backtick must be the last character on the line.

## The Big Workflow

The typical `spot_v1` workflow is:

1. prepare historical external spot parquet
2. build a fresh `spot_v1` dataset cache
3. train a `spot_v1` model
4. verify the artifact manifest says `spot_v1`
5. shadow the new artifact against a baseline
6. inspect logs, archives, and parity-style reports
7. only then consider promotion

## Historical Spot Data Requirements

### What the trainer expects

When you build a `spot_v1` cache, the training scripts call `load_external_spot_frame(...)` and `enrich_market_feature_frame_with_external_spot(...)`.

That means your historical external spot root must contain parquet data with at least these columns:

- `event_time`
- `btc_spot_price`
- `btc_spot_twap_60s`
- `btc_spot_age_ms`
- `btc_spot_is_fresh`
- `btc_spot_venues_fresh`
- `btc_spot_venue_divergence_bps`
- `btc_vol_effective_sample_size`
- `btc_spot_source`
- `btc_spot_return_30s`
- `btc_spot_return_120s`
- `btc_spot_return_300s`
- `btc_spot_return_900s`
- `btc_spot_vol_120s`
- `btc_spot_vol_300s`
- `btc_spot_vol_900s`
- `btc_spot_vol_1800s`
- `btc_spot_vol_ewma_hl300`

`event_time` is required and is interpreted in UTC.

### Accepted layout

You can point `--external-spot-root` at:

- one parquet file, or
- a directory tree containing many parquet files

The loader recursively reads all `*.parquet` files below that root and sorts them by `event_time`.

### Recommended folder layout

```text
data/
  external_spot/
    phase1_trade_proxy/
      2026-04-01.parquet
      2026-04-02.parquet
      2026-04-03.parquet
```

### Recommended source labels

Use `btc_spot_source` consistently:

- `quote` for quote-derived rows
- `trade_proxy` for trade-proxy history

## Training `spot_v1`

## Important Cache Rule

This is the easiest mistake to make:

If you pass `--reference-run-dir`, the training scripts will try to reuse that run’s dataset cache by default.

That is good for Kalshi-only retrains, but it is usually wrong for your first `spot_v1` build.

Why:

- the old cache may not contain any spot columns
- the script only builds a new cache if the target cache directory does not already exist

So for your first `spot_v1` run, explicitly set a new `--dataset-cache-dir`.

After that first build, you can reuse the `spot_v1` cache safely.

## Bagged LASSO Example

This is the best first training command if you want apples-to-apples comparison with an existing bagged-lasso split:

```powershell
python scripts/train_kxbtc15m_bagged_lasso.py `
  --reference-run-dir artifacts/kalshi/kxbtc15m_bagged_lasso/latest `
  --dataset-cache-dir artifacts/kalshi/kxbtc15m_bagged_lasso/spot_v1_phase1_20260418/datasets/spot_v1 `
  --feature-schema spot_v1 `
  --external-spot-root data/external_spot/phase1_trade_proxy `
  --external-spot-latency-ms 200 `
  --run-name spot_v1_phase1_20260418
```

What this does:

- reuses the existing train/validation/test split from `latest`
- builds a brand-new spot-aware cache
- trains bagged-lasso with `spot_v1`
- writes a new artifact run

### Bagged LASSO outputs

Look in the new run directory for:

- `summary.json`
- `feature_manifest.json`
- `policy.json`
- `split_manifest.json`
- `validation_predictions.parquet`
- `test_predictions.parquet`
- `validation_trade_records.parquet`
- `test_trade_records.parquet`
- `walk_forward.json` unless you used `--skip-walk-forward`

## LightGBM Example

```powershell
python scripts/train_kxbtc15m_lightgbm.py `
  --reference-run-dir artifacts/kalshi/kxbtc15m_lightgbm/latest `
  --dataset-cache-dir artifacts/kalshi/kxbtc15m_lightgbm/spot_v1_phase1_20260418/datasets/spot_v1 `
  --feature-schema spot_v1 `
  --external-spot-root data/external_spot/phase1_trade_proxy `
  --external-spot-latency-ms 200 `
  --run-name spot_v1_phase1_20260418
```

The same cache rule applies here: use a fresh cache path for the first `spot_v1` build.

## When You Can Omit `--external-spot-root`

You can omit `--external-spot-root` only when:

- you are reusing an already-built `spot_v1` cache, and
- `--dataset-cache-dir` points at that cache

You cannot omit it when building a fresh `spot_v1` cache.

## How To Confirm You Actually Trained `spot_v1`

After training, open `feature_manifest.json`.

You should see:

- `schema_name: "spot_v1"`
- `feature_order`
- `feature_order_hash`

The training summary should also reflect:

- `feature_schema: "spot_v1"`
- `external_spot_latency_ms`

If `schema_name` is not `spot_v1`, do not assume the artifact is spot-aware.

## Reusing the Artifact in Live and Shadow Runners

The runtime scripts detect `spot_v1` from the artifact’s `feature_manifest.json`.

That means:

- keep the artifact in its normal run directory layout
- do not copy only `model.joblib` somewhere else and expect automatic schema detection to keep working

The runtime looks up `feature_manifest.json` relative to the model file.

## Dedicated Bagged-Lasso Runner

## Shadow Mode Example

If your bagged-lasso run directory is a `spot_v1` artifact, the dedicated live runner will automatically start the external spot feed.

```powershell
python scripts/run_kalshi_bagged_lasso_live.py `
  --mode shadow `
  --run-dir artifacts/kalshi/kxbtc15m_bagged_lasso/spot_v1_phase1_20260418 `
  --log-root output/live/kalshi_bagged_lasso_spotv1_shadow
```

What happens at startup:

1. it reads `feature_manifest.json`
2. it sees `schema_name = "spot_v1"`
3. it starts the Coinbase + Kraken spot feed
4. it waits for a fresh spot snapshot
5. it only continues if the feed is ready

If the spot feed is not ready before the timeout, the runner fails hard instead of silently degrading.

## Live Mode Example

```powershell
python scripts/run_kalshi_bagged_lasso_live.py `
  --mode live `
  --confirm-live `
  --run-dir artifacts/kalshi/kxbtc15m_bagged_lasso/spot_v1_phase1_20260418 `
  --log-root output/live/kalshi_bagged_lasso_spotv1_live
```

This still requires the normal live-trading env and safety flags, including:

- valid Kalshi production credentials
- `KALSHI_PROD_EXECUTION_ENABLE_LIVE_TRADING=true`
- `--confirm-live`

## Useful live-run options

- `--spot-start-timeout-seconds`
- `--quote-start-timeout-seconds`
- `--signal-profile dedicated-v1|research-parity`
- `--enable-layering`

## Multi-Model Research Comparison

Use the research runner when you want to compare a baseline and a `spot_v1` artifact side by side on one shared stream without sending live orders.

Example:

```powershell
python scripts/run_kalshi_research_multi_model.py `
  --runtime-run-dir baseline=bagged_lasso=artifacts/kalshi/kxbtc15m_bagged_lasso/latest `
  --runtime-run-dir spotv1=bagged_lasso=artifacts/kalshi/kxbtc15m_bagged_lasso/spot_v1_phase1_20260418 `
  --series KXBTC15M `
  --log-root output/live_research/kalshi_spotv1_compare `
  --archive full `
  --max-runtime-seconds 3600
```

This is a good Phase 1 comparison path because:

- both artifacts see the same market-data stream
- spot feed is started once and shared
- research sampler/ledger outputs are easy to compare later

## Multi-Model Paper Comparison

Use the execution-engine runner when you want side-by-side paper trading.

Example:

```powershell
python scripts/run_kalshi_multi_model_execution_engine.py `
  --execution-mode paper `
  --runtime-run-dir baseline=bagged_lasso=artifacts/kalshi/kxbtc15m_bagged_lasso/latest `
  --runtime-run-dir spotv1=bagged_lasso=artifacts/kalshi/kxbtc15m_bagged_lasso/spot_v1_phase1_20260418 `
  --series KXBTC15M `
  --log-root output/live/kalshi_spotv1_shadow_compare
```

Important:

- this runner is still effectively a paper/shadow comparison path for these regularized artifacts
- do not treat it as the dedicated production runner

## What Gets Logged at Runtime

If a runner starts the spot feed, you will see external spot logs below the run’s `log_root`.

Typical structure:

```text
output/live/.../
  external_spot/
    external_spot_raw/
      production/
        YYYY-MM-DD/
          events.jsonl
    external_spot_normalized/
      production/
        YYYY-MM-DD/
          events.jsonl
```

The live and research archives also include spot-enriched feature rows:

- `feature_rows` now write `schema_version = kalshi_feature_row_v2`
- spot diagnostics are embedded in those archived rows

## How To Read Training Outputs

## `summary.json`

This is the quickest high-level read.

Look for:

- `feature_schema`
- `train_rows`
- `validation_rows`
- `test_rows`
- `validation_log_loss_calibrated`
- `test_log_loss_calibrated`
- `test_metrics`
- `walk_forward_completed`

## `feature_manifest.json`

This is the schema truth source for the artifact.

Check:

- `schema_name`
- `feature_order`
- `feature_order_hash`
- `hourly_context_series` if used

## `policy.json`

This tells you which policy settings the run selected from validation.

It matters because a better model can still look worse if the policy differs a lot.

## `validation_predictions.parquet` and `test_predictions.parquet`

Use these when you want:

- calibration checks
- reliability analysis
- near-the-money slicing
- custom comparisons against a baseline model

## `validation_trade_records.parquet` and `test_trade_records.parquet`

These are the best files for:

- PnL attribution
- trade-count comparison
- hold-time analysis
- price-band and regime analysis

## `walk_forward.json`

Use this for out-of-sample stability checks across time.

If you care about operational robustness, do not look only at one aggregate holdout number.

## How To Read Runtime Outputs

## Research logs

In research runs, focus on:

- research sample logs
- settlement logs
- archive feature rows
- archive model outputs

These are the best sources for Phase 1 shadow evidence.

## Paper/live execution logs

In execution runs, focus on:

- signal logs
- execution logs
- optional layering logs
- archive feature rows
- archive strategy events

## Dashboard usage

If you already use the local dashboard, point it at the run’s `log_root`.

The runners print the usual dashboard hint:

- `http://localhost:8765/tools/live_trading_dashboard.html`

## Recommended Comparison Workflow

## Phase 1: directional evidence

Recommended order:

1. train baseline and `spot_v1` on the same split
2. compare `summary.json`
3. compare `test_predictions.parquet`
4. run multi-model research or paper shadow
5. compare logged hypothetical PnL and behavior for at least several days

At this stage:

- improvement is evidence
- it is not yet hard proof of alpha

## Phase 2: stronger acceptance

Once you have better historical quote data:

1. regenerate `spot_v1` training data from higher-fidelity historical spot
2. rerun the same split
3. compare holdout log loss, replay PnL, and calibration
4. only then promote aggressively

## Using Parity Report After Live or Shadow Usage

After you have settled live trades or long enough shadow history, you can use the existing parity tooling to inspect whether the stack is leaking between research, shadow, and live.

See:

- `scripts/parity_report.py`
- `docs/PARITY_REPORT_TUTORIAL.md`

That is especially useful after a `spot_v1` rollout because it helps answer:

- is the model better offline but worse in live execution?
- is the shadow path matching research behavior?
- is the live path matching shadow behavior?

## Common Mistakes

## 1. Reusing a Kalshi-only cache for `spot_v1`

Symptom:

- training fails
- or the run does not contain the expected spot features

Fix:

- use a fresh `--dataset-cache-dir` for the first `spot_v1` build

## 2. Forgetting `--external-spot-root`

Symptom:

- `spot_v1` cache build exits immediately

Fix:

- pass `--external-spot-root` when the target cache does not already exist

## 3. Copying only `model.joblib`

Symptom:

- runner silently falls back to default schema detection
- spot feed does not start

Fix:

- keep the whole artifact run directory, especially `feature_manifest.json`

## 4. Using `^` in PowerShell

Symptom:

- parser errors before the Python script even starts

Fix:

- use the PowerShell backtick or a one-line command

## 5. Expecting the repo to generate historical spot bootstrap automatically

Symptom:

- you have the training code but no external parquet to feed it

Reality:

- the historical bootstrap generator is not part of this implementation yet

Fix:

- prepare external spot parquet yourself for now

## 6. Treating paper comparison as production readiness

Paper and research comparisons are necessary, but they are not the same as production validation.

Use them to narrow risk, not to skip later checks.

## Troubleshooting

## Runner fails waiting for spot feed

Check:

- internet connectivity
- firewall/proxy restrictions
- Coinbase or Kraken websocket availability
- `--spot-start-timeout-seconds`

## Feature rows are not scoreable under `spot_v1`

Check:

- whether the feed is publishing fresh snapshots
- whether archived or runtime spot age is too high
- whether the runner actually loaded a `spot_v1` artifact

## `feature_manifest.json` says `default`

That means you did not actually train a `spot_v1` artifact, even if you expected to.

Re-check:

- `--feature-schema spot_v1`
- dataset cache path
- run directory used at inference time

## Research or paper runner does not start the spot feed

Check the model run directories passed via `--runtime-run-dir`.

If none of the selected artifacts has `schema_name = "spot_v1"`, the shared spot feed will not start.

## Quick Command Reference

## Train bagged-lasso `spot_v1`

```powershell
python scripts/train_kxbtc15m_bagged_lasso.py `
  --reference-run-dir artifacts/kalshi/kxbtc15m_bagged_lasso/latest `
  --dataset-cache-dir artifacts/kalshi/kxbtc15m_bagged_lasso/spot_v1_phase1_20260418/datasets/spot_v1 `
  --feature-schema spot_v1 `
  --external-spot-root data/external_spot/phase1_trade_proxy `
  --external-spot-latency-ms 200 `
  --run-name spot_v1_phase1_20260418
```

## Train LightGBM `spot_v1`

```powershell
python scripts/train_kxbtc15m_lightgbm.py `
  --reference-run-dir artifacts/kalshi/kxbtc15m_lightgbm/latest `
  --dataset-cache-dir artifacts/kalshi/kxbtc15m_lightgbm/spot_v1_phase1_20260418/datasets/spot_v1 `
  --feature-schema spot_v1 `
  --external-spot-root data/external_spot/phase1_trade_proxy `
  --external-spot-latency-ms 200 `
  --run-name spot_v1_phase1_20260418
```

## Run dedicated bagged-lasso in shadow mode

```powershell
python scripts/run_kalshi_bagged_lasso_live.py `
  --mode shadow `
  --run-dir artifacts/kalshi/kxbtc15m_bagged_lasso/spot_v1_phase1_20260418 `
  --log-root output/live/kalshi_bagged_lasso_spotv1_shadow
```

## Run multi-model research comparison

```powershell
python scripts/run_kalshi_research_multi_model.py `
  --runtime-run-dir baseline=bagged_lasso=artifacts/kalshi/kxbtc15m_bagged_lasso/latest `
  --runtime-run-dir spotv1=bagged_lasso=artifacts/kalshi/kxbtc15m_bagged_lasso/spot_v1_phase1_20260418 `
  --series KXBTC15M `
  --log-root output/live_research/kalshi_spotv1_compare `
  --archive full `
  --max-runtime-seconds 3600
```

## Run multi-model paper comparison

```powershell
python scripts/run_kalshi_multi_model_execution_engine.py `
  --execution-mode paper `
  --runtime-run-dir baseline=bagged_lasso=artifacts/kalshi/kxbtc15m_bagged_lasso/latest `
  --runtime-run-dir spotv1=bagged_lasso=artifacts/kalshi/kxbtc15m_bagged_lasso/spot_v1_phase1_20260418 `
  --series KXBTC15M `
  --log-root output/live/kalshi_spotv1_shadow_compare
```

## Final Advice

Treat `spot_v1` as a schema and workflow change, not just “some extra columns.”

The safest way to work with it is:

- build a fresh spot-aware cache
- keep the artifact directory intact
- verify `feature_manifest.json`
- compare against a baseline in research or paper mode first
- inspect the generated summaries and archives before promotion

That discipline matters more here than with a small feature tweak, because `spot_v1` changes both the training distribution and the live runtime dependencies.
