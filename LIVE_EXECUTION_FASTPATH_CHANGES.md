# Live Execution Fast-Path Changes

Date: 2026-04-12

## Purpose

This document explains the live-execution latency reductions that were implemented after the analysis in [LIVE_EXECUTION_AUDIT.md](/c:/prediction-market-analysis/LIVE_EXECUTION_AUDIT.md):

1. remove the extra REST `GET orderbook` round trip from the hot path for immediate live orders
2. remove synchronous JSONL file appends from the event loop by moving them to a background writer
3. collapse the last queue hop before execution by handing trade intents directly into the execution engine in live mode
4. add an asymmetric `NO`-probe IOC limit cushion so `NO` opening probes can bid a few cents more aggressively than the model limit

This is a detailed implementation note, not a strategy note. It focuses on what changed in code, why it changed, what behavior now differs in live mode, and what is still intentionally unchanged.

## Problem Statement

The repo’s live stack was originally optimized for correctness, observability, and replay/debuggability. That design made sense while the system was still evolving, but it left avoidable latency in the live trigger-to-submit path.

Before these changes, an immediate live order could pay for:

1. multiple queue hops across collector -> feature engine -> scorer -> signal -> layering -> execution
2. a REST `GET /markets/{ticker}/orderbook` pre-submit check
3. multiple synchronous JSONL file writes on the event loop
4. a REST `POST /portfolio/orders`

Even when raw EC2-to-Kalshi network latency is good, the extra REST call plus event-loop blocking can materially reduce the chance of hitting fleeting top-of-book liquidity.

The audit conclusion was that the highest-ROI changes were:

1. bypass the extra REST orderbook check for immediate live orders
2. move logging off the event loop
3. collapse the trade-intent-source -> execution queue handoff in live mode
4. tune the remaining `NO`-side live miss-rate without loosening all orders globally

The broader multi-stage queue architecture was intentionally left mostly intact because a full redesign would be broader and riskier.

## Change 1: Immediate Live Orders Can Skip the REST Orderbook Check

### What changed

The execution config now supports a new live-execution flag:

- `skip_rest_orderbook_check_for_immediate_orders`

This was added in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:111).

It is loaded from environment in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:165).

I also added a new `check_source` field to `_LiveOrderbookCheckResult` in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:212), so logs can tell us whether a pre-submit decision came from:

- `rest_orderbook`
- `feed_quote`
- `disabled`

### New fast-path function

I added `_check_feed_quote_before_submit(...)` in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:1006).

This function:

1. reads the current local quote context from the collector/scorer path
2. derives the current executable ask and top-of-book size from that feed state
3. applies the same logical safety tests as the previous pre-submit gate, but without a new network call

For immediate orders it can reject locally for:

- `feed_limit_moved_away`
- `feed_insufficient_size`

It returns a fully populated `_LiveOrderbookCheckResult`, including:

- current executable ask
- top-book side
- top-book price
- top-book contracts
- feed timestamps
- latency from quote timestamps to check time
- `orderbook_roundtrip_ms=0.0`
- `check_source="feed_quote"`

### How submit behavior changed

The main live submit path in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:1298) now branches like this:

- if the order requires immediate matching and `skip_rest_orderbook_check_for_immediate_orders=True`, use the feed-only pre-submit check
- otherwise, keep using the original REST orderbook check

That decision is made in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:1303).

This means:

- immediate live orders can now skip a full REST round trip before submit
- passive probe orders do not use this shortcut unless they are also configured as immediate-match
- the safety gate is preserved, but the data source changes from remote REST orderbook to current feed state

### Why this was done

The extra REST `GET orderbook` had become part of the critical path for fast live entries. In practice that meant:

1. signal fires
2. execution engine asks Kalshi for a fresh book
3. only after that round trip does it place the order

That is protective, but expensive. On the markets you are trading, especially near expiry, that extra trip can easily be the difference between:

- hitting the visible liquidity
- arriving after it is gone

The feed-only check is not perfect, but it is much cheaper and still preserves the two most useful local guards:

- do not submit if the visible executable ask is already above the limit
- do not submit if visible size is already below requested contracts

### What did not change

I did not remove the orderbook check entirely.

The system still performs a pre-submit check. The change is only that, for immediate orders in the fast-path mode, the check uses local feed state instead of a fresh REST call.

I also did not make this the default for every possible execution config. The base config default is still conservative:

- `skip_rest_orderbook_check_for_immediate_orders=False`

That keeps generic codepaths safe by default.

### What the dedicated live runner now does

The dedicated bagged-lasso live runner now opts into the fast path by default in [run_kalshi_bagged_lasso_live.py](/c:/prediction-market-analysis/scripts/run_kalshi_bagged_lasso_live.py:135).

That means the specialized production runner uses:

- feed-only pre-submit checks for immediate orders
- original behavior for other submit policies

This is intentional. The fast-path optimization is being applied where it matters most, without silently changing every execution entry point in the codebase.

## Change 2: JSONL Logging Moved Off the Event Loop

### What changed

I added a shared background JSONL logger in [jsonl_logger.py](/c:/prediction-market-analysis/src/live/kalshi/jsonl_logger.py:16).

Before this change, several live components each had their own inline `JsonlEventLogger` implementation that:

1. created the day directory
2. opened the file
3. appended a JSONL row
4. did all of that directly inside async code on the main event loop

The affected components were:

- collector
- execution engine
- layering engine
- signal engine
- research engine

Those duplicated local logger classes were removed and replaced with the shared implementation.

### New logger design

The shared `JsonlEventLogger` works like this:

1. `write(...)` builds the row and target path
2. `write(...)` enqueues `(path, row)` into an internal `asyncio.Queue`
3. a background writer task consumes that queue
4. the writer task performs the actual filesystem append

Relevant methods:

- `write(...)` in [jsonl_logger.py](/c:/prediction-market-analysis/src/live/kalshi/jsonl_logger.py:32)
- `flush(...)` in [jsonl_logger.py](/c:/prediction-market-analysis/src/live/kalshi/jsonl_logger.py:44)
- `close(...)` in [jsonl_logger.py](/c:/prediction-market-analysis/src/live/kalshi/jsonl_logger.py:49)

### Why this was done

The original synchronous logger was simple and durable, but it added event-loop blocking exactly where we least wanted it:

- quote ingestion
- signal progression
- order submission
- execution state publication

For the execution engine in particular, a single live order may log several events in rapid succession:

- `intent_claimed`
- `pre_submit_orderbook_check`
- `submit_requested`
- `submit_response` or `submit_error`

Each synchronous append adds avoidable latency and jitter to the same loop that is supposed to keep moving the order forward.

Moving the file I/O to a background task preserves the JSONL audit trail while making the hot path much less sensitive to filesystem timing.

### Shutdown safety

One risk of async background logging is losing tail events during shutdown.

To avoid that, I added guarded logger-close handling to the components that own these loggers. Each component now:

1. emits its final stop event where appropriate
2. checks whether the logger supports `close()`
3. awaits it when present

That lets the logger drain its queue before the component fully exits.

This behavior was intentionally written in a guarded way so that tests using fake capturing loggers do not break.

### Why I used a shared logger instead of per-module variants

The repo previously had several nearly identical inline logger classes. Consolidating them into one shared implementation has three benefits:

1. the event-loop fix is implemented once, not five times
2. future logging behavior changes only need to happen in one place
3. the collector/execution/signal/layering stacks all use the same semantics for queueing, flushing, and shutdown

## Change 3: Live Mode Can Bypass the Trade-Intent Queue Before Execution

### What changed

The trade-intent source interface now supports optional direct callback handoff in addition to queue subscription. That interface change is in [trade_intent_source.py](/c:/prediction-market-analysis/src/live/kalshi/trade_intent_source.py:16).

Both live intent producers now implement the direct-callback methods:

- [signal_risk.py](/c:/prediction-market-analysis/src/live/kalshi/signal_risk.py:775)
- [layering.py](/c:/prediction-market-analysis/src/live/kalshi/layering.py:219)

The execution config now has a new live-only knob:

- `enable_direct_trade_intent_handoff_in_live_mode`

This was added in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:119) and loaded from environment in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:170).

### Previous behavior

Before this change, even after `signal_risk` or `layering` had already created a `KalshiTradeIntent`, the last step before execution still looked like this:

1. producer enqueues the trade intent
2. execution engine waits on its source queue
3. execution engine wakes up and dequeues the intent
4. execution finally enters `_handle_trade_intent(...)`

That queue hop was not the largest latency source in the system, but it was still an avoidable scheduler hop right before fire.

### New behavior

When all of the following are true:

1. execution mode is `LIVE`
2. `enable_direct_trade_intent_handoff_in_live_mode=True`
3. the trade-intent source supports callback subscription

the execution engine now registers a direct trade-intent callback instead of subscribing to the source queue.

The selection logic is in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:737), and start-up registration happens in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:634).

The callback entry point is [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:763). It immediately schedules an execution task instead of waiting for `_consume_loop()` to wake up on a queue.

The direct-consume wrapper is in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:773).

### Why I did not execute inline on the producer loop

I did not make `signal_risk` or `layering` directly `await _handle_trade_intent(...)`.

That would reduce one more tiny hop, but it would also make upstream producer loops pay for:

- reservation-claim work
- execution state publication
- live submit preparation
- REST order submission

That is too risky because it couples execution latency back into the signal/layering consumers.

Instead, the direct callback only schedules the execution task immediately. That preserves the main goal, which is to remove the intermediate source-queue wakeup before execution starts, without forcing producer loops to block on live submit work.

### Why serialized execution was preserved

The old queue consumer implicitly serialized execution intent handling. That matters for:

- reservation claiming
- local execution state transitions
- portfolio/accounting consistency
- deterministic behavior under bursts

I preserved that behavior by serializing direct callback tasks inside execution with an execution-side lock. So this is not a “fire all intents concurrently” design. It is a lower-latency entrance into the same serialized execution semantics.

### What changed in the producers

`signal_risk` now publishes approved trade intents to both callback subscribers and queue subscribers. The callback publication happens in [signal_risk.py](/c:/prediction-market-analysis/src/live/kalshi/signal_risk.py:2237).

`layering` now uses a dedicated `_emit_trade_intent(...)` helper so the same emitted intent reaches both callback subscribers and queue subscribers. That helper is in [layering.py](/c:/prediction-market-analysis/src/live/kalshi/layering.py:227).

### Dedicated live runner behavior

The dedicated bagged-lasso live runner now enables the direct-handoff mode by default in [run_kalshi_bagged_lasso_live.py](/c:/prediction-market-analysis/scripts/run_kalshi_bagged_lasso_live.py:136).

That means the specialized runner now opts into all three fast-path cuts:

1. feed-based pre-submit checks for immediate orders
2. background JSONL logging
3. direct live trade-intent handoff into execution

## Change 4: `NO` IOC Probe Orders Get an Extra Execution Cushion

### Why this was added

After the first three fast-path cuts, the IOC diagnostic run improved a lot overall, but the remaining miss-rate was highly asymmetric:

- `YES` was filling reasonably well
- `NO` was still missing too often

That pattern suggested the remaining problem was no longer “the whole fire path is too slow.” Instead, it looked more like `NO` probe orders were still too tight relative to live conditions.

Rather than loosening every live order, I added a narrow execution-layer rule:

- `YES` keeps its current behavior
- `NO` opening probes in immediate-match mode can bid a few cents above the model’s maximum acceptable entry price

This is an execution tuning rule, not a model change. The signal engine still generates the same trade intent. The extra cushion is applied only at execution time.

### What changed

The execution config now has:

- `no_probe_immediate_limit_cushion_cents`

This was added in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:120), validated in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:137), and loaded from environment in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:176).

The actual limit-adjustment logic lives in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:935).

It only applies when all of the following are true:

1. the order is an opening probe
2. the order requires immediate matching
3. the side is `NO`
4. the configured cushion is greater than `0`

If those conditions are met, execution increases the submitted limit price by the configured number of cents, capped at `99`.

### What does not change

This rule does not affect:

- `YES` orders
- non-probe tranches
- passive `good_till_canceled` probe orders
- signal generation
- model edge calculations

It is intentionally narrow so we can learn whether `NO` fill problems are mostly about aggressiveness without blurring the rest of the strategy.

### Logging and observability

Because this change intentionally allows execution to submit a limit that differs from the model limit, I added explicit diagnostics:

- `model_limit_price_cents`
- `execution_limit_adjustment_cents`

Those fields are now written into the pre-submit and submit logs in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:965), [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:1460), and [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:1482).

That makes the next live run much easier to interpret. We will be able to tell:

- what the model originally allowed
- how much extra execution cushion was used
- whether the extra aggressiveness translated into better `NO` fill rate

### Dedicated live runner behavior

The dedicated bagged-lasso live runner now sets:

- `no_probe_immediate_limit_cushion_cents=3`

in [run_kalshi_bagged_lasso_live.py](/c:/prediction-market-analysis/scripts/run_kalshi_bagged_lasso_live.py:137).

I chose `3` cents because the observed live IOC gap was large enough that `+1` would likely be too small to be conclusive, while `+3` is still modest enough to remain an execution test rather than a wholesale strategy rewrite.

## Logging and Telemetry Improvements

Because the immediate-order fast path can now come from different data sources, I also extended the logged payloads so later analysis can distinguish:

- feed-based pre-submit decisions
- REST-orderbook-based pre-submit decisions
- disabled checks

The execution logs now include:

- `check_source`
- `pre_submit_check_source`

Those fields were wired into the log payload in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:1358) and into the check logging payload builder in [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py:869).

This matters because otherwise later performance analysis would blur together:

- “the local feed already showed the move”
- “the REST book disagreed with the feed”
- “we skipped the network check entirely”

## Test Coverage Added

I added execution tests for the new immediate-order behavior in [test_kalshi_execution.py](/c:/prediction-market-analysis/tests/test_kalshi_execution.py:1049) and [test_kalshi_execution.py](/c:/prediction-market-analysis/tests/test_kalshi_execution.py:1151).

The new tests verify:

1. immediate live orders can use `feed_quote` as the pre-submit check source
2. no REST orderbook call is made on that path
3. feed-based limit-moved-away decisions can cancel locally before order submission

I also updated two existing shadow-mode tests in [test_kalshi_execution.py](/c:/prediction-market-analysis/tests/test_kalshi_execution.py:1656) and [test_kalshi_execution.py](/c:/prediction-market-analysis/tests/test_kalshi_execution.py:1743).

Those tests were not logically broken by the live-path change. They became timing-sensitive because the new async logger removed some incidental delay from the event loop. I adjusted them so they wait more explicitly for the intended shadow-requote timing instead of relying on the old synchronous logging slowdown.

I also added a direct-handoff execution test in [test_kalshi_execution.py](/c:/prediction-market-analysis/tests/test_kalshi_execution.py:1997). That test verifies that, in live mode with direct handoff enabled:

1. the execution engine does not subscribe to the source queue
2. the source callback path is registered instead
3. a published trade intent still reaches a filled execution state end to end

I also added asymmetric execution tests for the new `NO`-probe cushion in [test_kalshi_execution.py](/c:/prediction-market-analysis/tests/test_kalshi_execution.py:1285) and [test_kalshi_execution.py](/c:/prediction-market-analysis/tests/test_kalshi_execution.py:1395). Those tests verify:

1. `NO` IOC probe orders get the configured extra submitted limit
2. the logs expose both the model limit and the execution adjustment
3. `YES` IOC probe orders remain unchanged even when the cushion config is enabled

## Validation Performed

I ran:

```text
pytest tests/test_kalshi_execution.py tests/test_kalshi_signal_risk.py tests/test_kalshi_layering.py tests/test_kalshi_live_collector.py -q
```

Result:

```text
89 passed
```

That gives good coverage around:

- execution hot path
- collector interaction
- layering behavior
- signal/execution coordination

## Operational Effect

### What should be faster now

For immediate live orders on the dedicated runner:

- one REST call has been removed from the critical path
- file appends no longer block the event loop directly
- the last source-queue wakeup before execution fire has been removed
- `NO` opening probes can now submit with a small extra IOC cushion when configured

So the trigger-to-submit path should now spend less time in:

- synchronous disk I/O
- extra network round trips before fire
- waiting for the execution consumer to wake up on a trade-intent queue

### What should stay the same

The following behaviors are still intentionally preserved:

- live orders are still submitted via signed REST `POST /portfolio/orders`
- passive probe behavior remains distinct from immediate-match behavior
- execution still logs every important step to JSONL
- execution handling is still serialized for correctness
- the upstream collector -> feature -> scorer -> signal -> layering stages still remain separate
- `YES` orders are not loosened by the new `NO`-specific cushion rule

## Important Limits of These Changes

These changes are meaningful, but they are not a full low-latency redesign.

They do not address:

1. most of the upstream serialized queue hops across the multi-stage pipeline
2. REST order entry itself
3. exchange-side disappearing liquidity
4. the possibility that `NO` still needs a different order style such as short-TTL `GTC`
5. potential value from a separate FIX-based order-entry stack
6. broad live strategy selection quality

So if live performance is still unsatisfactory after this, the next likely bottleneck is not “logging is too slow” or “the extra REST book check is too slow.” The next likely bottleneck is the architecture between trigger and submit, or the fact that REST order entry itself is still the transport.

## Why These Changes Were Chosen First

These were the best first moves because they have a strong ratio of benefit to risk:

### High benefit

- remove one network round trip from immediate live submit
- reduce event-loop blocking everywhere that matters
- remove the last source-queue wakeup before execution fire
- make the remaining `NO`-side IOC miss-rate testable without loosening all sides and all tranches

### Low to moderate risk

- no change to model logic
- no change to signal logic
- no change to settlement/accounting semantics
- limited scope in the dedicated live runner
- explicit test coverage for the new path

By contrast, collapsing more of the upstream queue chain or redesigning transport around FIX would be larger changes with much more surface area.

## Summary

I made four concrete execution-path changes:

1. immediate live orders can now use feed-based pre-submit checks instead of paying for a fresh REST orderbook call
2. JSONL event logging now happens through a shared background writer instead of synchronous file appends on the event loop
3. live execution can now receive trade intents through a direct callback handoff instead of waiting on a trade-intent queue
4. `NO` opening IOC probes can now add a small execution-only limit cushion

I made these changes because the previous design was correct and debuggable, but too expensive on the hot path for fast live markets.

I intentionally did not yet change:

- most of the upstream multi-stage queue architecture
- REST order placement itself
- broader strategy selection rules

So this work should be understood as a fast-path optimization pass, not a full execution-stack rewrite.
