# Live Execution Audit

Date: 2026-04-12

## Scope

This report is based on:

- inspection of the live Kalshi stack in this repo
- the reported live results from `kalshi_bagged_lasso_live_gtc_ttl_edge2_v2`
- the current production wiring in `scripts/run_kalshi_bagged_lasso_live.py`

Goal: explain why `1`-contract orders can still miss or underperform, identify what is actually event-driven vs polled, and rank the highest-ROI fixes.

## Executive Summary

The main problem is not that the bot is written in Python.

The main problem is that the live path is still dominated by:

1. serialized queue hops across several asyncio stages
2. an extra REST round trip before every live submit
3. synchronous file logging on the hot path
4. broad live signal selection that is now getting filled often enough to expose mediocre entries

For the old IOC setup, a `GET orderbook` plus `POST order` path is enough to make `1`-contract liquidity disappear before your order lands, even with EC2-to-Kalshi network latency around `30 ms`.

For the current GTC setup, the system is no longer badly broken on execution. The evidence says it is now reaching market often enough, but passive probes are still getting filled at prices or in buckets that are not strong enough to preserve the shadow edge.

## Current Live Path

With layering enabled, the actual path is:

1. public websocket message arrives in the collector
2. collector publishes a `KalshiTickerUpdate`
3. feature engine consumes that update and publishes a `KalshiFeatureUpdate`
4. scorer consumes features and publishes a score update
5. signal engine consumes scores and publishes a signal decision
6. layering engine consumes signal decisions and, when appropriate, creates a trade intent
7. execution engine consumes the trade intent
8. execution engine does a pre-submit REST orderbook check
9. execution engine submits the order via REST
10. private websocket receives order/fill/cancel updates

Relevant wiring:

- layered live run disables direct auto-reservation in `scripts/run_kalshi_bagged_lasso_live.py:317-318`
- execution consumes intents from the layering engine in `scripts/run_kalshi_bagged_lasso_live.py:402-406`
- public market data uses one websocket in `src/live/kalshi/collector.py:579-629`
- private execution updates use one websocket in `src/live/kalshi/execution.py:1713-1768`

## What Is Already Event-Driven

The normal trigger-to-fire pipeline is mostly event-driven.

There is no polling loop that waits for a signal to become tradable. Instead, each stage blocks on its upstream queue:

- collector -> feature engine: `src/live/kalshi/feature_engine.py:261-279`
- feature engine -> scorer: `src/live/kalshi/scorer.py:810-827`
- scorer -> signal engine: `src/live/kalshi/signal_risk.py:1097-1106`
- signal engine -> layering engine: `src/live/kalshi/layering.py:289-299`
- layering engine -> execution engine: `src/live/kalshi/execution.py:1067-1076`

So the core trigger chain is already event-driven.

## Where Polling Still Exists

There are four polling-style loops in the live stack:

1. metadata refresh
   - `src/live/kalshi/collector.py:569-575`
   - default interval: `300s`
   - purpose: refresh market metadata and subscription set

2. reservation cleanup
   - `src/live/kalshi/signal_risk.py:1110-1116`
   - interval is `min(1.0, max(0.1, reservation_ttl_seconds / 2.0))`
   - with default `reservation_ttl_seconds=5`, cleanup runs every `1s`

3. reconcile loop
   - `src/live/kalshi/execution.py:1797-1814`
   - default interval: `15s`
   - purpose: refresh balances and settle/reconcile state

4. submit retry/reconcile polling
   - `src/live/kalshi/execution.py:1405-1413`
   - only on submit failure/reconciliation path
   - polls every `0.5s` up to `10s`

Important: these polling loops are not the main hot-path delay between signal and order submission. The main hot-path delay is the explicit REST work and logging done right before submit.

## The Real Hot Path Costs

### 1. Pre-submit REST orderbook check

Every live order currently does a REST orderbook fetch before submission:

- `src/live/kalshi/execution.py:903-1009`
- call made via `self._rest_client.get_market_orderbook(...)` at `src/live/kalshi/execution.py:931-935`

That means the live path is:

- signal intent created
- `GET /markets/{ticker}/orderbook`
- log result
- `POST /portfolio/orders`

Order placement itself is REST:

- `src/live/kalshi/client.py:116-120`
- `src/live/kalshi/execution.py:1332-1334`

For fast BTC expiry markets, that extra round trip is a major source of missed liquidity.

### 2. Synchronous file logging on the event loop

The collector, execution engine, and layering engine all use synchronous file writes inside async code:

- collector logger: `src/live/kalshi/collector.py:137-154`
- execution logger: `src/live/kalshi/execution.py:94-111`
- layering logger: `src/live/kalshi/layering.py:29-46`

These writes are not pushed to a background thread. They run directly on the event loop under an async lock.

On the execution hot path, a live order does multiple log writes before and after submit:

- `intent_claimed`
- `pre_submit_orderbook_check`
- `submit_requested`
- `submit_response` or `submit_error`

Relevant section:

- `src/live/kalshi/execution.py:1134-1149`
- `src/live/kalshi/execution.py:1273-1334`

This is not the biggest cost compared with network, but it absolutely adds jitter.

### 3. Serialized queue hops

Each stage is a single async consumer loop. That is simple and reliable, but it also means the pipeline is serialized:

- one collector websocket consumer
- one feature-engine consumer
- one scorer consumer
- one signal consumer
- one layering signal consumer
- one execution consumer

If the system is quiet, this is fine. In a burst, all stages queue behind each other.

## Is Python the Main Problem?

Short answer: no.

Why:

- feature generation is mostly lightweight NumPy math
- scoring is a single-row sklearn-style `predict_proba` or similar call
- none of the model code suggests a heavy CPU bottleneck for one market event
- the hot path is much more network- and I/O-bound than CPU-bound

Relevant scoring path:

- scorer loads the model once at startup in `src/live/kalshi/scorer.py:760-771`
- inference for one update is straightforward in `src/live/kalshi/scorer.py:821-832`

Moving this exact architecture from Python to Rust would not remove:

- the REST orderbook round trip
- the REST order submit round trip
- exchange-side disappearing liquidity
- synchronous disk logging if you keep the same behavior

Rust could help later if you are chasing sub-millisecond internal jitter. That is not the first-order bottleneck in this repo today.

## Is Kernel Bypass Worth It?

No, not for this stack right now.

Kernel bypass only makes sense when:

- you already have a very tight native transport stack
- you are colocated or near-colocated
- your dominant delays are kernel/network-stack overhead rather than exchange round trips and application architecture

That is not the case here. This bot is:

- running over internet links from EC2
- placing orders over REST
- doing JSON, logging, and asyncio queue hops in user space

Kernel bypass would be a massive engineering effort for very little gain unless you also replace the transport layer with FIX/native order entry and heavily redesign the runtime.

## Should You Pin the Process or Tune CPU Affinity?

Low ROI, but not crazy.

CPU pinning can reduce jitter a little on noisy systems, especially if:

- the EC2 instance is small
- other workloads share the box
- the process is bouncing between cores

But it will not explain `10%` IOC fill on a `1`-contract order. It is a polish step, not the core fix.

## Should You Run 100 WebSockets and Race Them?

No.

This is not the right move for Kalshi market data or for this architecture.

Problems with the `100 websocket` idea:

- you multiply duplicate traffic and local processing
- you increase reconnect complexity and state reconciliation
- you may hit exchange/session limits or trigger undesirable behavior
- you still submit the order over one REST path afterward
- you do not change the exchange event ordering itself

More importantly, the bot currently uses:

- one public websocket for market data
- one private websocket for user orders, fills, and positions

That is the right basic shape.

If you want redundancy, the sane version is:

- `1` active public websocket
- optionally `1` backup public websocket in a separate process or instance for telemetry/failover
- `1` private websocket for execution state

Not `100`.

## Is Anti-Jitter Logic Useful?

Yes, but only if aimed at the right layer.

Useful anti-jitter measures here:

- background log writer instead of synchronous hot-path writes
- keep market-data handling and execution on isolated processes
- reduce unnecessary REST checks
- use `uvloop`
- keep the REST client warm and avoid connection churn
- measure queue delay explicitly between stages

Not useful enough right now:

- aggressive multi-websocket racing
- kernel bypass
- a Rust rewrite before transport/architecture cleanup

## The Biggest Architectural Problems Today

### Problem 1: REST orderbook check before submit

This is the single most obvious latency tax in the live path.

For immediate orders, you are doing:

1. websocket-triggered signal
2. local compute
3. REST `GET orderbook`
4. local logging
5. REST `POST order`

If your market is moving fast, the book can change between steps `3` and `5`.

### Problem 2: Hot-path file writes

Even if each write is only a few milliseconds, several writes per order can add enough jitter to matter in a fast market.

### Problem 3: Probe pricing is still too aggressive

Current order payload construction always posts at `intent.max_acceptable_entry_price_cents`:

- `src/live/kalshi/execution.py:509-531`

That means passive GTC probes are not priced as conservative probes. They are posted at the maximum price the strategy is willing to tolerate, which improves fills but can degrade edge quality.

### Problem 4: The live profile is too broad

In the current runner, `research-parity` expands the live filters to:

- edge `2c`
- tau `0-15`
- price band `0-100`
- combo bans disabled
- regime blocks disabled

See `scripts/run_kalshi_bagged_lasso_live.py:91-106`.

That is good for data collection. It is not good as a profit-maximizing live profile.

## Ranking Your Hypotheses

### 1. "Is Python the reason?"

Verdict: mostly no.

### 2. "Is post-logic / too much between trigger and fire the reason?"

Verdict: yes, this is the closest to correct.

The biggest unnecessary work between trigger and fire is:

- queue serialization across multiple stages
- pre-submit REST orderbook check
- synchronous logging

### 3. "Should I get rid of polling and make it all event-driven?"

Verdict: mostly yes in spirit, but the hot path is already event-driven.

What you want to remove is not general polling. What you want to remove is:

- extra request-response validation before submit
- reconcile/retry logic from the hot path
- blocking I/O on the event loop

### 4. "Should I bypass the kernel?"

Verdict: no, not now.

### 5. "Should I pin the process?"

Verdict: maybe later, but low ROI.

### 6. "Should I use 100 websockets and keep the fastest?"

Verdict: no.

### 7. "Should I use anti-jitter logic?"

Verdict: yes, but use application-level anti-jitter, not websocket swarms.

## How Much Do You Poll?

Normal trigger-to-submit path:

- no polling loop
- event-driven through queue consumers
- explicit REST `GET orderbook` before submit
- explicit REST `POST order` to place the order

Background polling:

- metadata refresh every `300s`
- reservation cleanup every `1s` by default
- reconcile every `15s`
- retry lookup every `0.5s` only after submit problems

## Best Improvements, in Priority Order

### Priority 1: Remove the pre-submit REST orderbook check from the immediate-order hot path

For IOC/FOK-style orders, this is the highest-ROI change.

Recommended behavior:

- if the feed quote is fresh enough, submit immediately
- use the orderbook check only for telemetry or for stale-feed fallback
- do not force a `GET orderbook` before every immediate order

This directly removes one network round trip from the fire path.

### Priority 2: Move hot-path logging to a background writer

Replace the current synchronous `JsonlEventLogger.write(...)` pattern with:

- producer pushes a log item to an in-memory queue
- dedicated writer task flushes to disk

This will reduce event-loop jitter in collector and execution.

### Priority 3: Use more conservative passive probe pricing

For `10m` GTC probes:

- do not always post at `max_acceptable_entry_price_cents`
- instead use a passive probe price, such as `reference_price_cents` or `reference + 1`
- keep a short TTL and optionally allow one reprice

This is likely the best way to preserve more of the shadow edge while still getting live fills.

### Priority 4: Split opening criteria from scaling criteria

Opening probes and follow-on adds should not use the same aggressiveness.

Better live structure:

- opening probe: stricter edge, stricter price discipline
- scaling adds: only after thesis exists, can be somewhat looser

### Priority 5: Add end-to-end stage timing

You already log some submit-time metrics. Add stage timestamps for:

- collector receive
- feature publish
- scorer publish
- signal publish
- layering intent create
- execution claim
- submit requested
- submit response
- private websocket order echo

This will let you measure actual internal queue delay rather than guessing.

### Priority 6: Consider `uvloop`

This is a relatively cheap experiment and can modestly tighten asyncio scheduling jitter.

### Priority 7: FIX for medium-term execution

If you want a serious long-term execution stack, FIX is the right next transport project.

It is much more valuable than:

- Rust rewrite first
- kernel bypass first
- websocket swarms

## What I Would Not Do First

Do not do these first:

- rewrite the bot in Rust
- try kernel bypass
- open 100 websocket sessions
- optimize CPU affinity before fixing transport and hot-path I/O

These are lower ROI than removing the extra REST gate and cleaning up the application hot path.

## Recommended Roadmap

### Immediate

1. remove or soft-disable pre-submit REST orderbook check for immediate orders
2. move execution and collector log writes off the event loop
3. add end-to-end timing instrumentation

### Next

1. make GTC probe pricing passive and conservative
2. separate opening thresholds from scaling thresholds
3. narrow the live profile from the broad research-parity settings when running for PnL

### Later

1. `uvloop`
2. process isolation / mild CPU affinity
3. FIX order entry

## Bottom Line

Your diagnosis is directionally right: GTC improved fills, but it did not remove the underlying latency and architecture costs.

However, the fix is not:

- Rust first
- kernel bypass
- 100 websockets

The fix is:

- fewer blocking steps between trigger and submit
- no extra REST validation on the hot path for immediate orders
- background logging instead of event-loop file writes
- better probe pricing and tighter live selection

