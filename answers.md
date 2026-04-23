# Project-Grounded Answers to the 20 Trading-System Questions

This document answers the questions using only evidence that exists inside this repository as of `2026-04-17`. Where the codebase or local artifacts do not fully answer a question, I say that explicitly instead of guessing.

Evidence base used:
- Core runtime code in `src/live/kalshi/*`
- Runner entrypoints in `scripts/*`
- Existing system documentation in [architecture.md](/c:/prediction-market-analysis/architecture.md)
- Model artifacts under `artifacts/kalshi/kxbtc15m_*`
- Archived live/paper/shadow outputs under `output/live/*`
- Archived performance reports under `artifacts/kalshi/performance_reports/*`

## A. Underlying Market & Data

## 1. Spot BTC price source at decision time
The scorer does **not** have an external low-latency BTC spot feed from Coinbase, Binance, Bybit, CME, or any other non-Kalshi venue in the current live path. The runtime data path is Kalshi-native:

- The collector subscribes to Kalshi public WebSocket channels `trade`, `ticker`, and `market_lifecycle_v2`.
- The collector can also refresh a market by ticker via Kalshi REST.
- The only additional cross-market context I found is Kalshi's own hourly `KXBTCD` series, not an external spot venue.

So, at decision time, the system infers market state from:
- the current KXBTC15M Kalshi quote/trade stream
- optional Kalshi hourly context (`KXBTCD`)

There is no code for external BTC venue ingestion, no external venue client, and no external WSS/REST price adapter in the live scorer path. Because of that, the right answer is: **the model relies on Kalshi's own market stream rather than a direct external spot BTC feed**.

On staleness, the code measures quote freshness as `quote_age_seconds`, and the signal layer blocks candidates on `stale_quote` once quote age exceeds the configured threshold. The default threshold in `SignalConfig` is `3.0` seconds. In practice, the system is designed to reject stale Kalshi quotes rather than fuse in an external BTC spot oracle.

Primary references:
- `src/live/kalshi/collector.py`
- `src/live/kalshi/client.py`
- `src/live/kalshi/features.py`
- `src/live/kalshi/feature_engine.py`

## 2. Strike and time-to-expiry features
`tau_minutes` is explicitly fed into the model. Short-horizon realized-volatility style features are also explicit. But raw `log(S/K)` is **not** explicit in the current feature set.

What is explicit:
- `tau_minutes`
- `price_volatility_30s`
- `price_volatility_120s`
- `price_volatility_300s`
- rolling price returns, trade counts, signed flow, yes-taker share
- `z_implied`, which is derived from Kalshi market probability

What is not explicit:
- external spot price `S`
- explicit strike `K`
- raw `log(S/K)`

So the current model gets:
- **time to expiry explicitly**
- **short-horizon realized volatility explicitly**
- **moneyness only indirectly**, through Kalshi probability transformed into `z_implied`, plus the KXBTC15M-vs-KXBTCD context features when hourly context is enabled

Primary references:
- `src/live/kalshi/features.py`
- `src/live/kalshi/types.py`
- `src/live/kalshi/feature_engine.py`

## 3. Options-market context
I found **no ingestion of Deribit or similar crypto-derivatives context** in the current system. Specifically, I found no references to:

- Deribit
- DVOL
- IV surface / ATM IV
- 25-delta risk reversal
- perp funding
- perp basis
- skew feeds

The current architecture is Kalshi-centric. The only broader market context I found is the optional Kalshi `KXBTCD` hourly series used as a higher-timeframe context market.

Why these were excluded: the repo does not document an explicit reason. The most defensible inference is that the system was intentionally kept within a Kalshi-only data perimeter, and external derivatives data connectors were never built.

Primary references:
- `src/live/kalshi/collector.py`
- `src/live/kalshi/feature_engine.py`
- `src/live/kalshi/features.py`
- [architecture.md](/c:/prediction-market-analysis/architecture.md)

## B. Model Performance & Live Results

## 4. Current live/shadow PnL reality
There is **no single built-in 30/90-day production scorecard** in the repo that reports Sharpe, hit rate, realized-vs-expected edge, and asymmetry across all modes. So the best honest answer is a rough snapshot from archived runs and reports.

What the repo does support today:
- win rate and net PnL by run in `report.md`
- bucketed PnL
- execution fill metrics
- offline `policy.json` metrics like `return_pct` and `max_drawdown_dollars`

What it does **not** support today:
- live-mode Sharpe
- a standardized 30-day / 90-day by-mode performance dashboard
- realized edge vs expected edge rollup across all runs

Recent live snapshot from inspected bagged-lasso reports:
- `kalshi_bagged_lasso_live_fastpath_ioc_test_v3_local_patched`: `+$9.93`, but prior analysis shows this was dominated by one large late `NO` add rather than broad robust edge
- `kalshi_bagged_lasso_live_structural_regime_v1_local`: `-$2.36`, `43.75%` win rate
- `kalshi_bagged_lasso_live_structural_regime_v2_20260414_191118_local`: `+$0.61`, `100%` win rate, but only `2` settled rows
- `kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438_local`: `-$4.00`, `46.15%` win rate

Recent paper/shadow snapshot from inspected runs:
- `kalshi_4model_shadow_preview_20260416_013354`: about `+$7.72` total settled PnL across the four paper-mode models
- `kalshi_4model_shadow_preview_edge2_nobuckets_20260416_104542`: about `+$16.82` total settled PnL across the four paper-mode models
- `kalshi_bagged_lasso_live_local_shadow_researchparity_20260408`: `+$9.19`, `66.5%` win rate, `430` settled rows
- `shadow_layered_no_regime_edge2`: `+$30.75`, `76.4%` win rate, but this belongs to an older non-live regime and should be treated carefully when comparing to current, stricter execution realism

Across the local archive span that is actually present in this workspace, the rough mode-level rollup is:
- paper: about `+$24.54`
- shadow: about `+$618.58`
- live: recent inspected bagged-lasso reports sum to about `+$4.18`

That rollup needs a major warning label:
- the local archive span here is shorter than a true 90-day production history
- the shadow total is inflated by older runs from before the later non-live fill-realism fixes
- so the recent live reports are a better reality check than the raw historical shadow total

Win/loss asymmetry is real. One clear sign is that the `+$9.93` live run had a low raw win rate but positive PnL because one trade carried the run. So the live system currently shows skewed outcomes rather than clean, stable, high-frequency edge.

The local archive does not appear to contain a reliable full 90-day performance history, and older shadow totals are not fully trustworthy because the non-live fill model was fixed later.

Primary references:
- `artifacts/kalshi/performance_reports/kalshi_bagged_lasso_live_fastpath_ioc_test_v3_local_patched/report.md`
- `artifacts/kalshi/performance_reports/kalshi_bagged_lasso_live_structural_regime_v1_local/report.md`
- `artifacts/kalshi/performance_reports/kalshi_bagged_lasso_live_structural_regime_v2_20260414_191118_local/report.md`
- `artifacts/kalshi/performance_reports/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438_local/report.md`
- `artifacts/kalshi/performance_reports/kalshi_bagged_lasso_live_local_shadow_researchparity_20260408/report.md`
- `output/live/kalshi_4model_shadow_preview_20260416_013354`
- `output/live/kalshi_4model_shadow_preview_edge2_nobuckets_20260416_104542`

## 5. Which model is actually deployed?
For **dedicated live trading**, the repo is clearly built around **bagged_lasso** right now.

Evidence:
- The dedicated live entrypoint is `scripts/run_kalshi_bagged_lasso_live.py`
- It resolves the exact bagged-lasso artifact bundle and live policy envelope
- Recent live production outputs under `output/live/kalshi_bagged_lasso_live_*` are bagged-lasso runs

For **multi-model non-live previews**, the repo also supports running these four regularized families together:
- `lasso`
- `elastic_net`
- `bagged_lasso`
- `linear_svm`

LightGBM exists in the training code and artifact bundles, but I did not find it running in the current multi-model paper preview command path.

Why bagged-lasso was chosen for live: the repo does not explicitly justify the business decision. In fact, the latest offline artifacts suggest that some other families, especially LightGBM and Linear SVM, have stronger offline `return_pct` than bagged-lasso. So the best project-grounded answer is that **bagged-lasso is the current operational live choice, but the repo does not prove it was chosen because it was the strongest offline model**.

Primary references:
- `scripts/run_kalshi_bagged_lasso_live.py`
- `scripts/run_kalshi_multi_model_execution_engine.py`
- `scripts/run_kalshi_regularized_execution_engine.py`
- `artifacts/kalshi/kxbtc15m_bagged_lasso/latest/policy.json`
- `artifacts/kalshi/kxbtc15m_linear_svm/latest/policy.json`
- `artifacts/kalshi/kxbtc15m_lightgbm/latest/policy.json`

## 6. Calibration decay
The training stack explicitly uses **Platt scaling** for calibration, and it stores calibration artifacts to disk. But I found **no ongoing live calibration drift monitor**.

What exists:
- `calibration.py` implements `PlattCalibration`
- `offline_training.py` fits and saves `calibration.json`
- training outputs report validation/test log loss
- offline training can generate calibration bucket summaries

What I did not find:
- scheduled retraining
- automatic recalibration cadence
- live Brier-score tracking
- PSI tracking
- day-over-day production calibration dashboards

Bagged-lasso's latest artifact summary includes:
- calibrated validation log loss about `0.5116`
- calibrated test log loss about `0.5124`

That tells us training-time calibration was measured. It does **not** tell us whether production calibration drifts over days or weeks, because the runtime does not appear to log or report that.

Primary references:
- `src/live/kalshi/calibration.py`
- `src/live/kalshi/offline_training.py`
- `artifacts/kalshi/kxbtc15m_bagged_lasso/latest/summary.json`

## C. Execution Realism & Order Flow

## 7. Paper/shadow vs. live divergence
I did **not** find a formal "same event stream, same policy, paper vs live" A/B study in the repo.

The archive is close to supporting it, because the system records:
- signal decisions
- claimed intents
- execution lifecycle events
- settlements

But it is not turnkey today, because:
- live and non-live runners often use different policy envelopes
- older shadow/paper runs were produced before the non-live fill realism fixes
- there is no built-in paired comparison report

So the answer is:
- **No formal paired paper-vs-live divergence study is present**
- **The archive could support one with custom analysis, but not automatically today**

This matters because a known execution-realism gap existed historically in paper/shadow, and only later did the repo change non-live fills to use current executable price plus a refreshed edge check.

Primary references:
- `src/live/kalshi/execution.py`
- `output/live/*`
- `artifacts/kalshi/performance_reports/*`
- [architecture.md](/c:/prediction-market-analysis/architecture.md)

## 8. Fill assumptions in paper/shadow
Paper and shadow mode are now much better than before, but they still do **not** model full market microstructure realism.

What they do now:
- re-check the current executable price before filling
- use fresh quote-aware fill logic
- cancel if the quote is missing, stale, crossed, or beyond the refreshed acceptable limit
- if price moved beyond the original limit, recompute a new acceptable cap using `maintain_edge_cents` and fill only if edge still supports the moved price

What they still do **not** do:
- model depth beyond the executable price
- model partial fills
- model queue position
- model GTC queue dynamics

In other words, simulated non-live execution is now **price-aware**, but it is still effectively a **full-size immediate taker fill model**, not a depth/queue simulator.

Primary references:
- `src/live/kalshi/execution.py`
- [architecture.md](/c:/prediction-market-analysis/architecture.md)

## 9. Typical Kalshi book depth on KXBTC15M
The system does record top-of-book size in live pre-submit checks, so this can be answered from actual logs.

From sampled live `pre_submit_orderbook_check` / `submit_requested` data in recent live runs, I measured:
- minimum observed top-of-book contracts: `0`
- median: about `109`
- 90th percentile: about `1,582`
- maximum observed sample: about `3,117`

So on active KXBTC15M strikes, top-of-book can be thin at times but is often materially larger than a 1-contract order.

Current system size is small:
- most dedicated live orders are `1` contract
- some later-layer `NO` adds reached `7`, `8`, or `14` contracts before the newer `$1` same-side cap was added

That means:
- market impact modeling is **not** the main issue at today's typical live size
- it **would** become important if the system scaled toward `50+` contracts routinely
- the repo does not show any credible history of filling `500` or `5000` contracts on KXBTC15M

Primary references:
- `src/live/kalshi/execution.py`
- `output/live/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438/execution/bagged_lasso/production/2026-04-14/events.jsonl`
- `artifacts/kalshi/performance_reports/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438_local/report.md`

## 10. Latency budget
Latency is measured, but only partially and not in one single "decision-to-exchange-ack" metric.

What is logged:
- `ticker_update_to_submit_requested_ms`
- `received_at_to_submit_requested_ms`
- `pre_submit_orderbook_roundtrip_ms`

From sampled recent live runs:
- median `received_at_to_submit_requested_ms` was roughly `12` to `33` ms
- median `ticker_update_to_submit_requested_ms` was roughly `717` to `898` ms
- some huge outliers existed in the retry-oriented runs because submission could happen long after the original quote event

Interpretation:
- the raw event-to-submit-request path can be fast
- but orchestration and retry timing often dominate more than pure EC2-to-Kalshi network latency
- yes, latency is a real constraint for fleeting IOC liquidity, especially if a REST orderbook check is in the way

In some of the fast-path runs, the REST pre-submit orderbook check was intentionally skipped for immediate orders, which reduced round-trip overhead.

Primary references:
- `src/live/kalshi/execution.py`
- `LIVE_EXECUTION_AUDIT.md`
- `LIVE_EXECUTION_FASTPATH_CHANGES.md`
- `output/live/kalshi_bagged_lasso_live_fastpath_ioc_test_v3/execution/...`
- `output/live/kalshi_bagged_lasso_live_structural_regime_v1/execution/...`
- `output/live/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438/execution/...`

## D. Risk Management Reality

## 11. Actual sizing in production
The currently deployed dedicated live policy is **fixed-contract sizing**, not Kelly.

Specifically, the dedicated live runner sets:
- `contracts_per_order = 1`
- `capital_pct_per_order = None`
- `kelly_fraction_multiplier = None`
- `kelly_fraction_cap_pct = None`

There is a Kelly sizing path in `signal_risk.py`, but it is **not** what the dedicated live runner is using right now.

Additional live guardrails:
- reserve-cash gating
- same `ticker + side` exposure cap
- per-order eligibility/risk checks

There is no portfolio correlation model across overlapping strikes. The controls are mostly:
- cash reserve
- per-ticker-side exposure cap
- thesis-level gating
- window/tranche logic

Primary references:
- `src/live/kalshi/signal_risk.py`
- `scripts/run_kalshi_bagged_lasso_live.py`

## 12. Concurrent position reality
The repo does **not** maintain a clean historical metric for "maximum simultaneous positions across all strikes and windows." So this answer has to be partial.

What I can say confidently:
- recent live reports often show only a few open positions at report time, such as `0` to `3`
- before the new same-side `$1` cap, the system could stack materially on one market; in one live run it accumulated `7` and `8` contract `NO` layers on the same ticker, for about `16` contracts total on that market
- the dedicated live runner now enforces `max_ticker_side_exposure_dollars = 1.0`, so current intended behavior is much tighter than those older runs

What I cannot answer from the repo:
- the true all-time peak simultaneous dollar exposure across the whole live book at one instant

So the right takeaway is:
- historical live exposure was small in absolute dollars, but could still become concentrated in one ticker before the newer cap
- current policy is intentionally trying to keep that much tighter

Primary references:
- `artifacts/kalshi/performance_reports/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438_local/report.md`
- `src/live/kalshi/signal_risk.py`
- `scripts/run_kalshi_bagged_lasso_live.py`

## 13. Worst-case drawdown experienced
There are two different answers: offline model-policy drawdown, and observed recent live/session losses.

Offline artifact drawdowns in the latest model bundles:
- bagged_lasso: max drawdown about `$3.88`
- lasso: about `$8.07`
- elastic_net: about `$8.49`
- linear_svm: about `$7.41`
- lightgbm: about `$9.45`

Recent live/session losses I found in reports:
- `kalshi_bagged_lasso_live_target_fill_v1_local`: about `-$4.71`
- `kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438_local`: about `-$4.00`

What caused the worst recent live losses:
- weak low-price buckets
- bad `NO` late layers in some runs
- earlier over-aggressive retry behavior before the bounded target-fill changes

Did controls behave as designed?
- Per-order, yes: the system applied its configured rules
- Portfolio-level, no kill switch existed, so there was nothing to stop the session once aggregate drawdown grew

Primary references:
- `artifacts/kalshi/kxbtc15m_*/latest/policy.json`
- `artifacts/kalshi/performance_reports/kalshi_bagged_lasso_live_target_fill_v1_local/report.md`
- `artifacts/kalshi/performance_reports/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438_local/report.md`
- [architecture.md](/c:/prediction-market-analysis/architecture.md)

## E. Operational Infrastructure

## 14. Deployment topology
The runtime is a **single Python asyncio process** that hosts the collector, signal engine, layering engine, and execution engine together.

Where it runs:
- locally for paper/shadow/research work
- on EC2 for live trading in the documented rollout path

How it is supervised:
- most evidence points to **manual `tmux` sessions**, not systemd or Docker
- I did not find Dockerfiles, compose files, Kubernetes manifests, or systemd unit files for the trading bot

So the most accurate deployment summary is:
- **single-process async Python**
- **often run in tmux on EC2 for live**
- **manually operated, not heavily orchestrated**

Primary references:
- `scripts/run_kalshi_bagged_lasso_live.py`
- `scripts/run_kalshi_multi_model_execution_engine.py`
- `RunBotInEC2.md`
- `Live Rollout Plan for Exact-Match Bagged-Lasso Layering`
- [architecture.md](/c:/prediction-market-analysis/architecture.md)

## 15. What happens on failure today
If the public WSS disconnects:
- the collector logs the disconnect
- it schedules reconnect with backoff
- while disconnected, quotes can go stale and the signal layer will block new trades on `stale_quote`

If the private execution WSS disconnects:
- the execution engine logs the disconnect
- it retries
- it requests reconciliation

If the process crashes with open positions:
- the positions remain at Kalshi
- there is no automatic flattening
- there is no portfolio kill switch
- on restart, the execution engine refreshes balance and positions, and recovers recent orders

How you find out today:
- console output
- tmux session state
- JSONL logs
- dashboard views
- generated reports

I did **not** find a formal incident log or postmortem record in the repo, so I cannot honestly walk through a single confirmed, documented "last real incident" from project records.

Primary references:
- `src/live/kalshi/collector.py`
- `src/live/kalshi/execution.py`
- [architecture.md](/c:/prediction-market-analysis/architecture.md)
- `DataStreamRecordOverview.md`

## 16. Kalshi account structure
The codebase clearly supports Kalshi subaccounts. The dedicated live logs I inspected show live trading on:
- `subaccount = 0`

I did not find evidence in the repo that:
- multiple live subaccounts are being traded simultaneously
- a corporate account structure is being used
- the system is near Kalshi daily/per-market limits

So the strongest project-grounded answer is:
- **single live subaccount is clearly used**
- broader account-structure details are not documented in the repo

Primary references:
- `src/live/kalshi/execution.py`
- `scripts/run_kalshi_bagged_lasso_live.py`
- `output/live/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438/execution/bagged_lasso/production/2026-04-14/events.jsonl`

## F. Strategy & Research

## 17. What is known to work vs. not work
What seems most consistently bad in recent live evidence:
- very low-price buckets, especially `0-10` and `10-20`
- downtrend regime in at least one strong shadow report
- low-volume/choppy conditions, which were bad enough that a dedicated `low_liquidity_chop` gate was later added

Concrete examples:
- `kalshi_bagged_lasso_live_structural_regime_v1_local`: `0-10` and `60-70` were losing
- `kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438_local`: `10-20` and `50-60` were losing
- `kalshi_bagged_lasso_live_local_shadow_researchparity_20260408`: `downtrend` was strongly negative

What seems to have worked at times:
- some mid-price YES/NO buckets in older shadow runs
- some late cheap `NO` adds in individual live runs

But I would be careful calling anything "most consistently profitable" yet, because:
- some positive live runs were dominated by a single large winner
- some older shadow runs were generated before later execution-realism fixes
- the repo does not yet contain a clean stability study that says "this bucket wins across modes and over time"

Primary references:
- `artifacts/kalshi/performance_reports/kalshi_bagged_lasso_live_structural_regime_v1_local/report.md`
- `artifacts/kalshi/performance_reports/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438_local/report.md`
- `artifacts/kalshi/performance_reports/kalshi_bagged_lasso_live_local_shadow_researchparity_20260408/report.md`
- `src/live/kalshi/signal_risk.py`

## 18. Layering engine empirical value
I found **no formal archived A/B test** that cleanly compares:
- the current path-dependent layering engine
- versus a simple single-tranche strategy

The repo is much closer to being able to do that now than it was before, because the archive contains:
- `thesis_id`
- tranche windows (`10m`, `5m`, `4m`, `3m`)
- target execution summaries
- filled vs unfilled targets

But I did not find a completed report that says, for example:
- "layering added X bps of PnL"
- "layering reduced drawdown by Y"

So the best answer is:
- **No, I did not find a clean A/B result**
- **Yes, the archive now has enough structure that this comparison should be buildable**

Primary references:
- `src/live/kalshi/layering.py`
- `artifacts/kalshi/performance_reports/*/report.md`
- [architecture.md](/c:/prediction-market-analysis/architecture.md)

## 19. Research vs. production parity
There is no formal parity report in the repo today that says:
- same signal set
- theoretical research PnL
- paper/shadow execution PnL
- live execution PnL

Historically, there was definitely a material gap, because non-live fills used to be too optimistic or stale. That gap is the reason several execution-realism fixes were added later:
- non-live now fills at current executable market price
- non-live re-checks edge if price moved
- settlement repair/backfill tools were added for stale open-state issues

So my project-grounded view is:
- **older research/shadow PnL is not a reliable parity benchmark**
- **current parity should be much tighter after the execution fixes**
- **but there is still no formal matched-signal parity study in the repo**

Primary references:
- `src/live/kalshi/execution.py`
- `scripts/backfill_kalshi_live_execution_settlements.py`
- `artifacts/kalshi/performance_reports/*`
- [architecture.md](/c:/prediction-market-analysis/architecture.md)

## G. Objectives & Constraints

## 20. What "profitable" means here, and the operating constraints
The repo does **not** encode your human business objective. I could not find a file that says whether success means:
- monthly absolute return
- Sharpe
- Kelly growth
- max drawdown cap
- hobby account growth
- multi-year product buildout

What the research stack does encode is an implicit optimization preference:
- offline policy search stores `net_pnl_dollars`
- `max_drawdown_dollars`
- `return_pct`
- `log_loss`
- the offline policy objective is `total_pnl / max(1.0, max_drawdown)`

So the current system implicitly values:
- more PnL
- with less drawdown
- and secondarily cares about predictive quality via log loss

Hard runtime constraints that are actually present:
- reserve cash requirement
- fixed 1-contract live sizing in the dedicated runner
- `max_ticker_side_exposure_dollars = 1.0` in the newer dedicated live policy
- regime filters
- bucket/combo filters when enabled
- low-liquidity-chop gate when enabled
- no session-level kill switch

What is **not** encoded:
- a maximum allowed session drawdown before shutdown
- tax or regulatory workflow constraints
- a long-horizon portfolio mandate

So the honest answer is: **the repo has technical constraints, but not your actual business objective.** If recommendations are supposed to optimize for a specific account size, drawdown tolerance, or time horizon, that information still has to come from you.

Primary references:
- `src/live/kalshi/offline_training.py`
- `src/live/kalshi/signal_risk.py`
- `scripts/run_kalshi_bagged_lasso_live.py`
- [architecture.md](/c:/prediction-market-analysis/architecture.md)
