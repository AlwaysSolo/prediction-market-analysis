# KXBTC15M LightGBM Training Runbook

## Purpose

This document is the source of truth for the KXBTC15M LightGBM training workflow in this repository.

It exists for two audiences:

1. Humans who need to understand how the offline historical pipeline and the live trading stack fit together.
2. Coding agents who need clear constraints so they do not accidentally break alignment between offline training and live scoring.

This file focuses on the **KXBTC15M-only** model pipeline. The repo has two major Kalshi sections:

- An **offline historical testing and training section** for research, backtesting, and model selection.
- An **online trading section** for live collection, scoring, signal generation, and execution.

The current priority is the offline historical training workflow, but it is intentionally designed so the resulting artifact can be used by the live stack with minimal manual glue.

## Current Scope

### In scope

- Historical KXBTC15M trade-based feature generation
- Chronological train/validation/test splitting by full market ticker
- LightGBM training
- Optuna hyperparameter search
- Platt calibration
- Validation-time trading policy tuning
- Untouched test evaluation
- Walk-forward robustness checks
- Artifact publishing for live scoring
- Phase 2 live quote logging groundwork

### Out of scope for Phase 1

- Historical quote snapshot training
- Order book feature engineering
- Multi-series model training
- Training on a single individual market contract as the final intended production approach
- Production rollout without demo or paper validation

## High-Level Project View

### The repo-level structure

- `src/live/kalshi/` contains the live Kalshi stack.
- `scripts/train_kxbtc15m_lightgbm.py` is the offline training entrypoint.
- `scripts/run_kalshi_*.py` are live runtime entrypoints.
- `artifacts/kalshi/kxbtc15m_lightgbm/` is the new default artifact root for this model family.
- `Gemini scripts/` contains legacy historical scripts and model artifacts. They remain useful for reference and comparison, but the new training path should live under `src/live/kalshi/` and `scripts/`.

### The live stack dependency chain

The live components are intentionally layered:

1. `KalshiMarketDataCollector`
2. `KalshiFeatureStateEngine`
3. `KalshiLightGBMScorer`
4. `KalshiSignalRiskEngine`
5. `KalshiExecutionEngine`

The offline training pipeline mirrors the feature and scoring assumptions of that live chain.

## Core Design Decisions

These are not incidental implementation details. They are deliberate constraints.

### 1. The model is trained at the series level, not one single market

KXBTC15M is a repeating market family. The model should learn from many expiries and then generalize to future expiries.

Training only on one specific market contract is not the target production strategy because it overfits a single regime and a single expiry.

### 2. Splits are chronological and grouped by full market ticker

Rows from the same market ticker must stay in the same split.

Why:

- Random row splits leak near-identical market states across train and test.
- Time-ordered splits better approximate the live deployment problem.
- Grouping by market preserves the reality that many trades within the same contract are highly dependent.

### 3. The live feature contract is the source of truth

Offline training must produce features compatible with the live feature engine.

Why:

- A model that is good offline but impossible to reproduce live is not useful.
- Feature mismatches are one of the easiest ways to create fake backtest performance.

### 4. Phase 1 uses only trade-derived features

The workspace does not currently contain historical time-aligned Kalshi quote snapshots for every trade row.

That means Phase 1 intentionally excludes:

- bid/ask features
- spread features
- open interest deltas
- volume deltas from quote snapshots
- ticker-channel microstructure features that were not logged historically

The collector now logs the public `ticker` channel so those features can be added in the next training cycle once enough data exists.

### 5. Model selection is not based only on log-loss

The objective is live trading profit after costs, not just nice probability metrics.

That means the workflow is staged:

1. Train a probability model and measure calibration/log-loss.
2. Tune the trading policy on validation only.
3. Evaluate final net PnL and drawdown on untouched test data.

### 6. Test data is not for tuning

The test split is the final report card. It is not a tuning surface.

If test results influence hyperparameters, thresholds, or feature definitions, that test split is contaminated and must no longer be treated as final evidence.

### 7. Artifacts are non-destructive and versioned by run

The historical pipeline now writes to a run directory and then copies the deployable subset to `latest/`.

Why:

- repeated experiments should not wipe older work
- comparisons between runs should stay possible
- the live scorer needs a stable default model path

## The Implemented Pipeline

### Main offline entrypoint

`scripts/train_kxbtc15m_lightgbm.py`

This script performs the end-to-end offline training workflow:

1. Resolve input files
2. Load and filter resolved KXBTC15M markets
3. Build the train/validation/test split manifest
4. Build per-market feature cache files
5. Load split datasets
6. Sample up to 2,000,000 train rows for Optuna search
7. Run 40 Optuna trials
8. Retrain the best LightGBM configuration on the full train split
9. Save model artifacts
10. Fit Platt calibration on validation predictions only
11. Tune the trading policy on validation only
12. Evaluate test metrics with the frozen policy
13. Run 4 walk-forward robustness folds
14. Write run summary
15. Publish a `latest/` artifact set for the live scorer

### Main offline implementation module

`src/live/kalshi/offline_training.py`

This module contains the reusable implementation for:

- input resolution
- split creation
- feature cache generation
- LightGBM training
- Optuna search
- calibration
- policy simulation
- walk-forward evaluation
- latest artifact publication

## Data Requirements

### Markets input

Supported formats:

- `.parquet`
- `.csv`

Expected columns:

- `ticker`
- `result`
- `open_time`
- `close_time`

Important notes:

- The pipeline filters to tickers starting with `KXBTC15M-`.
- Only resolved markets with `result` in `{yes, no}` are kept.
- Markets are sorted by `close_time`, then `ticker`.

### Trades input

Supported forms:

- a directory of per-market parquet files: `<trades_path>/<ticker>.parquet`
- a single parquet file that can be filtered by `ticker`

Expected trade columns:

- `trade_id`
- `ticker`
- `count`
- `yes_price`
- `no_price`
- `taker_side`
- `created_time`

### Training window

Rows are only kept if:

- `0 < tau_minutes <= 15`

This means a market can exist in the split manifest but still contribute zero rows if it has no qualifying trades close enough to expiry.

## Feature Contract

### Current feature order

The model currently uses 29 features:

1. `z_implied`
2. `tau_minutes`
3. `price_momentum`
4. `abs_price_momentum`
5. `price_direction`
6. `distance_from_mid`
7. `last_trade_count`
8. `last_trade_side_sign`
9. `last_trade_signed_count`
10. `time_since_last_trade_seconds`
11. `minutes_since_market_open`
12. `trade_count_30s`
13. `contracts_sum_30s`
14. `signed_contracts_sum_30s`
15. `yes_taker_share_30s`
16. `price_return_30s`
17. `price_volatility_30s`
18. `trade_count_120s`
19. `contracts_sum_120s`
20. `signed_contracts_sum_120s`
21. `yes_taker_share_120s`
22. `price_return_120s`
23. `price_volatility_120s`
24. `trade_count_300s`
25. `contracts_sum_300s`
26. `signed_contracts_sum_300s`
27. `yes_taker_share_300s`
28. `price_return_300s`
29. `price_volatility_300s`

### Feature semantics

#### Base probability features

- `z_implied`: Gaussian z-score implied by market probability
- `tau_minutes`: minutes to market close
- `price_momentum`: current probability minus previous trade probability
- `abs_price_momentum`: absolute momentum
- `price_direction`: sign of momentum
- `distance_from_mid`: absolute distance from 0.5

#### Event-level trade features

- `last_trade_count`: contracts traded in the latest trade
- `last_trade_side_sign`: `+1` for yes taker, `-1` for no taker
- `last_trade_signed_count`: signed trade size
- `time_since_last_trade_seconds`: wall-clock seconds since the previous trade
- `minutes_since_market_open`: elapsed time since market open

#### Rolling windows

Windows are tracked over:

- 30 seconds
- 120 seconds
- 300 seconds

For each window, the pipeline computes:

- trade count
- contracts traded
- signed contracts traded
- yes taker share
- price return
- price volatility of per-trade probability changes

### Features intentionally excluded from Phase 1

The collector now tracks these live fields, but they are not used in Phase 1 model training because historical quote-aligned snapshots are not yet available:

- `last_price`
- `yes_bid`
- `yes_ask`
- `volume`
- `open_interest`
- `dollar_volume`
- `dollar_open_interest`
- quote deltas
- spread or mid-based features

## Split Strategy

### Primary split

The default fixed split is:

- train: earliest 65% of KXBTC15M market tickers
- validation: next 15%
- test: newest 20%

The split key is **market close time**, not row order across the whole dataset.

### Walk-forward evaluation

In addition to the primary split, the pipeline builds 6 chronological blocks and evaluates 4 walk-forward folds:

- fold 1: block 1 train, block 2 validation, block 3 test
- fold 2: blocks 1-2 train, block 3 validation, block 4 test
- fold 3: blocks 1-3 train, block 4 validation, block 5 test
- fold 4: blocks 1-4 train, block 5 validation, block 6 test

This is not a substitute for the final test split. It is a robustness check.

## Training Configuration

### LightGBM objective

- objective: `binary`
- evaluation metric: `binary_logloss`
- early stopping: 100 rounds

### Optuna search

The pipeline searches over:

- `n_estimators`
- `learning_rate`
- `num_leaves`
- `max_depth`
- `min_child_samples`
- `subsample`
- `colsample_bytree`
- `reg_lambda`

Default search settings:

- trials: `40`
- sample size: `2_000_000` train rows
- seed: `42`

### Calibration

Platt scaling is fitted **only** on validation predictions.

Artifact:

- `lightgbm/calibration.json`

If the validation split has only one observed class, calibration falls back to identity rather than crashing.

## Trading Policy Tuning

The model does not trade directly. It produces calibrated probabilities, and then the validation set is used to tune a trading rule.

### Policy grid

- `edge_threshold_cents`: `0.5` to `6.0` by `0.5`
- `min_tau_minutes`: `0`, `1`, `2`, `3`
- `max_tau_minutes`: `12`, `14`, `15`
- `price_band`: `10-90`, `20-80`, `30-70`
- `contracts_per_order`: `1`
- `slippage_pct`: `1.0`
- `allow_stacking`: `false`
- `reserve_cash_pct`: `30.0`

### Objective

Primary selection score:

`net_pnl_dollars / max(1.0, max_drawdown_dollars)`

Hard filters:

- net PnL must be positive
- number of trades must be at least `500`

Tie-breakers:

1. higher net PnL
2. lower log-loss

## Artifact Contract

### Run directory layout

Each run writes to:

`artifacts/kalshi/kxbtc15m_lightgbm/<run_name_or_timestamp>/`

Expected contents:

```text
artifacts/kalshi/kxbtc15m_lightgbm/
├── <run_name_or_timestamp>/
│   ├── feature_manifest.json
│   ├── split_manifest.json
│   ├── policy.json
│   ├── policy_search.json
│   ├── summary.json
│   ├── test_metrics.json
│   ├── test_predictions.parquet
│   ├── walk_forward.json
│   ├── datasets/
│   │   └── all/
│   │       └── <ticker>.parquet
│   └── lightgbm/
│       ├── model.txt
│       ├── metrics.json
│       ├── feature_importance.json
│       ├── calibration.json
│       └── optuna_history.json
└── latest/
    ├── feature_manifest.json
    ├── split_manifest.json
    ├── policy.json
    ├── summary.json
    ├── run.json
    └── lightgbm/
        ├── model.txt
        ├── metrics.json
        ├── feature_importance.json
        └── calibration.json
```

### Why `latest/` exists

The live scorer defaults to:

`artifacts/kalshi/kxbtc15m_lightgbm/latest/lightgbm/model.txt`

That means a completed training run can become the live default without hand-copying files.

### Backward compatibility

The scorer still falls back to the legacy model path if `latest/` does not exist.

It also aligns feature rows to the feature list recorded in `metrics.json`, so legacy 6-feature models remain scoreable.

## How to Run the Offline Workflow

### 1. Install dependencies

```powershell
uv sync
```

### 2. Run tests before a major training job

Recommended targeted suite:

```powershell
uv run pytest tests\test_kalshi_offline_training.py tests\test_kalshi_live_collector.py tests\test_kalshi_feature_engine.py tests\test_kalshi_lightgbm_scorer.py -q
```

Optional import/compile smoke check:

```powershell
python -m compileall src\live\kalshi scripts\train_kxbtc15m_lightgbm.py
```

### 3. Run a smoke training job first

Use a named run directory so it is obvious what the experiment was:

```powershell
uv run python scripts\train_kxbtc15m_lightgbm.py --series KXBTC15M --run-name smoke_kxbtc15m
```

If you want to pass explicit paths, use real quoted paths. Do **not** type angle brackets in PowerShell. For this repo, valid local examples are:

```powershell
uv run python scripts\train_kxbtc15m_lightgbm.py --series KXBTC15M --markets-path "output\kalshi_series_backfill\KXBTC15M_markets.parquet" --trades-path "output\kalshi_series_backfill\KXBTC15M_trades.parquet" --run-name smoke_kxbtc15m
```

### 4. Run a full training job

```powershell
uv run python scripts\train_kxbtc15m_lightgbm.py --series KXBTC15M --run-name kxbtc15m_v1
```

Explicit-path alternative:

```powershell
uv run python scripts\train_kxbtc15m_lightgbm.py --series KXBTC15M --markets-path "output\kalshi_series\KXBTC15M_markets.parquet" --trades-path "output\kalshi_series\KXBTC15M_trades.parquet" --run-name kxbtc15m_v1
```

### 5. Inspect outputs

Start with:

- `summary.json`
- `policy.json`
- `test_metrics.json`
- `walk_forward.json`
- `lightgbm/feature_importance.json`
- `lightgbm/optuna_history.json`

### 6. Resume an interrupted evaluation run

If a training run completed model fitting but stopped before writing `policy.json`, `test_metrics.json`, `walk_forward.json`, or `summary.json`, use:

```powershell
uv run python scripts\resume_kxbtc15m_evaluation.py --run-dir "artifacts\kalshi\kxbtc15m_lightgbm\kxbtc15m_v1"
```

Faster variant without walk-forward retraining:

```powershell
uv run python scripts\resume_kxbtc15m_evaluation.py --run-dir "artifacts\kalshi\kxbtc15m_lightgbm\kxbtc15m_v1" --skip-walk-forward
```

By default, the resume script can fall back to the best overall validation policy if no policy meets the original strict requirement of positive validation PnL and at least 500 trades. To preserve the original strict behavior and fail instead of falling back:

```powershell
uv run python scripts\resume_kxbtc15m_evaluation.py --run-dir "artifacts\kalshi\kxbtc15m_lightgbm\kxbtc15m_v1" --strict-policy-selection
```

## How to Run the Live Stack

The live stack should be used in demo or paper style first, not immediate production execution.

### Collector only

```powershell
uv run python scripts\run_kalshi_market_collector.py --environment demo --series KXBTC15M
```

### Collector + feature engine

```powershell
uv run python scripts\run_kalshi_feature_engine.py --environment demo --series KXBTC15M
```

### Collector + feature engine + scorer

```powershell
uv run python scripts\run_kalshi_lightgbm_scorer.py --environment demo --series KXBTC15M
```

### Collector + feature engine + scorer + signal/risk

```powershell
uv run python scripts\run_kalshi_signal_risk_engine.py --environment demo --series KXBTC15M
```

### Full live chain through execution

```powershell
uv run python scripts\run_kalshi_execution_engine.py --environment demo --series KXBTC15M
```

## Signal and Risk Configuration

The live signal engine reads environment-based configuration. The resolved config controls:

- edge threshold
- tau window
- contracts per order
- reserve cash
- slippage
- price band
- stacking
- reservation TTL
- cooldown

Environment variables can be set using either:

- `KALSHI_DEMO_SIGNAL_<NAME>`
- `KALSHI_SIGNAL_<NAME>`

Equivalent production prefixes exist with `PROD`.

Important signal variable suffixes:

- `EDGE_THRESHOLD_CENTS`
- `MIN_TAU_MINUTES`
- `MAX_TAU_MINUTES`
- `ALLOW_STACKING`
- `STARTING_CASH_DOLLARS`
- `CONTRACTS_PER_ORDER`
- `RESERVE_CASH_PCT`
- `SLIPPAGE_PCT`
- `PRICE_BAND_MIN_CENTS`
- `PRICE_BAND_MAX_CENTS`
- `RESERVATION_TTL_SECONDS`
- `TRADE_COOLDOWN_SECONDS`

## How to Read Results

### Metrics that matter first

For this strategy, start with:

1. test-set net PnL after fees and slippage
2. max drawdown
3. trade count
4. walk-forward consistency
5. calibration and log-loss

### Good signs

- positive validation PnL and positive test PnL
- enough trades to avoid a tiny-sample illusion
- reasonable drawdown relative to PnL
- profits spread across multiple walk-forward folds
- calibration that does not collapse at the tails

### Bad signs

- good log-loss but negative trading PnL
- all profits concentrated in one period
- very few qualifying trades
- strong sensitivity to tiny threshold changes
- test performance much worse than validation
- calibration or profitability improving only when test decisions leak back into tuning

## Optimization Strategy

Optimization should happen in this order.

### Stage 1: Verify the data

Before touching hyperparameters:

- confirm market count
- confirm resolved results are present
- confirm enough rows fall into `0 < tau <= 15`
- confirm train, validation, and test all have non-empty feature rows

### Stage 2: Verify feature correctness

Do not optimize a broken feature set.

Check:

- feature parity between offline and live paths
- rolling-window calculations
- no future leakage
- correct handling of `taker_side`
- correct price clipping and tau calculation

### Stage 3: Tune model hyperparameters

Only after stages 1 and 2 are stable.

Primary goal:

- improve probability quality without creating an overly complex model that is fragile over time

### Stage 4: Tune policy parameters

If the model probabilities are reasonable but PnL is weak, the issue may be in policy parameters rather than model quality.

Tune:

- edge threshold
- tau window
- price band
- slippage assumptions

### Stage 5: Evaluate robustness

If a configuration only works in one fold or one short period, do not trust it.

### Stage 6: Add richer features

Only after the trade-derived baseline is stable should Phase 2 quote-derived features be promoted into training.

## What to Do Next Based on Outcomes

### Case A: validation and test are both strong

Recommended next step:

- run the scorer and signal stack in demo mode
- compare live demo behavior to offline expectations
- do not widen scope yet

### Case B: log-loss is decent but PnL is weak

Recommended next step:

- inspect `policy_search.json`
- check whether trades are happening too early, too late, or at poor prices
- tune policy first before inventing more model complexity

### Case C: validation is good but test is weak

Recommended next step:

- assume regime instability or overfitting
- inspect walk-forward results
- simplify features or regularize the model
- avoid live deployment

### Case D: both validation and test are weak

Recommended next step:

- verify data and labels first
- inspect feature generation
- inspect whether the KXBTC15M market family is learnable with Phase 1 features alone
- only then consider Phase 2 quote features

### Case E: profits are highly concentrated in one fold

Recommended next step:

- treat the strategy as unstable
- do not ship live
- analyze that profitable fold to understand if the model is exploiting a temporary regime

## Recommended Go / No-Go Framework

This is not enforced in code, but it is the recommended operational standard.

### Green light for demo or paper

- positive test net PnL after costs
- positive or at least non-catastrophic drawdown-adjusted objective
- minimum trade count comfortably above 500
- walk-forward results not dominated by a single fold
- no obvious calibration collapse

### Yellow light

- some evidence of edge, but unstable fold results
- acceptable log-loss, weak or noisy PnL
- very narrow operating region that may not survive live conditions

Action:

- continue research
- do not move to production trading

### Red light

- negative test PnL after costs
- poor walk-forward behavior
- obvious overfitting or leakage
- feature mismatch between offline and live

Action:

- stop tuning on the same test split
- fix data or methodology before more experiments

## Phase 2 Plan

Phase 2 exists because the collector now subscribes to Kalshi's public `ticker` channel and stores the live fields needed for richer features.

### What Phase 2 should add

Once enough live ticker history has been collected:

- `yes_bid`
- `yes_ask`
- spread
- mid-based features
- quote-to-last trade gaps
- short-horizon volume deltas
- short-horizon open interest deltas

### What must be true before enabling Phase 2 features in training

- quote history is stored for the same periods and markets you want to train on
- quote timestamps can be aligned to the scoring event without future leakage
- offline feature generation is updated first
- live feature generation remains identical in ordering and semantics
- tests are added for parity and leakage

## Common Failure Modes

### 1. Training on a random row split

This creates leakage and optimistic results.

### 2. Tuning on the test set

This invalidates the reported generalization result.

### 3. Changing feature order without retraining

LightGBM models assume column order. Any mismatch can silently produce garbage predictions.

### 4. Mixing old and new artifacts

Always inspect `feature_manifest.json`, `metrics.json`, and `run.json` before assuming which model is live.

### 5. Assuming quote features exist historically

The collector has Phase 2 groundwork, but Phase 1 training is still trade-derived.

### 6. Over-optimizing on one profitable period

A narrow edge that only appears in one regime is not deployment-ready.

## Guidance for Future Coding Agents

If you are an agent modifying this pipeline, follow these rules.

### Non-negotiable rules

- Do not introduce random row splitting for model selection.
- Do not use test data for tuning.
- Do not add offline-only features that cannot be reproduced live.
- Do not change `FEATURE_ORDER` without updating manifests, retraining, and testing compatibility.
- Do not delete prior run artifacts automatically.

### Before changing features

1. Read `src/live/kalshi/features.py`
2. Read `src/live/kalshi/feature_engine.py`
3. Read `src/live/kalshi/scorer.py`
4. Confirm the live path can compute the same fields in the same order
5. Add or update tests for offline/live parity

### Before changing model selection logic

1. Read `src/live/kalshi/offline_training.py`
2. Confirm validation and test boundaries remain clean
3. Keep profitability tuning on validation only
4. Preserve a report that can be compared across runs

### Before changing live deployment behavior

1. Read `src/live/kalshi/signal_risk.py`
2. Understand whether the change affects policy semantics or just execution plumbing
3. Keep demo-first deployment discipline

## Recommended Working Sequence for This Repo

If starting a fresh research cycle, follow this order:

1. Validate raw KXBTC15M markets and trades
2. Run the targeted test suite
3. Run a smoke training job
4. Inspect summary and artifacts
5. Run the full training job
6. Review test metrics and walk-forward results
7. If acceptable, run scorer and signal engines in demo mode
8. Compare live demo behavior against offline assumptions
9. Only then consider production execution or Phase 2 feature expansion

## Summary

This repo now has a KXBTC15M-centered LightGBM workflow that is:

- chronological
- market-grouped
- trade-cost aware
- aligned with the live scorer
- non-destructive in artifact handling
- test-covered at the critical integration points

The current best strategy is not to chase more complexity immediately. The correct next move is to use this pipeline to produce a stable Phase 1 trade-derived baseline, evaluate it honestly, and only then decide whether the next bottleneck is:

- more data quality work
- better policy tuning
- better regularization
- or richer quote-derived Phase 2 features
