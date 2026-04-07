# Kalshi Research Archive Stack

This repo now supports an archive-enabled version of the research runner that records enough detail for:

- stop-loss and take-profit replay
- future strategy backtests on the same live market path
- model retraining from archived feature rows
- analysis of model outputs and research decisions without changing the normal research dashboard logs

The archive is **research-runner only** in v1. It is additive and does **not** replace:

- `output/live_research/<run_name>/research/.../events.jsonl`

Those research logs still drive the current dashboard and recovery flow.

## Archive Root

By default, archive mode writes under:

- `output/live_research/<run_name>/archive/`

You can override that with `--archive-root`.

## The 4 Archive Layers

### 1. `raw_ws/`

Purpose:
- immutable source-of-truth capture of incoming websocket envelopes

What it stores:
- `ticker`
- `trade`
- `market_lifecycle_v2`

Format:
- append-only `JSONL.zst`
- partitioned by `environment / date / hour / channel`

Why it exists:
- replay exact market paths later
- derive new features in the future
- evaluate intratrade logic like stop loss from raw quote movement

### 2. `market_events_staging/` -> `market_events/`

Purpose:
- canonical normalized market-state rows after collector updates

What it stores:
- market and timing metadata
- last trade price
- full quote book when available
- derived quote mid / spread / buy-YES / buy-NO / quote age
- trade count, side, volume, open interest, close/open times

Live format:
- hourly append-only `JSONL.zst` staging shards

Compacted format:
- Parquet under:
  - `archive/market_events/environment=<env>/date=<YYYY-MM-DD>/hour=<HH>/`

### 3. `feature_rows_staging/` -> `feature_rows/`

Purpose:
- one canonical feature snapshot per scoreable changed feature state

What it stores:
- feature metadata and lineage IDs
- quote metadata
- current market probability basis
- all canonical features
- hourly-context features
- derived linear features

Live format:
- hourly append-only `JSONL.zst` staging shards

Compacted format:
- Parquet under:
  - `archive/feature_rows/environment=<env>/date=<YYYY-MM-DD>/hour=<HH>/`

### 4. `model_outputs_staging/` -> `model_outputs/`

Purpose:
- one per-model scorer row for each archived feature row

What it stores:
- `model_label`
- `model_family`
- `model_file`
- predicted YES / NO probabilities
- model edge
- quote-facing fields used during scoring

Live format:
- hourly append-only `JSONL.zst` staging shards

Compacted format:
- Parquet under:
  - `archive/model_outputs/environment=<env>/date=<YYYY-MM-DD>/hour=<HH>/`

### 5. `strategy_events_staging/` -> `strategy_events/`

Purpose:
- normalized research events for replay and strategy analysis

What it stores:
- `recorded`
- `skipped`
- `settled`
- `summary_snapshot`

It links strategy rows back to:
- `feature_row_id`
- `model_output_id`
- `sample_id` where applicable

Live format:
- hourly append-only `JSONL.zst` staging shards

Compacted format:
- Parquet under:
  - `archive/strategy_events/environment=<env>/date=<YYYY-MM-DD>/hour=<HH>/`

## IDs And Lineage

The archive uses explicit join keys across layers:

- `raw_event_id`
- `market_event_id`
- `feature_row_id`
- `model_output_id`
- `strategy_event_id`
- `sample_id`

Lineage flow:

- collector raw websocket event -> `raw_event_id`
- collector normalized update -> `market_event_id`
- feature row -> `feature_row_id` linked to `market_event_id`
- model output -> `model_output_id` linked to `feature_row_id`
- strategy event -> linked to `feature_row_id` and `model_output_id`

## Write And Compaction Model

Live write path:
- raw websocket capture stays as raw `JSONL.zst`
- all other archive layers write hourly `JSONL.zst` staging shards

Compaction behavior:
- compact on hour rollover
- compact on graceful shutdown
- compact later with the repair command after a crash/restart

The raw websocket layer is never compacted away.

## Disk And Performance Notes

Archive mode is heavier than the standard research runner because it records:

- raw websocket traffic
- normalized market events
- full feature rows
- per-model scorer outputs
- normalized strategy events

Expect more disk usage and more files when `--archive full` is enabled.

## Recommended Usage

Use archive mode when you care about:

- future stop-loss / take-profit replay
- retraining with archived feature rows
- path-dependent strategy analysis
- detailed model-behavior studies across market regimes

Use the normal research run without archive mode when you only need:

- research dashboard monitoring
- current bucket sampling logs
- lighter storage usage

## Repair Command

If the machine restarts or the run is interrupted, compact any leftover staging shards with:

```powershell
uv run python scripts\repair_kalshi_research_archive.py `
  --archive-root output\live_research\kalshi_bucket_discovery_archive\archive `
  --environment demo
```

If you used the default archive root, the path is usually:

```text
output/live_research/<run_name>/archive
```
