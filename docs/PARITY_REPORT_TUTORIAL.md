# Parity Report Tutorial

## What This Feature Does

The parity report answers a simple but important question:

When the live system placed a trade, how did that outcome compare to:

1. what the research sampler would have done,
2. what a deterministic shadow execution would have achieved from the same decision snapshot,
3. what actually happened live.

The script produces three PnL layers per settled live trade:

- `research_theoretical_pnl_dollars`
- `shadow_simulated_pnl_dollars`
- `live_realized_pnl_dollars`

It then computes the two main gap series:

- `shadow_minus_research_pnl_dollars`
- `live_minus_shadow_pnl_dollars`

These are the two numbers you care about most:

- If `shadow - research` is drifting, the problem is usually in the research-to-execution simulation layer.
- If `live - shadow` is drifting, the problem is usually in real execution, fees, fills, or operational behavior.

## Where It Lives

- Script: `scripts/parity_report.py`
- Core logic: `src/live/kalshi/parity_analysis.py`
- Outputs: `artifacts/kalshi/parity_reports/YYYY-MM-DD/`

## PowerShell vs CMD

If you are using PowerShell, do not use `^` for line continuation.

- `^` is for `cmd.exe`
- PowerShell uses the backtick: `` ` ``

This works in PowerShell:

```powershell
python scripts/parity_report.py `
  --as-of-date 2026-04-18 `
  --lookback-days 14 `
  --rolling-window-days 14 `
  --models bagged_lasso `
  --live-run output/live/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438 `
  --research-run output/live_research/kalshi_bucket_discovery_regime_prod
```

This also works as a one-liner:

```powershell
python scripts/parity_report.py --as-of-date 2026-04-18 --lookback-days 14 --rolling-window-days 14 --models bagged_lasso --live-run output/live/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438 --research-run output/live_research/kalshi_bucket_discovery_regime_prod
```

Important: the backtick must be the last character on the line. No trailing spaces after it.

## Quick Start

### Recommended first run

Start with an explicit live run and an explicit research run.

```powershell
python scripts/parity_report.py `
  --as-of-date 2026-04-18 `
  --lookback-days 14 `
  --rolling-window-days 14 `
  --models bagged_lasso `
  --live-run output/live/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438 `
  --research-run output/live_research/kalshi_bucket_discovery_regime_prod
```

Why this is the best starting point:

- It is faster than scanning the entire repo.
- It makes the result easier to interpret.
- It avoids mixing unrelated runs while you are learning the workflow.

### What gets written

The script writes a dated report directory and refreshes `latest/`.

Example:

- `artifacts/kalshi/parity_reports/2026-04-18/`
- `artifacts/kalshi/parity_reports/latest/`

## Main Output Files

Each dated report directory contains:

- `paired_trades.parquet`
- `paired_trades.csv`
- `daily_rollup.csv`
- `daily_segment_rollup.csv`
- `rolling_rollup.csv`
- `summary.json`
- `manifest.json`
- `parity_report.md`
- `parity_gaps.png`

### Start here

If you only want the high-level result, open these first:

- `parity_report.md`
- `summary.json`
- `parity_gaps.png`

### Use these when debugging

- `paired_trades.parquet`: one row per live trade
- `daily_rollup.csv`: daily totals and nominal 95% intervals
- `daily_segment_rollup.csv`: rows broken out by drift segment
- `rolling_rollup.csv`: rolling-window verdict inputs and breach labels

## Reading The Results

## 1. `summary.json`

This is the best machine-readable snapshot.

Important keys:

- `job_status`
- `snapshot_match_coverage`
- `primary_cohort_coverage`
- `replay_share`
- `parity_breach`
- `low_coverage_warning`
- `latest_14d`
- `latest_30d`
- `latest_rolling`

What they mean:

- `snapshot_match_coverage`: share of settled live trades that matched a usable decision snapshot
- `primary_cohort_coverage`: share of settled live trades that were comparable enough to count toward the main verdict
- `replay_share`: share of rows that used replayed research logic instead of recorded research rows
- `parity_breach`: whether the latest rolling window breached parity rules

## 2. `parity_report.md`

This is the human summary.

It leads with:

- 14-day research/shadow/live PnL
- 14-day `shadow - research`
- 14-day `live - shadow`
- latest leak label
- coverage stats
- drift-segment counts

## 3. `parity_gaps.png`

This chart has two panels:

- Rolling cumulative gap dollars
- Rolling mean gap per trade

Use it to answer:

- Is the gap consistently drifting one way?
- Is the drift getting worse?
- Is the gap economically meaningful or just noise?

## 4. `paired_trades.parquet`

This is the main debugging table.

If a report looks wrong, this is where you go.

Important columns:

- `primary_cohort`
- `research_source`
- `research_action`
- `match_quality`
- `live_realized_pnl_dollars`
- `shadow_simulated_pnl_dollars`
- `research_theoretical_pnl_dollars`
- `shadow_minus_research_pnl_dollars`
- `live_minus_shadow_pnl_dollars`
- `live_minus_research_pnl_dollars`

## How To Interpret Key Columns

### `primary_cohort`

`true` means the row is included in the main parity verdict.

For a row to be in the primary cohort, it generally needs:

- a usable live snapshot,
- a usable shadow counterfactual,
- a research decision,
- `research_action == entered_same`,
- a valid match quality,
- recorded research by default.

If `primary_cohort` is `false`, the row is still useful, but it is no longer apples-to-apples enough for the main breach test.

### `research_source`

Expected values:

- `recorded`
- `replayed`
- effectively missing/unusable through exclusion

Interpretation:

- `recorded`: preferred
- `replayed`: fallback generated from archived model outputs and deterministic research logic

By default, replayed rows are excluded from the main verdict unless you pass `--allow-replayed-in-verdict`.

### `research_action`

Expected values:

- `entered_same`
- `declined`
- `entered_different_side`
- `entered_different_size`

Interpretation:

- `entered_same`: best comparison, counts toward primary verdict
- `declined`: research would not have traded; useful for opportunity-cost analysis, not leak attribution
- `entered_different_side`: research and live disagreed on direction
- `entered_different_size`: research and live disagreed on size

### `match_quality`

Expected values:

- `exact_id`
- `exact_feature_row`
- `exact_market_event`
- `timestamp_fallback`
- `model_version_mismatch`
- `unmatched`

Interpretation:

- `exact_*`: strong match
- `timestamp_fallback`: acceptable fallback, but weaker than exact matching
- `model_version_mismatch`: matched something structurally similar, but model file or feature schema did not agree, so the row is excluded from parity math
- `unmatched`: no good snapshot match was found

## Typical Workflow

## Workflow 1: Quick health check

Run:

```powershell
python scripts/parity_report.py `
  --as-of-date 2026-04-18 `
  --lookback-days 14 `
  --rolling-window-days 14 `
  --models bagged_lasso `
  --live-run output/live/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438 `
  --research-run output/live_research/kalshi_bucket_discovery_regime_prod
```

Then inspect:

1. `summary.json`
2. `parity_report.md`
3. `parity_gaps.png`

Questions to ask:

- Is `parity_breach` true?
- What is `latest_rolling.leak_label`?
- Is coverage good enough to trust the verdict?

## Workflow 2: Investigate a live execution leak

If `leak_label` is one of:

- `live_execution_adverse`
- `live_execution_favorable`

then focus on `paired_trades`.

Filter to:

- `primary_cohort == true`

Then compare:

- `live_fill_price_cents`
- `shadow_entry_price_cents`
- `live_fees_dollars`
- `shadow_fees_dollars`
- `live_realized_pnl_dollars`
- `shadow_simulated_pnl_dollars`

This helps answer:

- Did live fills come in worse than expected?
- Were fees materially different?
- Did partial fills distort outcomes?

## Workflow 3: Investigate a research/shadow leak

If `leak_label` is one of:

- `shadow_vs_research_adverse`
- `shadow_vs_research_favorable`

then the problem is upstream of live execution.

Check:

- `research_source`
- `research_action`
- `research_theoretical_pnl_dollars`
- `shadow_simulated_pnl_dollars`

Then inspect `daily_segment_rollup.csv`.

Questions to ask:

- Are many rows `declined`?
- Are many rows `entered_different_side`?
- Are replayed rows dominating?
- Is the shadow counterfactual using a different effective entry than research assumed?

## Workflow 4: Coverage problem

If `low_coverage_warning` is `true`, do not over-trust the breach label yet.

Check:

- `snapshot_match_coverage`
- `primary_cohort_coverage`
- `match_quality`
- `research_source`

Common causes:

- no exact feature-row match
- model version mismatch
- schema version mismatch
- insufficient recorded research coverage

## Useful Command Variants

### Include replayed rows in the verdict

```powershell
python scripts/parity_report.py `
  --as-of-date 2026-04-18 `
  --lookback-days 14 `
  --rolling-window-days 14 `
  --models bagged_lasso `
  --live-run output/live/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438 `
  --research-run output/live_research/kalshi_bucket_discovery_regime_prod `
  --allow-replayed-in-verdict
```

Use this only after you understand how much of the report is replay-driven.

### Fail the command if coverage is bad

```powershell
python scripts/parity_report.py `
  --as-of-date 2026-04-18 `
  --lookback-days 14 `
  --rolling-window-days 14 `
  --models bagged_lasso `
  --live-run output/live/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438 `
  --research-run output/live_research/kalshi_bucket_discovery_regime_prod `
  --fail-on-low-coverage
```

Exit code behavior:

- `0`: success
- `2`: low coverage and `--fail-on-low-coverage` was set
- `3`: parity breach and `--fail-on-parity-breach` was set
- `4`: output directory was not produced

### Fail the command if parity is breached

```powershell
python scripts/parity_report.py `
  --as-of-date 2026-04-18 `
  --lookback-days 14 `
  --rolling-window-days 14 `
  --models bagged_lasso `
  --live-run output/live/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438 `
  --research-run output/live_research/kalshi_bucket_discovery_regime_prod `
  --fail-on-parity-breach
```

### Use a custom output directory

```powershell
python scripts/parity_report.py `
  --as-of-date 2026-04-18 `
  --lookback-days 14 `
  --rolling-window-days 14 `
  --models bagged_lasso `
  --live-run output/live/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438 `
  --research-run output/live_research/kalshi_bucket_discovery_regime_prod `
  --output-root artifacts/kalshi/custom_parity_reports
```

### Tighten snapshot lag

```powershell
python scripts/parity_report.py `
  --as-of-date 2026-04-18 `
  --lookback-days 14 `
  --rolling-window-days 14 `
  --models bagged_lasso `
  --live-run output/live/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438 `
  --research-run output/live_research/kalshi_bucket_discovery_regime_prod `
  --max-snapshot-lag-ms 250
```

Use this when you want stricter snapshot matching in fast markets.

## Recommended Defaults

For day-to-day usage:

- keep `--allow-replayed-in-verdict` off
- specify `--models`
- specify `--live-run` and `--research-run`
- use `14` days for both lookback and rolling window first

Suggested operator command:

```powershell
python scripts/parity_report.py `
  --as-of-date 2026-04-18 `
  --lookback-days 14 `
  --rolling-window-days 14 `
  --models bagged_lasso `
  --live-run output/live/kalshi_bagged_lasso_live_structural_regime_v2_20260414_213438 `
  --research-run output/live_research/kalshi_bucket_discovery_regime_prod `
  --fail-on-low-coverage `
  --fail-on-parity-breach
```

## Troubleshooting

### Error: `Missing expression after unary operator '--'`

You used PowerShell with `^`.

Fix:

- use a one-line command, or
- replace `^` with PowerShell backticks `` ` ``

### The script is slow

Full auto-discovery across all historical runs can be slow.

Best fix:

- pass explicit `--live-run`
- pass explicit `--research-run`
- pass `--models`

### `shadow - research` and `live - shadow` totals are `None`

This usually means there were not enough comparable primary-cohort rows in the rolling window.

Check:

- `primary_cohort`
- `research_action`
- `match_quality`
- `research_source`

### Too many `model_version_mismatch` rows

This means the live snapshot and research row disagree on:

- `model_file`, or
- feature schema version

That is a good safety feature. It prevents silently comparing incompatible decisions.

### Too many `replayed` rows

That means recorded research coverage is thin for the chosen run pairing.

Options:

- choose a better paired research run
- accept replay for exploration only
- use `--allow-replayed-in-verdict` only when you intentionally want that behavior

### Low coverage warning

Default warnings trigger when:

- `snapshot_match_coverage < 0.80`
- `primary_cohort_coverage < 0.60`

If this happens, inspect `paired_trades` before trusting the breach label.

## Example Interpretation

Suppose the report says:

- `shadow - research` is near zero
- `live - shadow` is materially negative
- `leak_label = live_execution_adverse`

Interpretation:

- research logic and shadow counterfactual agree reasonably well
- the problem is not primarily in the research sampler
- the loss is likely happening in real execution, fees, fills, timing, or operations

Suppose instead:

- `shadow - research` is materially negative
- `live - shadow` is near zero
- `leak_label = shadow_vs_research_adverse`

Interpretation:

- live is behaving roughly like the shadow counterfactual
- the leak is between research logic and executable-price simulation
- the next thing to inspect is trade eligibility, prices, or deterministic replay assumptions

## Best Practices

- Always start with explicit live and research runs.
- Use `parity_report.md` for the summary and `paired_trades.parquet` for debugging.
- Treat low coverage as a warning, not a footnote.
- Do not mix `declined` rows into your leak interpretation.
- Keep replayed rows out of the main verdict until you are confident the replay path is representative enough for your use case.

## File Reference

Main files for this feature:

- `scripts/parity_report.py`
- `src/live/kalshi/parity_analysis.py`
- `artifacts/kalshi/parity_reports/latest/`

