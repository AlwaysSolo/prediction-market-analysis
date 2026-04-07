# Kelly LASSO Evaluation

## Scope

This file compares LASSO sizing variants on the same saved `KXBTC15M` model run:

- Model run: `artifacts/kalshi/kxbtc15m_lasso/kxbtc15m_lasso_v1`
- Signal policy kept fixed:
  - edge threshold: `4.5c`
  - tau window: `0` to `12` minutes
  - price band: `10` to `90c`
  - reserve cash: `30%`
  - slippage: `1%`
  - stacking: `off`
- Only the position sizing rule changes

## Sizing Rules Compared

| Strategy | Sizing rule |
| --- | --- |
| `baseline_1_contract` | Fixed `1` contract per trade |
| `capital_2pct` | `2%` of current equity per trade |
| `kelly_full` | Full Kelly fraction of current equity, no hard cap |
| `kelly_010x_cap2` | `0.10x` Kelly, capped at `2%` of equity per trade |
| `kelly_025x_cap2` | `0.25x` Kelly, capped at `2%` of equity per trade |
| `kelly_050x_cap2` | `0.50x` Kelly, capped at `2%` of equity per trade |

## Validation And Test Metrics

| Strategy | Validation trades | Validation PnL $ | Validation max DD $ | Validation objective | Test trades | Test PnL $ | Test max DD $ | Test objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_1_contract` | 608 | 19.49 | 8.07 | 2.4164 | 627 | 23.56 | 8.71 | 2.7039 |
| `capital_2pct` | 695 | 9,154.20 | 8,635.27 | 1.0601 | 698 | 47,806.26 | 51,654.50 | 0.9255 |
| `kelly_full` | 695 | 94,986.37 | 1,185,924.09 | 0.0801 | 698 | 45,686.55 | 2,997,485.59 | 0.0152 |
| `kelly_010x_cap2` | 694 | 8,669.43 | 3,787.42 | 2.2890 | 698 | 13,506.90 | 6,834.30 | 1.9763 |
| `kelly_025x_cap2` | 695 | 9,533.76 | 6,683.44 | 1.4265 | 698 | 46,757.36 | 35,173.82 | 1.3293 |
| `kelly_050x_cap2` | 695 | 9,154.20 | 8,635.27 | 1.0601 | 698 | 47,806.26 | 51,654.50 | 0.9255 |

## Test Bucket Health Snapshot

| Strategy | Test YES PnL $ | Test NO PnL $ | Best tau bucket | Worst tau bucket | Best price bucket | Worst price bucket | Worst time block |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `baseline_1_contract` | 15.46 | 8.10 | `10-12` (+15.43) | `0-2` (-1.52) | `10-20` (+8.68) | `70-80` (-4.46) | `block_10` (-6.27) |
| `capital_2pct` | 37,659.85 | 10,146.41 | `10-12` (+34,527.20) | `2-4` (-17,125.30) | `10-20` (+27,866.43) | `40-50` (-8,751.95) | `block_10` (-25,458.43) |
| `kelly_full` | 804,908.08 | -759,221.53 | `10-12` (+1,769,159.93) | `2-4` (-2,635,502.88) | `10-20` (+921,305.36) | `80-90` (-1,177,434.05) | `block_09` (-598,277.59) |
| `kelly_010x_cap2` | 6,214.77 | 7,292.13 | `10-12` (+5,804.20) | `2-4` (-748.98) | `10-20` (+7,745.84) | `40-50` (-2,094.90) | `block_10` (-3,470.84) |
| `kelly_025x_cap2` | 30,755.73 | 16,001.64 | `10-12` (+28,267.97) | `2-4` (-9,694.96) | `10-20` (+26,351.25) | `40-50` (-8,067.74) | `block_10` (-19,525.69) |
| `kelly_050x_cap2` | 37,659.85 | 10,146.41 | `10-12` (+34,527.20) | `2-4` (-17,125.30) | `10-20` (+27,866.43) | `40-50` (-8,751.95) | `block_10` (-25,458.43) |

## Readout

| Observation | Result |
| --- | --- |
| Best total test PnL | `capital_2pct` / `kelly_050x_cap2` at `+$47,806.26` |
| Best test objective | `baseline_1_contract` at `2.7039` |
| Best Kelly-based test objective | `kelly_010x_cap2` at `1.9763` |
| Worst sizing rule | `kelly_full` due to catastrophic drawdown despite positive PnL |
| Side balance that looks healthiest | `kelly_010x_cap2`, because both YES and NO stayed positive and closer together |
| Most important cap effect | `kelly_050x_cap2` effectively collapses into the same result as `capital_2pct` under this policy and data |

## Practical Conclusion

| Choice | Evaluation |
| --- | --- |
| `baseline_1_contract` | Best pure risk-adjusted result, but much lower raw dollar PnL |
| `capital_2pct` | Strong raw PnL, but drawdown is much larger and the objective drops materially |
| `kelly_full` | Not usable |
| `kelly_010x_cap2` | Best Kelly candidate so far if the goal is to keep Kelly logic without letting sizing run away |
| `kelly_025x_cap2` | Viable, but clearly riskier than `0.10x` |
| `kelly_050x_cap2` | Too close to plain `2%` capital sizing to add much value |

## Source Files

| Artifact | Path |
| --- | --- |
| Baseline summary | `artifacts/kalshi/kxbtc15m_lasso/kxbtc15m_lasso_v1/summary.json` |
| Baseline test diagnostics | `artifacts/kalshi/kxbtc15m_lasso/kxbtc15m_lasso_v1/test_diagnostics.json` |
| 2% capital summary | `artifacts/kalshi/kxbtc15m_lasso/kxbtc15m_lasso_v1/custom_evals/capital_2pct_no_stacking_summary.json` |
| Full Kelly summary | `artifacts/kalshi/kxbtc15m_lasso/kxbtc15m_lasso_v1/custom_evals/kelly_full_no_stacking_summary.json` |
| `0.10x` Kelly summary | `artifacts/kalshi/kxbtc15m_lasso/kxbtc15m_lasso_v1/custom_evals/kelly_010x_cap2_no_stacking_summary.json` |
| `0.25x` Kelly summary | `artifacts/kalshi/kxbtc15m_lasso/kxbtc15m_lasso_v1/custom_evals/kelly_025x_cap2_no_stacking_summary.json` |
| `0.50x` Kelly summary | `artifacts/kalshi/kxbtc15m_lasso/kxbtc15m_lasso_v1/custom_evals/kelly_050x_cap2_no_stacking_summary.json` |
| Consolidated fractional Kelly comparison | `artifacts/kalshi/kxbtc15m_lasso/kxbtc15m_lasso_v1/custom_evals/kelly_fractional_cap2_comparison.json` |
