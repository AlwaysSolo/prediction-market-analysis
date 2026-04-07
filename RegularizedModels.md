# Regularized Models Comparison

## Scope

This document compares the current offline `KXBTC15M` model families on the same historical split:

- Train rows: `2,952,517`
- Validation rows: `3,086,564`
- Test rows: `5,892,268`
- Feature count: `29`
- Trading assumptions:
  - `1` contract per trade
  - `1%` slippage
  - `30%` reserve cash
  - stacking `off`

## Model Comparison

| Model | Model type | Best params | Validation log-loss | Test log-loss | Validation PnL $ | Test PnL $ | Validation DD $ | Test DD $ | Validation objective | Test objective | Validation trades | Test trades | Selected policy | Walk-forward complete | Current evaluation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `LASSO` | `l1_logistic_regression` | `C=0.0003`, `l1_ratio=1.0` | `0.511984` | `0.512838` | `19.4932` | `23.5621` | `8.0670` | `8.7140` | `2.4164` | `2.7039` | `608` | `627` | edge `4.5c`, tau `0-12`, price `10-90` | `False` | Best current live-paper candidate |
| `Elastic Net` | `elastic_net_logistic_regression` | `C=0.0003`, `l1_ratio=0.5` | `0.512352` | `0.513150` | `11.6660` | `20.2705` | `8.4900` | `7.8156` | `1.3741` | `2.5936` | `530` | `547` | edge `5.0c`, tau `1-12`, price `10-90` | `False` | Good backup candidate, slightly lower edge |
| `Linear SVM` | `l1_linear_svm` | `C=0.0003`, `penalty=l1`, `loss=squared_hinge` | `0.512450` | `0.513182` | `29.6132` | `19.6918` | `7.4059` | `10.2751` | `3.9986` | `1.9165` | `524` | `571` | edge `5.5c`, tau `1-12`, price `10-90` | `False` | Positive but behind LASSO and Elastic Net on test |
| `LightGBM` | `gradient_boosted_trees` | `best_iteration=218`, `num_leaves=15`, `learning_rate=0.0151` | `0.510584` | `0.513725` | `39.2748` | `-33.4746` | `9.4532` | `42.0699` | `4.1547` | `-0.7957` | `991` | `1536` | edge `5.0c`, tau `2-12`, price `20-80` | `False` | Not suitable for live test in current form |

## Interpretation

| Rank | Model | Why |
| --- | --- | --- |
| `1` | `LASSO` | Best test PnL, best test objective, and slightly better test log-loss than Elastic Net while staying positive on validation and test |
| `2` | `Elastic Net` | Also positive on validation and test, with slightly lower test drawdown than LASSO, but weaker PnL and weaker objective |
| `3` | `Linear SVM` | Strong validation result, but weaker generalization than LASSO and Elastic Net on the untouched test split |
| `4` | `LightGBM` | Better validation fit, but failed the untouched test split and is currently not deployable |

## Current Decision

| Question | Answer |
| --- | --- |
| Best current model for live paper test | `LASSO` |
| Best backup model | `Elastic Net (l1_ratio=0.5)` |
| Where does `Linear SVM` rank? | `Third`, behind LASSO and Elastic Net |
| Should `LightGBM` be used for live paper test now? | `No` |
| Is any model ready for real-money deployment? | `No`, because walk-forward is still incomplete for all four |

## Source Runs

| Model | Summary path |
| --- | --- |
| `LASSO` | `artifacts/kalshi/kxbtc15m_lasso/kxbtc15m_l1_compare_v1_lasso_l1r100/summary.json` |
| `Elastic Net` | `artifacts/kalshi/kxbtc15m_elastic_net/kxbtc15m_elastic_net_l1r050_v1/summary.json` |
| `Linear SVM` | `artifacts/kalshi/kxbtc15m_linear_svm/kxbtc15m_linear_svm_v1/summary.json` |
| `LightGBM` | `artifacts/kalshi/kxbtc15m_lightgbm/kxbtc15m_v1/summary.json` |
