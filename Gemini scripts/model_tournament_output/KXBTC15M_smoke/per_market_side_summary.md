# KXBTC15M Tournament By Market Outcome Side

- Metric: `log_loss` on the held-out test split
- Market side buckets: `NO-leaning` for implied probability `< 50c`, `YES-leaning` for `>= 50c`
- Models: same tournament lineup as `smarterpred.py`

## Winners By Side

| Market Side | Test Rows | Best Model | Best Log-Loss |
| --- | ---: | --- | ---: |
| NO-leaning | 1,970 | Student-T df=5 | 0.51965 |
| YES-leaning | 2,030 | XGBoost | 0.51101 |

## Full Score Matrix

| Market Side | Test Rows | Standard Gaussian | Student-T df=5 | Student-T df=3 | Momentum-Adjusted Gaussian | Machine Learning (Logistic Regression) | XGBoost | LightGBM | CatBoost | PySR (Symbolic Regression) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NO-leaning | 1,970 | 0.51992 | 0.51965 | 0.52147 | 0.52024 | 0.52091 | 0.52425 | 0.52546 | 0.52236 | 0.52573 |
| YES-leaning | 2,030 | 0.51281 | 0.51434 | 0.51680 | 0.51559 | 0.51341 | 0.51101 | 0.51330 | 0.51256 | 0.51452 |
