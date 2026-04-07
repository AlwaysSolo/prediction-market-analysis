# KXBTC15M Tournament By Market Outcome Side

- Metric: `log_loss` on the held-out test split
- Market side buckets: `NO-leaning` for implied probability `< 50c`, `YES-leaning` for `>= 50c`
- Models: same tournament lineup as `smarterpred.py`

## Winners By Side

| Market Side | Test Rows | Best Model | Best Log-Loss |
| --- | ---: | --- | ---: |
| NO-leaning | 1,177,664 | LightGBM | 0.49289 |
| YES-leaning | 1,209,319 | LightGBM | 0.50399 |

## Full Score Matrix

| Market Side | Test Rows | Standard Gaussian | Student-T df=5 | Student-T df=3 | Momentum-Adjusted Gaussian | Machine Learning (Logistic Regression) | XGBoost | LightGBM | CatBoost | PySR (Symbolic Regression) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NO-leaning | 1,177,664 | 0.49769 | 0.50125 | 0.50470 | 0.49876 | 0.49762 | 0.49490 | 0.49289 | 0.49473 | 0.50345 |
| YES-leaning | 1,209,319 | 0.50803 | 0.51067 | 0.51363 | 0.50917 | 0.50827 | 0.50596 | 0.50399 | 0.50576 | 0.51247 |
