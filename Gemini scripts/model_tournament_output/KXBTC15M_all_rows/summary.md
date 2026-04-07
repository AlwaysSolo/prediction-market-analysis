# Model Tournament Results

- Series: `KXBTC15M`
- Rows used: `11,934,913`
- Test rows: `2,386,983`

| Model | Log-Loss |
| --- | ---: |
| LightGBM | 0.49902 |
| CatBoost | 0.50029 |
| XGBoost | 0.50045 |
| Standard Gaussian | 0.50269 |
| Machine Learning (Logistic Regression) | 0.50284 |
| Momentum-Adjusted Gaussian | 0.50375 |
| Student-T df=5 | 0.50583 |
| PySR (Symbolic Regression) | 0.50770 |
| Student-T df=3 | 0.50906 |

Best predictor: **LightGBM (0.49902)**
