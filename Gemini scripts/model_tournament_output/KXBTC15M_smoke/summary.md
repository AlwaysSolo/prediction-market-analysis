# Model Tournament Results

- Series: `KXBTC15M`
- Rows used: `20,000`
- Test rows: `4,000`

| Model | Log-Loss |
| --- | ---: |
| CatBoost | 0.51084 |
| Standard Gaussian | 0.51106 |
| Machine Learning (Logistic Regression) | 0.51170 |
| XGBoost | 0.51210 |
| Momentum-Adjusted Gaussian | 0.51255 |
| Student-T df=5 | 0.51295 |
| LightGBM | 0.51439 |
| Student-T df=3 | 0.51557 |
| PySR (Symbolic Regression) | 0.51586 |

Best predictor: **CatBoost (0.51084)**
