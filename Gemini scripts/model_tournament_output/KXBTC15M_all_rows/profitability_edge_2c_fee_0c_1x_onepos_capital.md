# KXBTC15M Tournament Profitability Evaluation

- Dataset: held-out tournament test split only
- Evaluation sample rows: `11,934,913`
- Test rows: `2,386,983`

## Default Assumptions

- Entry rule: buy `YES` when `model_prob - market_prob >= 2.00c`; buy `NO` when the reverse is true
- Exit rule: hold every filled trade to expiry
- Entry price: displayed market probability for `YES`, complementary price for `NO`
- Order size: `1` contract(s) per trade
- Fees: `0.00c` flat per contract
- Position rule: `one open position per ticker at a time`
- Capital rule: `enforce available cash before entry`
- Drawdown basis: equity = free cash + open-position cost basis, so only fees and settled PnL move equity

## Results

| Model | Log-Loss | Trades | Net PnL ($) | Return % | Max DD ($) | Max DD % | Win Rate | Skipped Capital |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.49784 | 8,490 | 275.77 | 2.76 | 20.97 | 0.21 | 50.02% | 0 |
| CatBoost | 0.49969 | 8,439 | 262.79 | 2.63 | 14.17 | 0.14 | 48.42% | 0 |
| XGBoost | 0.49987 | 8,441 | 236.14 | 2.36 | 17.86 | 0.17 | 47.78% | 0 |
| PySR (Symbolic Regression) | 0.50734 | 8,533 | 211.68 | 2.12 | 25.05 | 0.25 | 40.14% | 0 |
| Student-T df=3 | 0.50866 | 8,455 | 185.61 | 1.86 | 15.51 | 0.15 | 27.03% | 0 |
| Machine Learning (Logistic Regression) | 0.50232 | 8,340 | 145.21 | 1.45 | 22.70 | 0.22 | 54.29% | 0 |
| Student-T df=5 | 0.50541 | 8,321 | 107.23 | 1.07 | 15.87 | 0.16 | 17.73% | 0 |
| Standard Gaussian | 0.50223 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00% | 0 |
| Momentum-Adjusted Gaussian | 0.50331 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00% | 0 |
