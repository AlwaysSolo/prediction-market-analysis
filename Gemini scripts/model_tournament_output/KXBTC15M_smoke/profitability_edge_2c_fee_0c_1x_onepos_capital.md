# KXBTC15M Tournament Profitability Evaluation

- Dataset: held-out tournament test split only
- Evaluation sample rows: `20,000`
- Test rows: `4,000`

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
| PySR (Symbolic Regression) | 0.51199 | 2,020 | 25.57 | 0.26 | 12.67 | 0.13 | 31.88% | 0 |
| XGBoost | 0.51257 | 1,396 | 10.70 | 0.11 | 14.45 | 0.14 | 50.29% | 0 |
| Machine Learning (Logistic Regression) | 0.51071 | 1,203 | 0.10 | 0.00 | 21.63 | 0.22 | 62.93% | 0 |
| Standard Gaussian | 0.50975 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00% | 0 |
| Momentum-Adjusted Gaussian | 0.51107 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00% | 0 |
| LightGBM | 0.51339 | 1,677 | -1.15 | -0.01 | 32.80 | 0.33 | 45.20% | 0 |
| Student-T df=3 | 0.51565 | 1,641 | -2.95 | -0.03 | 12.31 | 0.12 | 13.77% | 0 |
| Student-T df=5 | 0.51266 | 1,235 | -4.50 | -0.04 | 13.36 | 0.13 | 8.99% | 0 |
| CatBoost | 0.51292 | 1,288 | -5.38 | -0.05 | 27.79 | 0.28 | 52.95% | 0 |
