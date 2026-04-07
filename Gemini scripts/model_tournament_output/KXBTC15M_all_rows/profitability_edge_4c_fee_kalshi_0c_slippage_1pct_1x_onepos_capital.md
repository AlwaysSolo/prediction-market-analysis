# KXBTC15M Tournament Profitability Evaluation

- Dataset: held-out tournament test split only
- Evaluation sample rows: `11,934,913`
- Test rows: `2,386,983`

## Default Assumptions

- Entry rule: buy `YES` when `model_prob - market_prob >= 4.00c`; buy `NO` when the reverse is true
- Exit rule: hold every filled trade to expiry
- Entry price: displayed market probability for `YES`, complementary price for `NO`, then apply `1.00%` adverse slippage
- Order size: `1` contract(s) per trade
- Fees: `ceil(0.07 × contracts × price × (1 - price))` dollars, rounded up to the nearest cent
- Position rule: `one open position per ticker at a time`
- Capital rule: `enforce available cash before entry`
- Drawdown basis: equity = free cash + open-position cost basis, so only fees and settled PnL move equity

## Results

| Model | Log-Loss | Trades | Net PnL ($) | Return % | Max DD ($) | Max DD % | Win Rate | Skipped Capital |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.49784 | 8,176 | 289.05 | 2.89 | 14.37 | 0.14 | 50.57% | 0 |
| XGBoost | 0.49987 | 7,017 | 246.98 | 2.47 | 14.46 | 0.14 | 44.71% | 0 |
| CatBoost | 0.49969 | 7,357 | 219.18 | 2.19 | 16.23 | 0.16 | 44.62% | 0 |
| Machine Learning (Logistic Regression) | 0.50232 | 5,541 | 136.78 | 1.37 | 26.18 | 0.26 | 56.58% | 0 |
| Student-T df=3 | 0.50867 | 8,187 | 18.84 | 0.19 | 35.69 | 0.36 | 12.51% | 0 |
| Standard Gaussian | 0.50224 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00% | 0 |
| Momentum-Adjusted Gaussian | 0.50332 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00% | 0 |
| Student-T df=5 | 0.50542 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00% | 0 |
| PySR (Symbolic Regression) | 0.50735 | 8,468 | -18.71 | -0.19 | 46.74 | 0.47 | 33.64% | 0 |
