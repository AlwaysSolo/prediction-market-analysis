# KXBTC15M Tournament Profitability Evaluation

- Dataset: held-out tournament test split only
- Evaluation sample rows: `11,934,913`
- Test rows: `2,001,881`
- Model scope: `ml`

## Default Assumptions

- Entry rule: buy `YES` when `model_prob - market_prob >= 3.00c`; buy `NO` when the reverse is true
- Standard Gaussian override: none
- Exit rule: hold every filled trade to expiry
- Entry price: displayed market probability for `YES`, complementary price for `NO`, then apply `1.00%` adverse slippage
- Order size: `1` contract(s) per trade
- Trade window: only rows with `tau_minutes` between `2.00` and `14.00` inclusive
- Fees: `ceil(0.07 × contracts × price × (1 - price))` dollars, rounded up to the nearest cent
- Position rule: `multiple open positions per ticker allowed`
- Capital rule: `enforce available cash before entry`
- Drawdown basis: equity = free cash + open-position cost basis, so only fees and settled PnL move equity

## Results

| Model | Log-Loss | Trades | Net PnL ($) | Return % | Max DD ($) | Max DD % | Win Rate | Skipped Capital |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.49784 | 348,889 | 15551.19 | 155.51 | 215.78 | 2.02 | 58.42% | 0 |
| CatBoost | 0.49969 | 182,976 | 6277.86 | 62.78 | 441.85 | 4.29 | 55.12% | 0 |
| XGBoost | 0.49986 | 129,391 | 4414.46 | 44.14 | 322.34 | 3.14 | 55.36% | 0 |
| Machine Learning (Logistic Regression) | 0.50231 | 21,429 | 166.97 | 1.67 | 95.09 | 0.94 | 61.20% | 0 |
| PySR (Symbolic Regression) | 0.50733 | 505,312 | -9999.95 | -100.00 | 10005.09 | 100.00 | 26.17% | 695,145 |
