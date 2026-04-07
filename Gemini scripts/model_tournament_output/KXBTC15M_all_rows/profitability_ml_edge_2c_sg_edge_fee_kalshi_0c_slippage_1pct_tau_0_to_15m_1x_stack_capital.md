# KXBTC15M Tournament Profitability Evaluation

- Dataset: held-out tournament test split only
- Evaluation sample rows: `11,934,913`
- Test rows: `2,386,983`
- Model scope: `ml`

## Default Assumptions

- Entry rule: buy `YES` when `model_prob - market_prob >= 2.00c`; buy `NO` when the reverse is true
- Standard Gaussian override: none
- Exit rule: hold every filled trade to expiry
- Entry price: displayed market probability for `YES`, complementary price for `NO`, then apply `1.00%` adverse slippage
- Order size: `1` contract(s) per trade
- Trade window: only rows with `tau_minutes` between `0.00` and `15.00` inclusive
- Fees: `ceil(0.07 × contracts × price × (1 - price))` dollars, rounded up to the nearest cent
- Position rule: `multiple open positions per ticker allowed`
- Capital rule: `enforce available cash before entry`
- Drawdown basis: equity = free cash + open-position cost basis, so only fees and settled PnL move equity

## Results

| Model | Log-Loss | Trades | Wins | Losses | Win Rate | Total Invested ($) | Net PnL ($) | Return % | Max DD ($) | Max DD % | Skipped Capital |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.49786 | 816,333 | 492,361 | 323,972 | 60.31% | 452058.40 | 25885.53 | 258.86 | 391.58 | 3.30 | 0 |
| CatBoost | 0.49970 | 581,556 | 341,135 | 240,421 | 58.66% | 318143.76 | 12378.13 | 123.78 | 768.09 | 7.04 | 0 |
| XGBoost | 0.49988 | 525,398 | 314,921 | 210,477 | 59.94% | 295095.34 | 10436.82 | 104.37 | 750.29 | 6.87 | 0 |
| Machine Learning (Logistic Regression) | 0.50232 | 178,579 | 115,354 | 63,225 | 64.60% | 112752.97 | -860.21 | -8.60 | 1389.59 | 13.83 | 0 |
| PySR (Symbolic Regression) | 0.50735 | 352,116 | 96,132 | 255,984 | 27.30% | 100116.94 | -9999.97 | -100.00 | 10003.18 | 100.00 | 1,231,101 |
