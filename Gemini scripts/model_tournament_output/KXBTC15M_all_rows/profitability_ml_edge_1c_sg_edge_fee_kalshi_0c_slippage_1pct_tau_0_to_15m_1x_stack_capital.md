# KXBTC15M Tournament Profitability Evaluation

- Dataset: held-out tournament test split only
- Evaluation sample rows: `11,934,913`
- Test rows: `2,386,983`
- Model scope: `ml`

## Default Assumptions

- Entry rule: buy `YES` when `model_prob - market_prob >= 1.00c`; buy `NO` when the reverse is true
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
| LightGBM | 0.49785 | 1,443,244 | 871,512 | 571,732 | 60.39% | 820939.67 | 25691.43 | 2569.14 | 934.30 | 73.29 | 0 |
| PySR (Symbolic Regression) | 0.50733 | 29,058 | 9,064 | 19,994 | 31.19% | 9554.97 | -999.97 | -100.00 | 1000.39 | 100.00 | 1,904,858 |
| XGBoost | 0.49987 | 148,573 | 87,345 | 61,228 | 58.79% | 85854.42 | -999.98 | -100.00 | 1110.83 | 100.00 | 1,100,216 |
| CatBoost | 0.49969 | 153,021 | 89,895 | 63,126 | 58.75% | 88306.90 | -999.99 | -100.00 | 1101.01 | 100.00 | 1,139,637 |
| Machine Learning (Logistic Regression) | 0.50232 | 65,807 | 43,112 | 22,695 | 65.51% | 42890.33 | -1000.00 | -100.00 | 1052.09 | 100.00 | 1,276,365 |
