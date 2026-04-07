# KXBTC15M Tournament Profitability Evaluation

- Dataset: held-out tournament test split only
- Evaluation sample rows: `11,934,913`
- Test rows: `2,386,983`
- Model scope: `ml`

## Default Assumptions

- Entry rule: buy `YES` when `model_prob - market_prob >= 5.00c`; buy `NO` when the reverse is true
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

| Model | Log-Loss | Trades | Net PnL ($) | Return % | Max DD ($) | Max DD % | Win Rate | Skipped Capital |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.49784 | 125,348 | 10103.31 | 101.03 | 68.42 | 0.52 | 61.20% | 0 |
| CatBoost | 0.49969 | 52,560 | 4473.65 | 44.74 | 60.21 | 0.53 | 61.99% | 0 |
| XGBoost | 0.49986 | 40,198 | 3390.19 | 33.90 | 67.79 | 0.53 | 63.86% | 0 |
| Machine Learning (Logistic Regression) | 0.50231 | 12,884 | 715.31 | 7.15 | 19.16 | 0.19 | 56.83% | 0 |
| PySR (Symbolic Regression) | 0.50734 | 454,848 | -9999.99 | -100.00 | 10005.70 | 100.00 | 21.73% | 273,923 |
