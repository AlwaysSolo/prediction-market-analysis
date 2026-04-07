# KXBTC15M Tournament Profitability Evaluation

- Dataset: held-out tournament test split only
- Evaluation sample rows: `11,934,913`
- Test rows: `2,386,983`
- Model scope: `ml`

## Default Assumptions

- Entry rule: buy `YES` when `model_prob - market_prob >= 6.00c`; buy `NO` when the reverse is true
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
| LightGBM | 0.49784 | 73,016 | 45,137 | 27,879 | 61.82% | 36698.42 | 7117.12 | 711.71 | 51.10 | 2.37 | 0 |
| CatBoost | 0.49969 | 32,610 | 20,682 | 11,928 | 63.42% | 16751.03 | 3346.03 | 334.60 | 53.24 | 1.80 | 0 |
| XGBoost | 0.49986 | 26,042 | 16,734 | 9,308 | 64.26% | 13571.36 | 2718.25 | 271.83 | 52.89 | 2.05 | 0 |
| Machine Learning (Logistic Regression) | 0.50231 | 9,388 | 5,297 | 4,091 | 56.42% | 4468.88 | 650.06 | 65.01 | 11.77 | 1.10 | 0 |
| PySR (Symbolic Regression) | 0.50733 | 31,531 | 6,258 | 25,273 | 19.85% | 6700.14 | -999.92 | -99.99 | 1003.18 | 99.99 | 223,800 |
