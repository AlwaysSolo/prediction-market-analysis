# KXBTC15M Tournament Profitability Evaluation

- Dataset: held-out tournament test split only
- Evaluation sample rows: `11,934,913`
- Test rows: `2,001,881`
- Model scope: `ml`

## Default Assumptions

- Entry rule: buy `YES` when `model_prob - market_prob >= 2.00c`; buy `NO` when the reverse is true
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
| LightGBM | 0.49784 | 679,654 | 20256.53 | 202.57 | 460.60 | 4.38 | 59.24% | 0 |
| CatBoost | 0.49968 | 471,665 | 8527.09 | 85.27 | 1100.61 | 9.50 | 56.96% | 0 |
| XGBoost | 0.49986 | 417,546 | 6918.93 | 69.19 | 1020.03 | 9.21 | 58.14% | 0 |
| Machine Learning (Logistic Regression) | 0.50231 | 133,868 | -1080.43 | -10.80 | 1644.06 | 16.44 | 66.62% | 0 |
| PySR (Symbolic Regression) | 0.50732 | 356,234 | -9999.99 | -100.00 | 10003.91 | 100.00 | 27.10% | 1,060,515 |
