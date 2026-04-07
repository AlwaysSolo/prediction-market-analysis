# KXBTC15M Tournament Profitability Evaluation

- Dataset: held-out tournament test split only
- Evaluation sample rows: `11,934,913`
- Test rows: `2,386,983`
- Model scope: `ml`

## Default Assumptions

- Entry rule: buy `YES` when `model_prob - market_prob >= 4.00c`; buy `NO` when the reverse is true
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

| Model | Log-Loss | Trades | Total Invested ($) | Net PnL ($) | Return % | Max DD ($) | Max DD % | Win Rate | Skipped Capital |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.49782 | 230,748 | 120724.43 | 14716.55 | 147.17 | 124.91 | 0.81 | 60.52% | 0 |
| CatBoost | 0.49968 | 107,490 | 56168.87 | 6606.11 | 66.06 | 124.65 | 0.82 | 60.27% | 0 |
| XGBoost | 0.49985 | 76,241 | 40657.86 | 4790.96 | 47.91 | 72.70 | 0.59 | 61.40% | 0 |
| Machine Learning (Logistic Regression) | 0.50230 | 19,771 | 10256.39 | 843.14 | 8.43 | 20.51 | 0.20 | 58.04% | 0 |
| PySR (Symbolic Regression) | 0.50733 | 521,236 | 123673.03 | -9999.95 | -100.00 | 10003.74 | 100.00 | 23.54% | 501,758 |
