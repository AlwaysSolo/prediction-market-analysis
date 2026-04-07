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
- Position rule: `multiple open positions per ticker allowed`
- Capital rule: `enforce available cash before entry`
- Drawdown basis: equity = free cash + open-position cost basis, so only fees and settled PnL move equity

## Results

| Model | Log-Loss | Trades | Net PnL ($) | Return % | Max DD ($) | Max DD % | Win Rate | Skipped Capital |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.49784 | 816,430 | 44809.80 | 448.10 | 292.83 | 1.48 | 60.31% | 0 |
| CatBoost | 0.49968 | 581,564 | 26167.98 | 261.68 | 549.67 | 2.01 | 58.66% | 0 |
| XGBoost | 0.49986 | 525,272 | 22749.20 | 227.49 | 520.58 | 1.98 | 59.94% | 0 |
| Machine Learning (Logistic Regression) | 0.50230 | 187,322 | 3873.11 | 38.73 | 600.97 | 4.45 | 64.69% | 0 |
| Standard Gaussian | 0.50221 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00% | 0 |
| Momentum-Adjusted Gaussian | 0.50329 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00% | 0 |
| PySR (Symbolic Regression) | 0.50732 | 1,583,138 | -2378.22 | -23.78 | 8182.39 | 63.70 | 28.79% | 0 |
| Student-T df=5 | 0.50540 | 856,076 | -5496.81 | -54.97 | 6173.10 | 61.64 | 8.18% | 0 |
| Student-T df=3 | 0.50864 | 1,211,684 | -6201.92 | -62.02 | 7925.19 | 74.15 | 12.77% | 0 |
