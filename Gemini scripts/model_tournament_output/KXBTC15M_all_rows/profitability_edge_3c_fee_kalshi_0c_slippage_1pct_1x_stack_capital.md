# KXBTC15M Tournament Profitability Evaluation

- Dataset: held-out tournament test split only
- Evaluation sample rows: `11,934,913`
- Test rows: `2,386,983`

## Default Assumptions

- Entry rule: buy `YES` when `model_prob - market_prob >= 3.00c`; buy `NO` when the reverse is true
- Exit rule: hold every filled trade to expiry
- Entry price: displayed market probability for `YES`, complementary price for `NO`, then apply `1.00%` adverse slippage
- Order size: `1` contract(s) per trade
- Fees: `ceil(0.07 × contracts × price × (1 - price))` dollars, rounded up to the nearest cent
- Position rule: `multiple open positions per ticker allowed`
- Capital rule: `enforce available cash before entry`
- Drawdown basis: equity = free cash + open-position cost basis, so only fees and settled PnL move equity

## Results

| Model | Log-Loss | Trades | Net PnL ($) | Return % | Max DD ($) | Max DD % | Win Rate | Skipped Capital |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.49785 | 438,002 | 20591.19 | 205.91 | 204.13 | 1.51 | 60.13% | 0 |
| CatBoost | 0.49969 | 247,145 | 9700.78 | 97.01 | 350.26 | 3.31 | 58.21% | 0 |
| XGBoost | 0.49987 | 190,432 | 7500.50 | 75.01 | 247.01 | 2.35 | 59.52% | 0 |
| Machine Learning (Logistic Regression) | 0.50231 | 36,872 | 742.85 | 7.43 | 47.56 | 0.47 | 59.61% | 0 |
| Standard Gaussian | 0.50222 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00% | 0 |
| Momentum-Adjusted Gaussian | 0.50329 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00% | 0 |
| Student-T df=5 | 0.50540 | 178,126 | -3179.40 | -31.79 | 3267.12 | 32.67 | 4.70% | 0 |
| Student-T df=3 | 0.50865 | 656,774 | -9999.99 | -100.00 | 10003.81 | 100.00 | 9.00% | 242,384 |
| PySR (Symbolic Regression) | 0.50733 | 510,365 | -10000.00 | -100.00 | 10004.55 | 100.00 | 26.21% | 811,343 |
