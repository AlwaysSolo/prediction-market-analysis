# KXBTC15M Tournament Profitability Evaluation

- Dataset: held-out tournament test split only
- Evaluation sample rows: `11,934,913`
- Test rows: `2,386,983`

## Default Assumptions

- Entry rule: buy `YES` when `model_prob - market_prob >= 3.00c`; buy `NO` when the reverse is true
- Standard Gaussian override: use market `z = norm.ppf(price)` and trade when `|z| >= 0.07527`
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
| LightGBM | 0.49783 | 438,046 | 20558.41 | 205.58 | 200.85 | 1.45 | 60.13% | 0 |
| CatBoost | 0.49968 | 247,190 | 9790.93 | 97.91 | 343.80 | 3.24 | 58.26% | 0 |
| XGBoost | 0.49985 | 190,366 | 7529.91 | 75.30 | 251.72 | 2.39 | 59.55% | 0 |
| Machine Learning (Logistic Regression) | 0.50230 | 37,338 | 700.27 | 7.00 | 46.59 | 0.45 | 59.58% | 0 |
| Momentum-Adjusted Gaussian | 0.50328 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00% | 0 |
| Student-T df=5 | 0.50539 | 178,133 | -3184.60 | -31.85 | 3272.49 | 32.72 | 4.70% | 0 |
| Standard Gaussian | 0.50221 | 575,709 | -9999.74 | -100.00 | 10008.29 | 100.00 | 72.74% | 1,615,244 |
| PySR (Symbolic Regression) | 0.50731 | 510,175 | -9999.98 | -100.00 | 10004.54 | 100.00 | 26.21% | 811,391 |
| Student-T df=3 | 0.50864 | 656,792 | -9999.99 | -100.00 | 10003.85 | 100.00 | 9.00% | 242,394 |
