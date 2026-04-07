# KXBTC15M Tournament Profitability Evaluation

- Dataset: held-out tournament test split only
- Evaluation sample rows: `20,000`
- Test rows: `4,000`
- Model scope: `lightgbm`

## Default Assumptions

- Entry rule: buy `YES` when `model_prob - market_prob >= 3.00c`; buy `NO` when the reverse is true
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
| LightGBM | 0.51364 | 1,513 | 692 | 821 | 45.74% | 701.75 | -37.79 | -0.38 | 54.53 | 0.55 | 0 |
