# KXBTC15M Tournament Profitability Evaluation

- Dataset: held-out tournament test split only
- Evaluation sample rows: `11,934,913`
- Test rows: `2,386,983`
- Model scope: `lightgbm`

## Default Assumptions

- Entry rule: buy `YES` when `model_prob - market_prob >= 4.00c`; buy `NO` when the reverse is true
- Standard Gaussian override: none
- Exit rule: hold every filled trade to expiry
- Entry price: displayed market probability for `YES`, complementary price for `NO`, then apply `1.00%` adverse slippage
- Order size: `10` contract(s) per trade
- Trade window: only rows with `tau_minutes` between `0.00` and `15.00` inclusive
- Fees: `ceil(0.07 × contracts × price × (1 - price))` dollars, rounded up to the nearest cent
- Position rule: `multiple open positions per ticker allowed`
- Capital rule: `enforce available cash before entry`
- Reserve rule: keep at least `30.00%` of current portfolio equity uninvested
- Drawdown basis: equity = free cash + open-position cost basis, so only fees and settled PnL move equity

## Results

| Model | Log-Loss | Trades | Wins | Losses | Win Rate | Total Invested ($) | Net PnL ($) | Return % | Max DD ($) | Max DD % | Skipped Capital | Skipped Reserve |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.49783 | 230,757 | 139,634 | 91,123 | 60.51% | 1207068.47 | 156156.67 | 15615.67 | 1080.49 | 9.01 | 0 | 0 |
