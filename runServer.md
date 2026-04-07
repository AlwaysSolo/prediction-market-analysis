Absolutely. Run these from c:\prediction-market-analysis in two separate PowerShell terminals.

1. Start the live paper run

uv run python scripts\run_kalshi_multi_model_execution_engine.py `
  --environment demo `
  --execution-mode paper `
  --run-dir "lasso=artifacts\kalshi\kxbtc15m_lasso\kxbtc15m_l1_compare_v1_lasso_l1r100" `
  --run-dir "elastic_net=artifacts\kalshi\kxbtc15m_elastic_net\kxbtc15m_elastic_net_l1r050_v1" `
  --run-dir "bagged_lasso=artifacts\kalshi\kxbtc15m_bagged_lasso\kxbtc15m_bagged_lasso_v1" `
  --run-dir "linear_svm=artifacts\kalshi\kxbtc15m_linear_svm\kxbtc15m_linear_svm_v1" `
  --kelly-fraction-multiplier 0.25 `
  --kelly-fraction-cap-pct 2.0
2. Start the local dashboard server

uv run python -m http.server 8765
Then open:

http://localhost:8765/tools/live_trading_dashboard.html
Once the page opens, choose the folder:

output/live/kalshi
If the dashboard looks stale, do a hard refresh with Ctrl+F5.


If you were running the default 4-model research run before, use the same command again with the same --log-root:

uv run python scripts\run_kalshi_research_multi_model.py `
  --environment demo `
  --log-root output\live_research\kalshi_bucket_discovery `
  --min-edge-cents 2
That resumes against the existing output\live_research\kalshi_bucket_discovery folder.

If you had been running the 8-model version instead, use the same 8-model command from runServer.md. The key is: keep the same --log-root so it picks up the same run.


new paper run:
uv run python scripts\run_kalshi_research_multi_model.py `
  --environment demo `
  --log-root output\live_research\kalshi_bucket_discovery_archive `
  --min-edge-cents 2 `
  --archive full


=====================================
For the **4-model research demo run** with the new regime hard gate, use:
$env:APPLY_REGIME_HARD_GATE="true"
uv run python scripts\run_kalshi_research_multi_model.py `
  --environment demo `
  --log-root output\live_research\kalshi_bucket_discovery_regime `
  --min-edge-cents 2 `
  --min-tau-minutes 0 `
  --max-tau-minutes 15 `
  --price-band-min-cents 0 `
  --price-band-max-cents 100 `
  --archive full

If you want the **broader sampling** version so it doesn’t get filtered out as much by price/tau:

```powershell
$env:APPLY_REGIME_HARD_GATE="true"
uv run python scripts\run_kalshi_research_multi_model.py `
  --environment demo `
  --log-root output\live_research\kalshi_bucket_discovery_regime `
  --min-edge-cents 2 `
  --min-tau-minutes 0 `
  --max-tau-minutes 15 `
  --price-band-min-cents 0 `
  --price-band-max-cents 100 `
  --archive full
```

If instead you mean the **actual paper-trading multi-model demo** rather than research-only sampling, use:

```powershell
$env:APPLY_REGIME_HARD_GATE="true"
uv run python scripts\run_kalshi_multi_model_execution_engine.py `
  --environment demo `
  --log-root output\live\kalshi_all_models_regime_demo
```

`APPLY_REGIME_HARD_GATE=true` makes the regime behavior explicit, even though the env-based config now defaults to it.


$env:APPLY_REGIME_HARD_GATE="true"
uv run python scripts\run_kalshi_research_multi_model.py `
  --environment production `
  --log-root output\live_research\kalshi_bucket_discovery_regime_prod `
  --min-edge-cents 2 `
  --min-tau-minutes 0 `
  --max-tau-minutes 15 `
  --price-band-min-cents 0 `
  --price-band-max-cents 100 `
  --archive full

=========================================================================================
Use the research runner in `production` with archive mode on. This uses live production market data but still does **not** place real bets.

```powershell
$env:KALSHI_PROD_RESEARCH_APPLY_REGIME_HARD_GATE="true"
$env:KALSHI_PROD_RESEARCH_ENABLE_BUCKET_BAN_POLICY="true"

uv run python scripts\run_kalshi_research_multi_model.py `
  --environment production `
  --model-family lasso `
  --model-family elastic_net `
  --model-family bagged_lasso `
  --model-family linear_svm `
  --log-root output\live_research\kalshi_bucket_discovery_bucketban_prod `
  --min-edge-cents 2 `
  --min-tau-minutes 0 `
  --max-tau-minutes 15 `
  --price-band-min-cents 0 `
  --price-band-max-cents 100 `
  --archive full
```

If you want the dashboard too:

```powershell
uv run python -m http.server 8765
```

Then open:

```text
http://localhost:8765/tools/live_research_dashboard.html
```

and select:

```text
output/live_research/kalshi_bucket_discovery_bucketban_prod
```

If you want to resume the same run later, rerun the exact same command with the same `--log-root`.

========================================================
Live Mode:

Implemented the bagged-lasso live rollout end to end.

Key pieces are in [run_kalshi_bagged_lasso_live.py](/c:/prediction-market-analysis/scripts/run_kalshi_bagged_lasso_live.py), [execution.py](/c:/prediction-market-analysis/src/live/kalshi/execution.py), [client.py](/c:/prediction-market-analysis/src/live/kalshi/client.py), [live_archive.py](/c:/prediction-market-analysis/src/live/kalshi/live_archive.py), and [runServer.md](/c:/prediction-market-analysis/runServer.md). The dedicated runner is now single-model `bagged_lasso` only, forces regime hard gate + bucket bans + fixed 1-contract sizing + no stacking, supports `shadow` and `live`, requires `--confirm-live` for live mode, and refuses live trading without a nonzero production subaccount and valid credentials. Order submission and order recovery are now subaccount-aware, and the live path writes a full archive under `output/live/<run_name>/archive/...` while keeping the existing dashboard logs intact.

One important behavior change: `shadow` mode now acts like production-data paper trading instead of “reject every trade.” That gives you a clean operator workflow: production quotes, no real orders, but realistic signal/execution/archive flow without the resubmission loop that a pure reject path caused.

Verification passed:
- `uv run pytest tests/test_kalshi_execution.py tests/test_kalshi_live_archive.py tests/test_kalshi_bagged_lasso_live_runner.py tests/test_kalshi_signal_risk.py tests/test_kalshi_research.py tests/test_kalshi_performance_analysis.py -q`
- result: `65 passed`
- `python -m py_compile scripts/run_kalshi_bagged_lasso_live.py scripts/repair_kalshi_live_archive.py src/live/kalshi/execution.py src/live/kalshi/client.py src/live/kalshi/live_archive.py src/live/kalshi/__init__.py`

Use these commands:
```powershell
$env:KALSHI_PROD_API_KEY_ID="..."
$env:KALSHI_PROD_PRIVATE_KEY_PATH="..."
$env:KALSHI_PROD_EXECUTION_ENABLE_LIVE_TRADING="true"
$env:KALSHI_PROD_EXECUTION_SUBACCOUNT="123"
$env:KALSHI_PROD_SIGNAL_APPLY_REGIME_HARD_GATE="true"
$env:KALSHI_PROD_SIGNAL_ENABLE_BUCKET_BAN_POLICY="true"
```

```powershell
uv run python scripts\run_kalshi_bagged_lasso_live.py `
  --mode shadow `
  --log-root output\live\kalshi_bagged_lasso_live_v1
```

```powershell
uv run python scripts\run_kalshi_bagged_lasso_live.py `
  --mode live `
  --confirm-live `
  --log-root output\live\kalshi_bagged_lasso_live_v1
```

```powershell
uv run python scripts\repair_kalshi_live_archive.py `
  --archive-root output\live\kalshi_bagged_lasso_live_v1\archive `
  --environment production
```

I did not run the repo’s full test suite.
