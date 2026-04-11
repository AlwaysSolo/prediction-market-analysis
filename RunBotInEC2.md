Use this as the EC2 runbook for the exact-match bagged-lasso layering rollout.

The goal of this rollout is simple:
- match the winning shadow configuration as closely as possible
- use a tiny isolated production subaccount
- keep the first live day operationally tight

Exact config we are preserving:
- `--signal-profile dedicated-v1`
- `--enable-layering`
- `--allow-stacking`
- `--disable-regime-control`
- `--edge-threshold-cents 2`

Recommended bankroll for the first live rollout:
- dedicated subaccount funded with about `$25`

**1. SSH into EC2**

```bash
ssh -i ~/path/to/Trisecta-VA-ProdN.pem ec2-user@your-ec2-host
```

Use `ubuntu@...` instead of `ec2-user@...` if your AMI is Ubuntu.

**2. Update the repo**

```bash
cd ~/prediction-market-analysis
git fetch origin
git switch <your-live-rollout-branch>
git pull
```

**3. Verify the bagged-lasso artifact bundle**

```bash
ls artifacts/kalshi/kxbtc15m_bagged_lasso/latest
ls artifacts/kalshi/kxbtc15m_bagged_lasso/latest/bagged_lasso
```

Required files:
- `policy.json`
- `feature_manifest.json`
- `bagged_lasso/model.joblib`

**4. Install dependencies**

```bash
uv sync
```

**5. Put the Kalshi signing key on the box**

```bash
mkdir -p ~/.kalshi
chmod 700 ~/.kalshi
chmod 600 ~/.kalshi/kalshi-prod-api.pem
```

**6. Create the runtime env file**

Create `.env` in the repo root:

```bash
cat > .env <<'EOF'
KALSHI_PROD_API_KEY_ID=your_kalshi_api_key_id
KALSHI_PROD_PRIVATE_KEY_PATH=/home/ec2-user/.kalshi/kalshi-prod-api.pem
KALSHI_PROD_EXECUTION_SUBACCOUNT=123
KALSHI_PROD_EXECUTION_ENABLE_LIVE_TRADING=true
EOF
```

Important:
- `KALSHI_PROD_EXECUTION_SUBACCOUNT` must be the numeric Kalshi `subaccount_number`
- it is not a UI label, UUID, or arbitrary account identifier
- the runner now validates this at startup and will print the valid subaccount numbers returned by Kalshi

Load it:

```bash
set -a
source .env
set +a
```

**7. Sanity-check the env**

```bash
echo $KALSHI_PROD_API_KEY_ID
echo $KALSHI_PROD_PRIVATE_KEY_PATH
echo $KALSHI_PROD_EXECUTION_SUBACCOUNT
echo $KALSHI_PROD_EXECUTION_ENABLE_LIVE_TRADING
```

**8. Start a persistent shell**

```bash
tmux new -s kalshi-live
```

**9. Run the exact-match shadow smoke on EC2 first**

Use a fresh shadow log root:

```bash
cd ~/prediction-market-analysis
set -a
source .env
set +a

uv run python scripts/run_kalshi_bagged_lasso_live.py \
  --mode shadow \
  --enable-layering \
  --allow-stacking \
  --disable-regime-control \
  --edge-threshold-cents 2 \
  --signal-profile dedicated-v1 \
  --log-root output/live/kalshi_bagged_lasso_shadow_ec2_exact_shadow_edge2
```

What to confirm:
- `Runner mode: shadow`
- `Environment: production`
- `Subaccount: <nonzero>`
- `regime=False`
- `edge=2.0c`
- `stacking=True`
- `layering` activity shows up in the logs/dashboard
- no `consume_error`
- no `Unknown reservation decision id`

**10. Start the dashboard server**

```bash
cd ~/prediction-market-analysis
uv run python -m http.server 8765
```

From your laptop:

```bash
ssh -i /path/to/Trisecta-VA-ProdN.pem -L 8765:localhost:8765 ec2-user@your-ec2-host
```

Then open:

```text
http://localhost:8765/tools/live_trading_dashboard.html
```

Choose the shadow folder:

```text
output/live/kalshi_bagged_lasso_shadow_ec2_exact_shadow_edge2
```

**11. Watch the right live-health signals**

Watch these first:
- blocked signal breakdown
- claimed trades
- execution errors / consume errors
- portfolio snapshot
- tranche cadence by `10m`, `5m`, `3m`, `1m`

For the shadow smoke, the only hard requirement is that the stack behaves cleanly with the exact rollout flags.

**12. Stop the shadow smoke and launch live**

Use a new live log root. Do not reuse the shadow folder.

```bash
cd ~/prediction-market-analysis
set -a
source .env
set +a

uv run python scripts/run_kalshi_bagged_lasso_live.py \
  --mode live \
  --confirm-live \
  --enable-layering \
  --allow-stacking \
  --disable-regime-control \
  --edge-threshold-cents 2 \
  --signal-profile dedicated-v1 \
  --log-root output/live/kalshi_bagged_lasso_live_exact_shadow_edge2
```

What to confirm immediately:
- `Runner mode: live`
- `Environment: production`
- `Subaccount: <nonzero>`
- `regime=False`
- `edge=2.0c`
- `stacking=True`
- `layering` decisions begin to appear
- no startup exceptions

**13. Manual stop rules for the first live day**

Stop the run immediately if any of these happen:
- `Unknown reservation decision id`
- repeated order submit or reconcile errors
- dashboard stops updating
- event logs stall unexpectedly
- live fills look clearly inconsistent with the visible book for multiple trades
- unexpected open positions appear outside the strategy’s expected thesis flow

Default daily stop for the first live rollout:
- stop if realized drawdown reaches `$5`
- stop if two operational errors occur, even if PnL is still positive

**14. Tail the logs directly if needed**

```bash
tail -f output/live/kalshi_bagged_lasso_live_exact_shadow_edge2/signal/bagged_lasso/production/$(date +%F)/events.jsonl
```

```bash
tail -f output/live/kalshi_bagged_lasso_live_exact_shadow_edge2/execution/bagged_lasso/production/$(date +%F)/events.jsonl
```

```bash
tail -f output/live/kalshi_bagged_lasso_live_exact_shadow_edge2/layering/bagged_lasso/production/$(date +%F)/events.jsonl
```

**15. Generate the post-run report**

```bash
cd ~/prediction-market-analysis
uv run python scripts/analyze_kalshi_performance.py \
  output/live/kalshi_bagged_lasso_live_exact_shadow_edge2 \
  --mode live_execution \
  --environment production \
  --output-dir artifacts/kalshi/performance_reports/kalshi_bagged_lasso_live_exact_shadow_edge2
```

Then compare live against the winning shadow run on:
- thesis count
- tranche count
- fill rate
- realized PnL
- `YES` vs `NO`
- `10m`, `5m`, `3m`, `1m`
- rejection reasons
- execution errors

**16. Resume after disconnect**

```bash
tmux attach -t kalshi-live
```

If the process died, rerun the exact same live command with the same `--log-root`.

**17. Archive repair if the process crashes**

```bash
uv run python scripts/repair_kalshi_live_archive.py \
  --archive-root output/live/kalshi_bagged_lasso_live_exact_shadow_edge2/archive \
  --environment production
```

**18. Go/no-go checklist before leaving it alone**

All of these should be true:
- production Kalshi API key is correct
- production signing key path is correct
- subaccount is the isolated live-test subaccount
- subaccount size is intentionally tiny for rollout one
- EC2 shadow smoke with the exact flags ran cleanly
- dashboard is reading the exact live log root
- no reservation/execution handoff errors are appearing
- only then let the full live day run
