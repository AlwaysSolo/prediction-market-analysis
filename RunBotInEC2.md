Use this as the clean EC2 checklist for the bagged-lasso live bot.

**1. SSH into EC2**
Use the correct EC2 SSH key and username for the AMI:

```bash
ssh -i ~/path/to/Trisecta-VA-ProdN.pem ec2-user@your-ec2-host
```

If the AMI is Ubuntu, use `ubuntu@...` instead of `ec2-user@...`.

**2. Clone the branch and enter the repo**
If the repo is not on the box yet:

```bash
git clone https://github.com/<your-user>/prediction-market-analysis.git
cd prediction-market-analysis
git switch bagged-lasso-live-rollout-v1
```

If it is already there:

```bash
cd prediction-market-analysis
git fetch origin
git switch bagged-lasso-live-rollout-v1
git pull
```

**3. Verify the live artifact exists**
The live runner needs the deploy artifact bundle:

```bash
ls artifacts/kalshi/kxbtc15m_bagged_lasso/latest
ls artifacts/kalshi/kxbtc15m_bagged_lasso/latest/bagged_lasso
```

You should see at least:
- `policy.json`
- `feature_manifest.json`
- `bagged_lasso/model.joblib`

**4. Install deps**
You already did this, but the command is:

```bash
uv sync
```

**5. Put your Kalshi API signing key on the server**
This is the Kalshi API private key, not the EC2 SSH key.

Example:

```bash
mkdir -p ~/.kalshi
chmod 700 ~/.kalshi
```

Copy your Kalshi API private key there, then:

```bash
chmod 600 ~/.kalshi/kalshi-prod-api.pem
```

**6. Create the runtime env file**
Create a local `.env` in the repo:

```bash
nano .env
```

Use this template:

```bash
KALSHI_PROD_API_KEY_ID=your_kalshi_api_key_id
KALSHI_PROD_PRIVATE_KEY_PATH=/home/ec2-user/.kalshi/kalshi-prod-api.pem
KALSHI_PROD_EXECUTION_ENABLE_LIVE_TRADING=true
KALSHI_PROD_EXECUTION_SUBACCOUNT=123
KALSHI_PROD_SIGNAL_APPLY_REGIME_HARD_GATE=true
KALSHI_PROD_SIGNAL_ENABLE_BUCKET_BAN_POLICY=true
```

Then load it:

```bash
set -a
source .env
set +a
```

**7. Sanity-check the env**
Make sure the important ones are loaded:

```bash
echo $KALSHI_PROD_API_KEY_ID
echo $KALSHI_PROD_PRIVATE_KEY_PATH
echo $KALSHI_PROD_EXECUTION_SUBACCOUNT
```

**8. Start a persistent shell session**
Use `tmux` so the bot survives your SSH disconnect:

```bash
tmux new -s kalshi-live
```

If `tmux` is missing, install it first with your distro package manager.

**9. Run shadow mode first**
This is the safest first pass: production data, no real orders.

```bash
cd ~/prediction-market-analysis
set -a
source .env
set +a

uv run python scripts/run_kalshi_bagged_lasso_live.py \
  --mode shadow \
  --log-root output/live/kalshi_bagged_lasso_live_v1
```

**10. What you should see before trusting it**
The runner should print:
- `Runner mode: shadow`
- `Environment: production`
- `Subaccount: <nonzero>`
- `Model file: ...bagged_lasso...model.joblib`
- `Policy file: ...policy.json`
- a `Subaccount preflight` line with balance and open positions

Then it should start printing:
- `SIGNAL ...`
- `EXECUTION ...`

That means the full stack is alive.

**11. Check that files are being written**
In another SSH session:

```bash
cd ~/prediction-market-analysis
find output/live/kalshi_bagged_lasso_live_v1 -maxdepth 4 -type f | head
```

You should see files under:
- `output/live/kalshi_bagged_lasso_live_v1/signal/...`
- `output/live/kalshi_bagged_lasso_live_v1/execution/...`
- `output/live/kalshi_bagged_lasso_live_v1/archive/...`

**12. Monitor logs live**
Useful quick checks:

```bash
tail -f output/live/kalshi_bagged_lasso_live_v1/signal/bagged_lasso/production/$(date +%F)/events.jsonl
```

```bash
tail -f output/live/kalshi_bagged_lasso_live_v1/execution/bagged_lasso/production/$(date +%F)/events.jsonl
```

**13. Review the dashboard from your laptop**
On EC2, in another `tmux` window or shell:

```bash
cd ~/prediction-market-analysis
uv run python -m http.server 8765
```

From your laptop, open an SSH tunnel:

```bash
ssh -i /path/to/Trisecta-VA-ProdN.pem -L 8765:localhost:8765 ec2-user@your-ec2-host
```

Then open locally:

```text
http://localhost:8765/tools/live_trading_dashboard.html
```

Choose:

```text
output/live/kalshi_bagged_lasso_live_v1
```

You do not need to expose port `8765` publicly if you use SSH tunneling.

**14. Generate a performance report**
After the bot has run for a while:

```bash
cd ~/prediction-market-analysis
uv run python scripts/analyze_kalshi_performance.py output/live/kalshi_bagged_lasso_live_v1 --mode live_execution
```

This writes reports under:

```text
artifacts/kalshi/performance_reports/kalshi_bagged_lasso_live_v1/
```

Main files to read:
- `report.md`
- `model_totals.csv`
- `bucket_summary.csv`
- `combo_summary.csv`
- `timeline_summary.csv`

**15. When shadow looks clean, switch to live**
Stop the shadow runner with `Ctrl+C`, then run:

```bash
cd ~/prediction-market-analysis
set -a
source .env
set +a

uv run python scripts/run_kalshi_bagged_lasso_live.py \
  --mode live \
  --confirm-live \
  --log-root output/live/kalshi_bagged_lasso_live_v1
```

This uses the same log root, so your run history stays together.

**16. Resume after disconnect or reboot**
SSH back in, reattach `tmux`, or restart with the same command and same `--log-root`:

```bash
tmux attach -t kalshi-live
```

If the process died, rerun the same command from step 15.

**17. Repair the archive after a crash**
If the machine dies mid-run:

```bash
uv run python scripts/repair_kalshi_live_archive.py \
  --archive-root output/live/kalshi_bagged_lasso_live_v1/archive \
  --environment production
```

**18. Safe go/no-go checklist before leaving it alone**
Make sure all of these are true:
- correct production Kalshi API key is loaded
- correct Kalshi signing key path is loaded
- subaccount is nonzero and intended for this bot
- shadow mode showed real `SIGNAL` and `EXECUTION` events
- `output/live/.../signal`, `execution`, and `archive` are all being written
- dashboard updates correctly
- performance analyzer runs without path issues
- only then switch to `--mode live --confirm-live`

If you want, I can next turn this into a copy-paste EC2 runbook file like `EC2LiveBotRunbook.md` inside the repo.