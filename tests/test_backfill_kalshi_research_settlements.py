from __future__ import annotations

import json
from pathlib import Path

from scripts.backfill_kalshi_research_settlements import backfill_research_settlements


def _write_events(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_backfill_research_settlements_appends_settlement_and_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "live_research" / "run"
    events_path = run_dir / "research" / "lasso" / "demo" / "2026-01-01" / "events.jsonl"
    _write_events(
        events_path,
        [
            {
                "logged_at": "2026-01-01T12:00:00+00:00",
                "event_type": "research_sample_recorded",
                "payload": {
                    "sample_id": "sample-1",
                    "ticker": "KXBTC15M-TEST1",
                    "side": "YES",
                    "contracts": 1,
                    "reference_price_cents": 56,
                    "max_acceptable_entry_price_cents": 70,
                    "predicted_yes_probability": 0.78,
                    "predicted_no_probability": 0.22,
                    "chosen_side_probability": 0.78,
                    "feature_basis_market_prob": 0.55,
                    "raw_model_edge": 0.23,
                    "yes_post_cost_edge": 0.18,
                    "no_post_cost_edge": -0.28,
                    "chosen_post_cost_edge": 0.18,
                    "last_yes_price_cents": 55,
                    "last_price_cents": 55,
                    "yes_bid_cents": 54,
                    "yes_ask_cents": 56,
                    "buy_yes_price_cents": 56,
                    "buy_no_price_cents": 46,
                    "quote_mid_prob": 0.55,
                    "quote_spread_cents": 2,
                    "quote_age_seconds": 0.1,
                    "tau_minutes": 7.0,
                    "tau_bucket": "6-8",
                    "price_bucket": "50-60",
                    "chosen_side_probability_bucket": "70-80",
                    "chosen_side_edge_bucket": "10-20",
                    "estimated_entry_cost_dollars": 0.57,
                    "estimated_fees_dollars": 0.02,
                    "estimated_cash_required_dollars": 0.59,
                },
            }
        ],
    )

    patched = backfill_research_settlements(
        run_dir,
        ticker_results={"KXBTC15M-TEST1": "YES"},
        environment="demo",
    )

    assert patched == {"lasso": {"patched_samples": 1}}

    rows: list[dict] = []
    for path in sorted((run_dir / "research" / "lasso" / "demo").rglob("events.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    assert any(row["event_type"] == "research_sample_settled" for row in rows)
    assert any(row["event_type"] == "research_summary_snapshot" for row in rows)
    settlement = next(row for row in rows if row["event_type"] == "research_sample_settled")
    assert settlement["payload"]["sample_id"] == "sample-1"
    assert settlement["payload"]["settlement_result"] == "YES"
    assert settlement["payload"]["is_win"] is True
