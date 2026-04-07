from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.report_live_kalshi_research_buckets import generate_research_bucket_report


def _write_events(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_generate_research_bucket_report_outputs_csv_and_markdown(tmp_path: Path) -> None:
    run_dir = tmp_path / "live_research" / "my_research_run"
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
            },
            {
                "logged_at": "2026-01-01T12:10:00+00:00",
                "event_type": "research_sample_settled",
                "payload": {
                    "sample_id": "sample-1",
                    "ticker": "KXBTC15M-TEST1",
                    "side": "YES",
                    "settlement_result": "YES",
                    "is_win": True,
                    "contracts": 1,
                    "cash_required_dollars": 0.59,
                    "realized_pnl_dollars": 0.41,
                    "cumulative_realized_pnl_dollars": 0.41,
                },
            },
            {
                "logged_at": "2026-01-01T12:01:00+00:00",
                "event_type": "research_sample_recorded",
                "payload": {
                    "sample_id": "sample-2",
                    "ticker": "KXBTC15M-TEST2",
                    "side": "NO",
                    "contracts": 1,
                    "reference_price_cents": 46,
                    "max_acceptable_entry_price_cents": 60,
                    "predicted_yes_probability": 0.22,
                    "predicted_no_probability": 0.78,
                    "chosen_side_probability": 0.78,
                    "feature_basis_market_prob": 0.55,
                    "raw_model_edge": -0.33,
                    "yes_post_cost_edge": -0.28,
                    "no_post_cost_edge": 0.18,
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
                    "tau_minutes": 9.0,
                    "tau_bucket": "8-10",
                    "price_bucket": "40-50",
                    "chosen_side_probability_bucket": "70-80",
                    "chosen_side_edge_bucket": "10-20",
                    "estimated_entry_cost_dollars": 0.46,
                    "estimated_fees_dollars": 0.02,
                    "estimated_cash_required_dollars": 0.48,
                },
            },
            {
                "logged_at": "2026-01-01T12:11:00+00:00",
                "event_type": "research_sample_settled",
                "payload": {
                    "sample_id": "sample-2",
                    "ticker": "KXBTC15M-TEST2",
                    "side": "NO",
                    "settlement_result": "NO",
                    "is_win": True,
                    "contracts": 1,
                    "cash_required_dollars": 0.48,
                    "realized_pnl_dollars": 0.52,
                    "cumulative_realized_pnl_dollars": 0.93,
                },
            },
        ],
    )

    outputs = generate_research_bucket_report(run_dir, artifacts_root=tmp_path / "artifacts")

    assert outputs["samples_csv"].exists()
    assert outputs["aggregate_csv"].exists()
    assert outputs["per_model_csv"].exists()
    assert outputs["report_md"].exists()

    markdown = outputs["report_md"].read_text(encoding="utf-8")
    assert "Model Totals" in markdown
    assert "Aggregate Price Buckets" in markdown

    with outputs["aggregate_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["dimension"] == "price" and row["bucket"] == "40-50" for row in rows)

    with outputs["samples_csv"].open("r", encoding="utf-8", newline="") as handle:
        sample_rows = list(csv.DictReader(handle))
    assert len(sample_rows) == 2
    assert sample_rows[0]["status"] == "settled"
