from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_performance_scorecard import (
    FILL_REALISM_FIX_COMMIT,
    generate_performance_scorecard,
)
from src.live.kalshi.performance_analysis import generate_kalshi_performance_reports


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _make_live_execution_run(
    base: Path,
    *,
    run_name: str,
    mode: str,
    trade_rows: list[dict],
    model: str = "lasso",
) -> Path:
    run_dir = base / "output" / "live" / run_name
    signal_path = run_dir / "signal" / model / "production" / trade_rows[0]["trade_date"] / "events.jsonl"
    execution_path = run_dir / "execution" / model / "production" / trade_rows[0]["trade_date"] / "events.jsonl"

    signal_rows: list[dict] = []
    execution_rows: list[dict] = [
        {
            "logged_at": trade_rows[0]["recorded_at"],
            "event_type": "execution_started",
            "payload": {
                "environment": "production",
                "mode": mode,
                "simulate_immediate_fills": False,
                "shadow_fill_latency_seconds": 0.25 if mode == "paper" else None,
            },
        }
    ]
    cumulative_pnl = 0.0
    for trade in trade_rows:
        predicted_yes_probability = trade["predicted_yes_probability"]
        signal_rows.append(
            {
                "logged_at": trade["recorded_at"],
                "event_type": "signal_decision",
                "payload": {
                    "approved": True,
                    "decision_id": trade["decision_id"],
                    "ticker": trade["ticker"],
                    "side": trade["side"],
                    "predicted_yes_probability": predicted_yes_probability,
                    "predicted_no_probability": 1.0 - predicted_yes_probability,
                    "feature_basis_market_prob": trade["feature_basis_market_prob"],
                    "raw_model_edge": predicted_yes_probability - trade["feature_basis_market_prob"],
                    "yes_post_cost_edge": trade["yes_post_cost_edge"],
                    "no_post_cost_edge": trade["no_post_cost_edge"],
                    "tau_minutes": trade["tau_minutes"],
                    "reference_price_cents": trade["reference_price_cents"],
                    "quote_spread_cents": trade["quote_spread_cents"],
                    "quote_age_seconds": trade["quote_age_seconds"],
                    "regime_label": trade["regime_label"],
                    "bearish_vote_count": trade.get("bearish_vote_count", 0),
                    "bullish_vote_count": trade.get("bullish_vote_count", 0),
                },
            }
        )
        cumulative_pnl += trade["realized_pnl_dollars"]
        execution_rows.append(
            {
                "logged_at": trade["settled_at"],
                "event_type": "simulated_position_settled",
                "payload": {
                    "decision_id": trade["decision_id"],
                    "ticker": trade["ticker"],
                    "side": trade["side"],
                    "settlement_result": trade["settlement_result"],
                    "contracts": 1,
                    "cash_required_dollars": trade["cash_required_dollars"],
                    "realized_pnl_dollars": trade["realized_pnl_dollars"],
                    "cumulative_realized_pnl_dollars": cumulative_pnl,
                },
            }
        )

    _write_jsonl(signal_path, signal_rows)
    _write_jsonl(execution_path, execution_rows)
    return run_dir


def test_scorecard_prefers_report_over_raw_duplicate(tmp_path: Path) -> None:
    run_dir = _make_live_execution_run(
        tmp_path,
        run_name="btc_live_dedupe",
        mode="live",
        trade_rows=[
            {
                "trade_date": "2026-04-15",
                "recorded_at": "2026-04-15T16:00:00+00:00",
                "settled_at": "2026-04-15T16:10:00+00:00",
                "decision_id": "dec-1",
                "ticker": "KXBTC15M-TEST1",
                "side": "YES",
                "predicted_yes_probability": 0.70,
                "feature_basis_market_prob": 0.60,
                "yes_post_cost_edge": 0.08,
                "no_post_cost_edge": -0.18,
                "tau_minutes": 8.0,
                "reference_price_cents": 55,
                "quote_spread_cents": 1,
                "quote_age_seconds": 0.2,
                "regime_label": "neutral",
                "realized_pnl_dollars": 0.40,
                "cash_required_dollars": 0.60,
                "settlement_result": "YES",
            },
            {
                "trade_date": "2026-04-15",
                "recorded_at": "2026-04-15T17:00:00+00:00",
                "settled_at": "2026-04-15T17:09:00+00:00",
                "decision_id": "dec-2",
                "ticker": "KXBTC15M-TEST2",
                "side": "NO",
                "predicted_yes_probability": 0.40,
                "feature_basis_market_prob": 0.50,
                "yes_post_cost_edge": -0.16,
                "no_post_cost_edge": 0.07,
                "tau_minutes": 7.0,
                "reference_price_cents": 48,
                "quote_spread_cents": 1,
                "quote_age_seconds": 0.1,
                "regime_label": "uptrend",
                "realized_pnl_dollars": -0.20,
                "cash_required_dollars": 0.20,
                "settlement_result": "YES",
            },
        ],
    )
    reports_root = tmp_path / "reports"
    generate_kalshi_performance_reports(run_dir, output_dir=reports_root / run_dir.name)

    outputs = generate_performance_scorecard(
        report_root=reports_root,
        live_roots=(tmp_path / "output" / "live",),
        live_research_roots=(),
        output_dir=tmp_path / "scorecard",
        as_of_date=date(2026, 4, 17),
    )
    run_scorecard = pd.read_csv(outputs["run_scorecard_csv"])

    assert len(run_scorecard) == 1
    row = run_scorecard.iloc[0]
    assert row["run_name"] == "btc_live_dedupe"
    assert row["data_source"] == "performance_report"
    assert row["net_pnl_after_fees_dollars"] == pytest.approx(0.20)


def test_scorecard_flags_contaminated_paper_runs_and_answers_live_last_30(tmp_path: Path) -> None:
    _make_live_execution_run(
        tmp_path,
        run_name="btc_shadow_preview_20260409",
        mode="paper",
        trade_rows=[
            {
                "trade_date": "2026-04-09",
                "recorded_at": "2026-04-09T15:00:00+00:00",
                "settled_at": "2026-04-09T15:11:00+00:00",
                "decision_id": "paper-dec-1",
                "ticker": "KXBTC15M-PAPER1",
                "side": "YES",
                "predicted_yes_probability": 0.68,
                "feature_basis_market_prob": 0.57,
                "yes_post_cost_edge": 0.09,
                "no_post_cost_edge": -0.19,
                "tau_minutes": 9.0,
                "reference_price_cents": 56,
                "quote_spread_cents": 2,
                "quote_age_seconds": 0.2,
                "regime_label": "neutral",
                "realized_pnl_dollars": 0.44,
                "cash_required_dollars": 0.56,
                "settlement_result": "YES",
            }
        ],
    )
    _make_live_execution_run(
        tmp_path,
        run_name="btc_live_20260415",
        mode="live",
        trade_rows=[
            {
                "trade_date": "2026-04-15",
                "recorded_at": "2026-04-15T16:00:00+00:00",
                "settled_at": "2026-04-15T16:10:00+00:00",
                "decision_id": "live-dec-1",
                "ticker": "KXBTC15M-LIVE1",
                "side": "YES",
                "predicted_yes_probability": 0.72,
                "feature_basis_market_prob": 0.61,
                "yes_post_cost_edge": 0.08,
                "no_post_cost_edge": -0.18,
                "tau_minutes": 8.0,
                "reference_price_cents": 58,
                "quote_spread_cents": 1,
                "quote_age_seconds": 0.1,
                "regime_label": "neutral",
                "realized_pnl_dollars": 0.40,
                "cash_required_dollars": 0.60,
                "settlement_result": "YES",
            },
            {
                "trade_date": "2026-04-15",
                "recorded_at": "2026-04-16T18:00:00+00:00",
                "settled_at": "2026-04-16T18:12:00+00:00",
                "decision_id": "live-dec-2",
                "ticker": "KXBTC15M-LIVE2",
                "side": "NO",
                "predicted_yes_probability": 0.45,
                "feature_basis_market_prob": 0.51,
                "yes_post_cost_edge": -0.15,
                "no_post_cost_edge": 0.05,
                "tau_minutes": 6.0,
                "reference_price_cents": 47,
                "quote_spread_cents": 2,
                "quote_age_seconds": 0.3,
                "regime_label": "downtrend",
                "realized_pnl_dollars": -0.20,
                "cash_required_dollars": 0.20,
                "settlement_result": "YES",
            },
        ],
    )

    outputs = generate_performance_scorecard(
        report_root=tmp_path / "no_reports_here",
        live_roots=(tmp_path / "output" / "live",),
        live_research_roots=(),
        output_dir=tmp_path / "scorecard",
        as_of_date=date(2026, 4, 17),
    )

    run_scorecard = pd.read_csv(outputs["run_scorecard_csv"])
    paper_row = run_scorecard.loc[run_scorecard["run_name"] == "btc_shadow_preview_20260409"].iloc[0]
    assert bool(paper_row["contamination_flag"]) is True
    assert FILL_REALISM_FIX_COMMIT in str(paper_row["contamination_reason"])

    live_30 = outputs["queries"]["live_last_30_calendar_days"]
    assert live_30["net_pnl_after_fees_dollars"] == pytest.approx(0.20)
    assert live_30["avg_realized_edge_cents"] == pytest.approx(10.0)
    assert live_30["avg_expected_edge_cents"] == pytest.approx(6.5)
    assert live_30["realized_vs_expected_edge_cents"] == pytest.approx(3.5)
    assert live_30["sharpe"] is not None

    rollups = pd.read_csv(outputs["window_rollups_csv"])
    assert {"model", "regime", "bucket", "time_of_day"}.issubset(set(rollups["segment_type"]))
    overall_live_30 = rollups.loc[
        (rollups["window_days"] == 30)
        & (rollups["mode_scope"] == "live")
        & (rollups["segment_type"] == "overall")
    ].iloc[0]
    assert overall_live_30["net_pnl_after_fees_dollars"] == pytest.approx(0.20)
