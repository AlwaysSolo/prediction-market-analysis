from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.live.kalshi.performance_analysis import (
    detect_analysis_targets,
    generate_kalshi_performance_reports,
)


def _write_jsonl(path: Path, rows: list[dict], *, malformed_tail: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row) for row in rows)
    if malformed_tail is not None:
        content = f"{content}\n{malformed_tail}"
    path.write_text(content + "\n", encoding="utf-8")


def _make_live_execution_run(base: Path) -> Path:
    run_dir = base / "output" / "live" / "my_live_run"
    signal_path = run_dir / "signal" / "lasso" / "demo" / "2026-04-01" / "events.jsonl"
    execution_path = run_dir / "execution" / "lasso" / "demo" / "2026-04-01" / "events.jsonl"
    _write_jsonl(
        signal_path,
        [
            {
                "logged_at": "2026-04-01T12:00:00+00:00",
                "event_type": "signal_decision",
                "payload": {
                    "approved": True,
                    "decision_id": "dec-1",
                    "ticker": "KXBTC15M-TEST1",
                    "side": "YES",
                    "predicted_yes_probability": 0.76,
                    "predicted_no_probability": 0.24,
                    "feature_basis_market_prob": 0.56,
                    "raw_model_edge": 0.20,
                    "yes_post_cost_edge": 0.11,
                    "no_post_cost_edge": -0.21,
                    "tau_minutes": 9.0,
                    "reference_price_cents": 54,
                    "quote_spread_cents": 2,
                    "quote_age_seconds": 0.4,
                    "quote_mid_prob": 0.55,
                    "buy_yes_price_cents": 54,
                    "buy_no_price_cents": 46,
                    "yes_bid_cents": 54,
                    "yes_ask_cents": 56,
                    "regime_label": "neutral",
                    "bearish_vote_count": 1,
                    "bullish_vote_count": 1,
                },
            },
            {
                "logged_at": "2026-04-01T12:01:00+00:00",
                "event_type": "signal_decision",
                "payload": {
                    "approved": False,
                    "decision_id": "dec-2",
                    "ticker": "KXBTC15M-TEST2",
                    "block_reason": "blocked_by_bucket_policy",
                    "bucket_policy_side": "YES",
                    "bucket_policy_dimension": "price",
                    "bucket_policy_bucket": "30-40",
                },
            },
        ],
        malformed_tail='{"logged_at":"broken"',
    )
    _write_jsonl(
        execution_path,
        [
            {
                "logged_at": "2026-04-01T12:10:00+00:00",
                "event_type": "simulated_position_settled",
                "payload": {
                    "decision_id": "dec-1",
                    "ticker": "KXBTC15M-TEST1",
                    "side": "YES",
                    "settlement_result": "YES",
                    "contracts": 1,
                    "cash_required_dollars": 0.56,
                    "realized_pnl_dollars": 0.44,
                    "cumulative_realized_pnl_dollars": 0.44,
                },
            }
        ],
    )
    return run_dir


def _make_live_research_run(base: Path) -> Path:
    run_dir = base / "output" / "live_research" / "my_research_run"
    research_path = run_dir / "research" / "lasso" / "demo" / "2026-04-01" / "events.jsonl"
    _write_jsonl(
        research_path,
        [
            {
                "logged_at": "2026-04-01T12:00:00+00:00",
                "event_type": "research_sample_recorded",
                "payload": {
                    "sample_id": "sample-1",
                    "ticker": "KXBTC15M-TEST1",
                    "side": "YES",
                    "reference_price_cents": 52,
                    "predicted_yes_probability": 0.73,
                    "predicted_no_probability": 0.27,
                    "chosen_side_probability": 0.73,
                    "feature_basis_market_prob": 0.54,
                    "raw_model_edge": 0.19,
                    "yes_post_cost_edge": 0.09,
                    "no_post_cost_edge": -0.19,
                    "chosen_post_cost_edge": 0.09,
                    "tau_minutes": 8.0,
                    "tau_bucket": "8-10",
                    "price_bucket": "50-60",
                    "chosen_side_probability_bucket": "70-80",
                    "chosen_side_edge_bucket": "5-10",
                    "estimated_cash_required_dollars": 0.54,
                    "quote_spread_cents": 2,
                    "quote_age_seconds": 0.2,
                    "regime_label": "downtrend",
                    "bearish_vote_count": 2,
                    "bullish_vote_count": 0,
                },
            },
            {
                "logged_at": "2026-04-01T12:10:00+00:00",
                "event_type": "research_sample_settled",
                "payload": {
                    "sample_id": "sample-1",
                    "ticker": "KXBTC15M-TEST1",
                    "side": "YES",
                    "settlement_result": "YES",
                    "is_win": True,
                    "cash_required_dollars": 0.54,
                    "realized_pnl_dollars": 0.46,
                    "cumulative_realized_pnl_dollars": 0.46,
                },
            },
            {
                "logged_at": "2026-04-01T12:01:00+00:00",
                "event_type": "research_sample_skipped",
                "payload": {
                    "ticker": "KXBTC15M-TEST2",
                    "reason": "missing_quote",
                },
            },
        ],
    )
    return run_dir


def _make_offline_artifact_run(base: Path) -> Path:
    run_dir = base / "artifacts" / "kalshi" / "kxbtc15m_lasso" / "kxbtc15m_lasso_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "series": "KXBTC15M",
                "model_family": "lasso",
                "model_type": "l1_logistic_regression",
                "feature_count": 3,
                "train_rows": 10,
                "validation_rows": 5,
                "test_rows": 6,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "feature_manifest.json").write_text(json.dumps({"feature_order": ["z_implied", "tau_minutes"]}), encoding="utf-8")
    (run_dir / "validation_metrics.json").write_text(
        json.dumps({"trades": 2, "net_pnl_dollars": 1.0, "return_pct": 0.01, "max_drawdown_dollars": 0.5, "max_drawdown_pct": 0.005, "log_loss": 0.52}),
        encoding="utf-8",
    )
    (run_dir / "test_metrics.json").write_text(
        json.dumps({"trades": 3, "net_pnl_dollars": 1.5, "return_pct": 0.015, "max_drawdown_dollars": 0.6, "max_drawdown_pct": 0.006, "log_loss": 0.51}),
        encoding="utf-8",
    )
    (run_dir / "validation_diagnostics.json").write_text(json.dumps({"side_breakdown": []}), encoding="utf-8")
    (run_dir / "test_diagnostics.json").write_text(
        json.dumps(
            {
                "side_breakdown": [{"side": "YES", "trades": 2, "win_rate": 0.5}, {"side": "NO", "trades": 1, "win_rate": 1.0}],
                "calibration": {"calibrated_probability_buckets": [{"bucket": "70-80", "count": 2, "avg_predicted_probability": 0.74, "actual_yes_rate": 0.75}]},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"ticker": "KXBTC15M-1", "actual_outcome": "YES", "raw_probability": 0.78, "calibrated_probability": 0.76},
            {"ticker": "KXBTC15M-2", "actual_outcome": "NO", "raw_probability": 0.28, "calibrated_probability": 0.30},
            {"ticker": "KXBTC15M-3", "actual_outcome": "YES", "raw_probability": 0.66, "calibrated_probability": 0.68},
        ]
    ).to_parquet(run_dir / "test_predictions.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ticker": "KXBTC15M-1",
                "trade_id": "trade-1",
                "created_time": "2026-04-01T12:00:00+00:00",
                "close_time": "2026-04-01T12:10:00+00:00",
                "side": "YES",
                "actual_outcome": "YES",
                "is_win": True,
                "market_prob": 0.55,
                "predicted_yes_probability": 0.76,
                "tau_minutes": 8.0,
                "reference_price_cents": 54,
                "post_cost_edge": 0.09,
                "cash_required_dollars": 0.56,
                "net_pnl_dollars": 0.44,
                "price_bucket": "50-60",
                "tau_bucket": "8-10",
                "regime_label": "neutral",
                "bearish_vote_count": 1,
                "bullish_vote_count": 1,
            },
            {
                "ticker": "KXBTC15M-2",
                "trade_id": "trade-2",
                "created_time": "2026-04-01T12:01:00+00:00",
                "close_time": "2026-04-01T12:11:00+00:00",
                "side": "NO",
                "actual_outcome": "NO",
                "is_win": True,
                "market_prob": 0.54,
                "predicted_yes_probability": 0.31,
                "tau_minutes": 10.0,
                "reference_price_cents": 46,
                "post_cost_edge": 0.08,
                "cash_required_dollars": 0.48,
                "net_pnl_dollars": 0.52,
                "price_bucket": "40-50",
                "tau_bucket": "10-12",
                "regime_label": "downtrend",
                "bearish_vote_count": 2,
                "bullish_vote_count": 0,
            },
        ]
    ).to_parquet(run_dir / "test_trade_records.parquet", index=False)
    return run_dir


def test_detect_analysis_targets_for_run_dirs_and_files(tmp_path: Path) -> None:
    live_run = _make_live_execution_run(tmp_path)
    research_run = _make_live_research_run(tmp_path)
    offline_run = _make_offline_artifact_run(tmp_path)

    live_detection = detect_analysis_targets(live_run)
    assert live_detection.detected_mode == "live_execution"
    assert live_detection.targets[0].run_dir == live_run

    live_file_detection = detect_analysis_targets(live_run / "signal" / "lasso" / "demo" / "2026-04-01" / "events.jsonl")
    assert live_file_detection.targets[0].run_dir == live_run

    research_detection = detect_analysis_targets(research_run)
    assert research_detection.detected_mode == "live_research"
    assert research_detection.targets[0].run_dir == research_run

    offline_file_detection = detect_analysis_targets(offline_run / "summary.json")
    assert offline_file_detection.detected_mode == "offline_artifacts"
    assert offline_file_detection.targets[0].run_dir == offline_run


def test_detect_analysis_targets_for_offline_root(tmp_path: Path) -> None:
    offline_run = _make_offline_artifact_run(tmp_path)
    root = offline_run.parents[1]
    detection = detect_analysis_targets(root)
    assert detection.detected_mode == "offline_artifacts"
    assert detection.targets[0].run_dir == offline_run


def test_generate_kalshi_performance_reports_for_live_execution(tmp_path: Path) -> None:
    live_run = _make_live_execution_run(tmp_path)
    outputs = generate_kalshi_performance_reports(live_run, output_dir=tmp_path / "reports")
    artifacts = outputs["runs"][0]["artifacts"]

    metadata = json.loads(Path(artifacts["run_metadata.json"]).read_text(encoding="utf-8"))
    assert metadata["detected_source_type"] == "live_execution"
    assert any(metadata["skipped_malformed_lines_by_file"].values())
    skip_df = pd.read_csv(artifacts["skip_reason_summary.csv"])
    blocked_row = skip_df.loc[skip_df["reason"] == "blocked_by_bucket_policy"].iloc[0]
    assert blocked_row["bucket_policy_side"] == "YES"
    assert blocked_row["bucket_policy_dimension"] == "price"
    assert blocked_row["bucket_policy_bucket"] == "30-40"

    report_text = Path(artifacts["report.md"]).read_text(encoding="utf-8")
    assert "Model Totals" in report_text
    assert "Skip Reasons" in report_text
    assert "Quote Quality Effects" in report_text


def test_generate_kalshi_performance_reports_for_live_research(tmp_path: Path) -> None:
    research_run = _make_live_research_run(tmp_path)
    outputs = generate_kalshi_performance_reports(research_run, output_dir=tmp_path / "reports")
    artifacts = outputs["runs"][0]["artifacts"]

    assert Path(artifacts["skip_reason_summary.csv"]).exists()
    assert Path(artifacts["regime_bucket_summary.csv"]).exists()
    report_text = Path(artifacts["report.md"]).read_text(encoding="utf-8")
    assert "Highlights" in report_text
    assert "Regime Summary" in report_text
    assert "Skip Reasons" in report_text


def test_generate_kalshi_performance_reports_for_offline_artifacts(tmp_path: Path) -> None:
    offline_run = _make_offline_artifact_run(tmp_path)
    outputs = generate_kalshi_performance_reports(offline_run, output_dir=tmp_path / "reports")
    artifacts = outputs["runs"][0]["artifacts"]

    assert Path(artifacts["prediction_metrics.csv"]).exists()
    assert Path(artifacts["trade_metrics.csv"]).exists()
    assert Path(artifacts["calibration_summary.csv"]).exists()

    report_text = Path(artifacts["report.md"]).read_text(encoding="utf-8")
    assert "Prediction Metrics" in report_text
    assert "Trade Metrics" in report_text
