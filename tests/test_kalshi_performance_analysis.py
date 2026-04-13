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
            {
                "logged_at": "2026-04-01T12:02:00+00:00",
                "event_type": "signal_decision",
                "payload": {
                    "approved": True,
                    "decision_id": "dec-3",
                    "ticker": "KXBTC15M-TEST3",
                    "side": "NO",
                    "predicted_yes_probability": 0.34,
                    "predicted_no_probability": 0.66,
                    "feature_basis_market_prob": 0.48,
                    "raw_model_edge": -0.14,
                    "yes_post_cost_edge": -0.20,
                    "no_post_cost_edge": 0.08,
                    "tau_minutes": 5.0,
                    "reference_price_cents": 32,
                    "quote_spread_cents": 1,
                    "quote_age_seconds": 0.2,
                    "quote_mid_prob": 0.33,
                    "buy_yes_price_cents": 68,
                    "buy_no_price_cents": 32,
                    "yes_bid_cents": 67,
                    "yes_ask_cents": 68,
                    "regime_label": "downtrend",
                    "bearish_vote_count": 2,
                    "bullish_vote_count": 0,
                    "tranche_window": "5m",
                },
            },
            {
                "logged_at": "2026-04-01T12:03:00+00:00",
                "event_type": "signal_decision",
                "payload": {
                    "approved": True,
                    "decision_id": "dec-4",
                    "ticker": "KXBTC15M-TEST4",
                    "side": "NO",
                    "predicted_yes_probability": 0.29,
                    "predicted_no_probability": 0.71,
                    "feature_basis_market_prob": 0.44,
                    "raw_model_edge": -0.15,
                    "yes_post_cost_edge": -0.22,
                    "no_post_cost_edge": 0.10,
                    "tau_minutes": 3.0,
                    "reference_price_cents": 41,
                    "quote_spread_cents": 2,
                    "quote_age_seconds": 0.1,
                    "quote_mid_prob": 0.42,
                    "buy_yes_price_cents": 59,
                    "buy_no_price_cents": 41,
                    "yes_bid_cents": 58,
                    "yes_ask_cents": 59,
                    "regime_label": "downtrend",
                    "bearish_vote_count": 2,
                    "bullish_vote_count": 0,
                    "tranche_window": "3m",
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
            },
            {
                "logged_at": "2026-04-01T12:02:05+00:00",
                "event_type": "submit_requested",
                "payload": {
                    "decision_id": "dec-3",
                    "ticker": "KXBTC15M-TEST3",
                    "side": "NO",
                    "contracts": 1,
                    "limit_price_cents": 50,
                    "reference_price_cents": 32,
                    "subaccount": 0,
                    "target_id": "target-3",
                    "attempt_index": 1,
                    "desired_contracts": 1,
                    "remaining_contracts_before_submit": 1,
                    "hard_max_price_cents": 52,
                    "retry_reason": "target_created",
                    "was_first_attempt": True,
                },
            },
            {
                "logged_at": "2026-04-01T12:02:05.100000+00:00",
                "event_type": "submit_response",
                "payload": {
                    "decision_id": "dec-3",
                    "response": {
                        "order": {
                            "order_id": "order-3",
                            "client_order_id": "dec-3",
                            "ticker": "KXBTC15M-TEST3",
                            "side": "no",
                            "status": "canceled",
                            "no_price_dollars": "0.5000",
                            "yes_price_dollars": "0.5000",
                            "fill_count_fp": "0.00",
                            "remaining_count_fp": "0.00",
                            "initial_count_fp": "1.00",
                            "taker_fees_dollars": "0.000000",
                            "maker_fees_dollars": "0.000000",
                            "taker_fill_cost_dollars": "0.000000",
                            "maker_fill_cost_dollars": "0.000000",
                            "created_time": "2026-04-01T12:02:05.090000Z",
                            "last_update_time": "2026-04-01T12:02:05.090000Z",
                        }
                    },
                },
            },
            {
                "logged_at": "2026-04-01T12:03:05+00:00",
                "event_type": "submit_requested",
                "payload": {
                    "decision_id": "dec-4",
                    "ticker": "KXBTC15M-TEST4",
                    "side": "NO",
                    "contracts": 1,
                    "limit_price_cents": 50,
                    "reference_price_cents": 41,
                    "subaccount": 0,
                    "target_id": "target-4",
                    "attempt_index": 1,
                    "desired_contracts": 1,
                    "remaining_contracts_before_submit": 1,
                    "hard_max_price_cents": 50,
                    "retry_reason": "target_created",
                    "was_first_attempt": True,
                },
            },
            {
                "logged_at": "2026-04-01T12:03:05.100000+00:00",
                "event_type": "submit_error",
                "payload": {
                    "decision_id": "dec-4",
                    "status_code": 400,
                    "response": "{\"error\":{\"code\":\"invalid_subaccount_number\",\"message\":\"invalid subaccount number\",\"service\":\"exchange\"}}",
                },
            },
            {
                "logged_at": "2026-04-01T12:02:06+00:00",
                "event_type": "submit_requested",
                "payload": {
                    "decision_id": "dec-3-retry",
                    "ticker": "KXBTC15M-TEST3",
                    "side": "NO",
                    "contracts": 1,
                    "limit_price_cents": 52,
                    "reference_price_cents": 32,
                    "subaccount": 0,
                    "target_id": "target-3",
                    "attempt_index": 2,
                    "desired_contracts": 1,
                    "remaining_contracts_before_submit": 1,
                    "hard_max_price_cents": 52,
                    "retry_reason": "execution_cancelled",
                    "was_first_attempt": False,
                },
            },
            {
                "logged_at": "2026-04-01T12:02:06.100000+00:00",
                "event_type": "submit_response",
                "payload": {
                    "decision_id": "dec-3-retry",
                    "response": {
                        "order": {
                            "order_id": "order-3b",
                            "client_order_id": "dec-3-retry",
                            "ticker": "KXBTC15M-TEST3",
                            "side": "no",
                            "status": "executed",
                            "no_price_dollars": "0.5200",
                            "yes_price_dollars": "0.4800",
                            "fill_count_fp": "1.00",
                            "remaining_count_fp": "0.00",
                            "initial_count_fp": "1.00",
                            "taker_fees_dollars": "0.010000",
                            "maker_fees_dollars": "0.000000",
                            "taker_fill_cost_dollars": "0.520000",
                            "maker_fill_cost_dollars": "0.000000",
                            "created_time": "2026-04-01T12:02:06.090000Z",
                            "last_update_time": "2026-04-01T12:02:06.090000Z",
                        }
                    },
                },
            },
        ],
    )
    return run_dir


def _write_strategy_events_parquet(run_dir: Path, rows: list[dict]) -> None:
    path = (
        run_dir
        / "archive"
        / "strategy_events"
        / "environment=demo"
        / "date=2026-04-01"
        / "hour=12"
        / "part-test.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


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
    execution_df = pd.read_csv(artifacts["execution_rows.csv"])
    assert set(execution_df["decision_id"]) == {"dec-3", "dec-3-retry", "dec-4"}
    assert "cancelled_zero_fill" in set(execution_df["execution_outcome"])
    assert "submit_rejected" in set(execution_df["execution_outcome"])
    execution_quality_df = pd.read_csv(artifacts["execution_quality_summary.csv"])
    side_row = execution_quality_df.loc[
        (execution_quality_df["scope"] == "all_models")
        & (execution_quality_df["dimension"] == "side")
        & (execution_quality_df["bucket"] == "NO")
    ].iloc[0]
    assert int(side_row["attempted_count"]) == 3
    assert int(side_row["zero_fill_cancel_count"]) == 1
    assert int(side_row["attempted_contract_volume"]) == 3

    target_df = pd.read_csv(artifacts["target_execution_summary.csv"])
    target_row = target_df.loc[target_df["target_id"] == "target-3"].iloc[0]
    assert int(target_row["attempt_count"]) == 2
    assert bool(target_row["retry_rescued"]) is True
    assert int(target_row["filled_contracts"]) == 1

    model_totals_df = pd.read_csv(artifacts["model_totals.csv"])
    assert int(model_totals_df.iloc[0]["open_count"]) == 0
    assert int(model_totals_df.iloc[0]["closed_no_fill_count"]) == 2

    report_text = Path(artifacts["report.md"]).read_text(encoding="utf-8")
    assert "Model Totals" in report_text
    assert "Skip Reasons" in report_text
    assert "Quote Quality Effects" in report_text
    assert "Execution Quality" in report_text
    assert "Target Execution" in report_text
    assert "Open positions" in report_text
    assert "Closed no-fill rows" in report_text


def test_generate_kalshi_performance_reports_for_live_execution_low_cpu_mode(tmp_path: Path) -> None:
    live_run = _make_live_execution_run(tmp_path)
    outputs = generate_kalshi_performance_reports(
        live_run,
        output_dir=tmp_path / "reports_low_cpu",
        include_archive=False,
        include_row_exports=False,
    )
    artifacts = outputs["runs"][0]["artifacts"]

    assert "canonical_rows.csv" not in artifacts
    assert "execution_rows.csv" not in artifacts
    assert Path(artifacts["execution_quality_summary.csv"]).exists()

    metadata = json.loads(Path(artifacts["run_metadata.json"]).read_text(encoding="utf-8"))
    assert any("Strategy archive scan skipped" in note for note in metadata["notes"])


def test_generate_kalshi_performance_reports_lookback_prunes_old_live_event_paths(tmp_path: Path) -> None:
    live_run = _make_live_execution_run(tmp_path)
    _write_jsonl(
        live_run / "signal" / "lasso" / "demo" / "2026-03-01" / "events.jsonl",
        [
            {
                "logged_at": "2026-03-01T12:00:00+00:00",
                "event_type": "signal_decision",
                "payload": {
                    "approved": True,
                    "decision_id": "old-dec-1",
                    "ticker": "KXBTC15M-OLD1",
                    "side": "YES",
                    "predicted_yes_probability": 0.75,
                    "feature_basis_market_prob": 0.55,
                    "reference_price_cents": 54,
                },
            }
        ],
    )
    _write_jsonl(
        live_run / "execution" / "lasso" / "demo" / "2026-03-01" / "events.jsonl",
        [
            {
                "logged_at": "2026-03-01T12:10:00+00:00",
                "event_type": "simulated_position_settled",
                "payload": {
                    "decision_id": "old-dec-1",
                    "ticker": "KXBTC15M-OLD1",
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

    outputs = generate_kalshi_performance_reports(
        live_run,
        output_dir=tmp_path / "reports_lookback",
        lookback_days=1,
        include_archive=False,
    )
    artifacts = outputs["runs"][0]["artifacts"]
    canonical_df = pd.read_csv(artifacts["canonical_rows.csv"])
    assert "old-dec-1" not in set(canonical_df["decision_id"])


def test_generate_kalshi_performance_reports_for_live_execution_archive_fallback(tmp_path: Path) -> None:
    live_run = tmp_path / "output" / "live" / "archive_only_live_run"
    (live_run / "signal" / "bagged_lasso" / "demo" / "2026-04-01").mkdir(parents=True, exist_ok=True)
    (live_run / "execution" / "bagged_lasso" / "demo" / "2026-04-01").mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        live_run / "execution" / "bagged_lasso" / "demo" / "2026-04-01" / "events.jsonl",
        [
            {
                "logged_at": "2026-04-01T12:05:00+00:00",
                "event_type": "submit_requested",
                "payload": {
                    "decision_id": "dec-archive-1",
                    "ticker": "KXBTC15M-ARCHIVE1",
                    "side": "NO",
                    "contracts": 1,
                    "limit_price_cents": 50,
                    "reference_price_cents": 32,
                    "subaccount": 0,
                },
            }
        ],
    )
    _write_strategy_events_parquet(
        live_run,
        [
            {
                "run_name": "archive_only_live_run",
                "environment": "demo",
                "model_label": "bagged_lasso",
                "model_family": "bagged_lasso",
                "strategy_event_id": "strategy:bagged_lasso:signal:dec-archive-1",
                "event_kind": "signal_approved",
                "event_time": "2026-04-01T12:05:00+00:00",
                "ticker": "KXBTC15M-ARCHIVE1",
                "side": "NO",
                "decision_id": "dec-archive-1",
                "approved": True,
                "status": "approved",
                "reason": None,
                "reference_price_cents": 32,
                "limit_price_cents": 50,
                "predicted_yes_probability": 0.34,
                "predicted_no_probability": 0.66,
                "feature_basis_market_prob": 0.48,
                "raw_model_edge": -0.14,
                "post_cost_edge": 0.08,
                "yes_post_cost_edge": -0.20,
                "no_post_cost_edge": 0.08,
                "tau_minutes": 5.0,
                "last_yes_price_cents": 68,
                "yes_bid_cents": 67,
                "yes_ask_cents": 68,
                "buy_yes_price_cents": 68,
                "buy_no_price_cents": 32,
                "quote_mid_prob": 0.33,
                "quote_spread_cents": 1,
                "quote_age_seconds": 0.2,
                "regime_label": "downtrend",
                "bearish_vote_count": 2,
                "bullish_vote_count": 0,
                "regime_price_momentum_bearish": False,
                "regime_signed_flow_bearish": True,
                "regime_yes_share_bearish": True,
                "regime_price_momentum_bullish": False,
                "regime_signed_flow_bullish": False,
                "regime_yes_share_bullish": False,
                "tau_bucket": "4-6",
                "price_bucket": "30-40",
                "probability_bucket": "60-70",
                "edge_bucket": "5-10",
                "bucket_policy_dimension": None,
                "bucket_policy_bucket": None,
                "bucket_policy_side": None,
                "contracts": 1,
                "estimated_entry_cost_dollars": 0.32,
                "estimated_fees_dollars": 0.02,
                "estimated_cash_required_dollars": 0.34,
                "thesis_id": "thesis-1",
                "tranche_index": 0,
                "tranche_window": "10m",
                "tranche_reason": "opened",
                "lifecycle_state": "opened",
                "total_thesis_budget_dollars": 1.0,
                "payout_if_yes_dollars": -0.32,
                "payout_if_no_dollars": 0.68,
                "expected_value_dollars": 0.08,
                "worst_case_loss_dollars": 0.34,
                "execution_mode": "live",
                "subaccount": 0,
            },
            {
                "run_name": "archive_only_live_run",
                "environment": "demo",
                "model_label": "bagged_lasso",
                "model_family": "bagged_lasso",
                "strategy_event_id": "strategy:bagged_lasso:execution:dec-archive-1:cancelled",
                "event_kind": "execution_cancelled",
                "event_time": "2026-04-01T12:05:01+00:00",
                "ticker": "KXBTC15M-ARCHIVE1",
                "side": "NO",
                "decision_id": "dec-archive-1",
                "client_order_id": "dec-archive-1",
                "order_id": "order-archive-1",
                "approved": None,
                "status": "cancelled",
                "reason": "order_cancelled",
                "reference_price_cents": 32,
                "limit_price_cents": 50,
                "contracts": 1,
                "filled_contracts": 0,
                "remaining_contracts": 0,
                "fill_price_cents": None,
                "entry_cost_dollars": 0.0,
                "fees_dollars": 0.0,
                "cash_required_dollars": 0.0,
                "available_cash_dollars": 10.0,
                "realized_pnl_dollars": None,
                "cumulative_realized_pnl_dollars": None,
                "settlement_result": None,
                "thesis_id": "thesis-1",
                "tranche_index": 0,
                "tranche_window": "10m",
                "tranche_reason": "opened",
                "lifecycle_state": "opened",
                "total_thesis_budget_dollars": 1.0,
                "payout_if_yes_dollars": -0.32,
                "payout_if_no_dollars": 0.68,
                "expected_value_dollars": 0.08,
                "worst_case_loss_dollars": 0.34,
                "execution_mode": "live",
                "subaccount": 0,
            },
        ],
    )

    outputs = generate_kalshi_performance_reports(
        live_run,
        output_dir=tmp_path / "reports_archive_fallback",
        environment="demo",
    )
    artifacts = outputs["runs"][0]["artifacts"]
    execution_df = pd.read_csv(artifacts["execution_rows.csv"])
    assert len(execution_df) == 1
    row = execution_df.loc[execution_df["decision_id"] == "dec-archive-1"].iloc[0]
    assert row["price_bucket"] == "30-40"
    assert row["tranche_window"] == "10m"
    assert row["execution_outcome"] == "cancelled_zero_fill"
    model_totals_df = pd.read_csv(artifacts["model_totals.csv"])
    assert int(model_totals_df.iloc[0]["recorded_count"]) == 1


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
