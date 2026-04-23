from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.live.kalshi.parity_analysis import PARITY_SCHEMA_VERSION, build_parity_artifacts


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _live_signal_event(
    *,
    logged_at: datetime,
    decision_id: str,
    ticker: str,
    side: str,
    feature_row_id: str | None = None,
    market_event_id: str | None = None,
    contracts: int = 1,
    predicted_yes_probability: float = 0.76,
    reference_price_cents: int = 56,
    buy_yes_price_cents: int = 56,
    buy_no_price_cents: int = 44,
) -> dict:
    return {
        "logged_at": _iso(logged_at),
        "event_type": "signal_decision",
        "payload": {
            "approved": True,
            "decision_id": decision_id,
            "ticker": ticker,
            "side": side,
            "feature_row_id": feature_row_id,
            "market_event_id": market_event_id,
            "predicted_yes_probability": predicted_yes_probability,
            "predicted_no_probability": 1.0 - predicted_yes_probability,
            "feature_basis_market_prob": 0.55,
            "raw_model_edge": 0.21,
            "yes_post_cost_edge": 0.18,
            "no_post_cost_edge": -0.28,
            "tau_minutes": 7.0,
            "reference_price_cents": reference_price_cents,
            "quote_spread_cents": 2,
            "quote_age_seconds": 0.2,
            "buy_yes_price_cents": buy_yes_price_cents,
            "buy_no_price_cents": buy_no_price_cents,
            "yes_bid_cents": 54,
            "yes_ask_cents": 56,
            "regime_label": "uptrend",
            "contracts": contracts,
        },
    }


def _live_settlement_event(
    *,
    logged_at: datetime,
    decision_id: str,
    ticker: str,
    side: str,
    settlement_result: str,
    realized_pnl_dollars: float,
    cash_required_dollars: float,
    contracts: int = 1,
    filled_contracts: int | None = None,
) -> dict:
    return {
        "logged_at": _iso(logged_at),
        "event_type": "simulated_position_settled",
        "payload": {
            "decision_id": decision_id,
            "ticker": ticker,
            "side": side,
            "settlement_result": settlement_result,
            "contracts": contracts,
            "filled_contracts": contracts if filled_contracts is None else filled_contracts,
            "cash_required_dollars": cash_required_dollars,
            "realized_pnl_dollars": realized_pnl_dollars,
            "cumulative_realized_pnl_dollars": realized_pnl_dollars,
            "fill_price_cents": 56 if side == "YES" else 44,
        },
    }


def _research_started_event(logged_at: datetime) -> dict:
    return {
        "logged_at": _iso(logged_at),
        "event_type": "research_started",
        "payload": {
            "min_edge_cents": 2.0,
            "min_tau_minutes": 2.0,
            "max_tau_minutes": 14.0,
            "apply_regime_hard_gate": False,
            "price_band_min_cents": 20,
            "price_band_max_cents": 80,
            "quote_max_age_seconds": 3.0,
            "contracts_per_sample": 1,
            "slippage_pct": 1.0,
            "enable_bucket_ban_policy": False,
            "banned_yes_tau_buckets": [],
            "banned_yes_price_buckets": [],
            "banned_yes_probability_buckets": [],
            "banned_no_price_buckets": [],
        },
    }


def _research_recorded_event(
    *,
    logged_at: datetime,
    sample_id: str,
    ticker: str,
    side: str,
    feature_row_id: str,
    market_event_id: str,
    contracts: int = 1,
    predicted_yes_probability: float = 0.76,
    reference_price_cents: int = 56,
) -> dict:
    return {
        "logged_at": _iso(logged_at),
        "event_type": "research_sample_recorded",
        "payload": {
            "sample_id": sample_id,
            "ticker": ticker,
            "side": side,
            "contracts": contracts,
            "reference_price_cents": reference_price_cents,
            "max_acceptable_entry_price_cents": 70,
            "predicted_yes_probability": predicted_yes_probability,
            "predicted_no_probability": 1.0 - predicted_yes_probability,
            "chosen_side_probability": predicted_yes_probability if side == "YES" else 1.0 - predicted_yes_probability,
            "feature_basis_market_prob": 0.55,
            "raw_model_edge": 0.21,
            "yes_post_cost_edge": 0.18,
            "no_post_cost_edge": -0.28,
            "chosen_post_cost_edge": 0.18,
            "last_yes_price_cents": 55,
            "last_price_cents": 55,
            "yes_bid_cents": 54,
            "yes_ask_cents": 56,
            "no_bid_cents": 44,
            "no_ask_cents": 46,
            "buy_yes_price_cents": 56,
            "buy_no_price_cents": 44,
            "quote_mid_prob": 0.55,
            "quote_spread_cents": 2,
            "quote_age_seconds": 0.2,
            "regime_label": "uptrend",
            "bearish_vote_count": 0,
            "bullish_vote_count": 3,
            "regime_price_momentum_bearish": False,
            "regime_signed_flow_bearish": False,
            "regime_yes_share_bearish": False,
            "regime_price_momentum_bullish": True,
            "regime_signed_flow_bullish": True,
            "regime_yes_share_bullish": True,
            "tau_minutes": 7.0,
            "tau_bucket": "6-8",
            "price_bucket": "50-60",
            "chosen_side_probability_bucket": "70-80",
            "chosen_side_edge_bucket": "10-20",
            "estimated_entry_cost_dollars": 0.57,
            "estimated_fees_dollars": 0.02,
            "estimated_cash_required_dollars": 0.59,
            "market_event_id": market_event_id,
            "feature_row_id": feature_row_id,
            "raw_event_id": "raw-1",
            "received_at": _iso(logged_at + timedelta(milliseconds=200)),
            "source": "ticker",
        },
    }


def _research_skipped_event(
    *,
    logged_at: datetime,
    ticker: str,
    feature_row_id: str,
    market_event_id: str,
    reason: str = "blocked_by_bucket_policy",
    predicted_yes_probability: float = 0.76,
) -> dict:
    return {
        "logged_at": _iso(logged_at),
        "event_type": "research_sample_skipped",
        "payload": {
            "sample_id": None,
            "ticker": ticker,
            "reason": reason,
            "side": "YES",
            "predicted_yes_probability": predicted_yes_probability,
            "predicted_no_probability": 1.0 - predicted_yes_probability,
            "feature_basis_market_prob": 0.55,
            "raw_model_edge": 0.21,
            "yes_post_cost_edge": 0.18,
            "no_post_cost_edge": -0.28,
            "chosen_post_cost_edge": 0.18,
            "tau_minutes": 7.0,
            "reference_price_cents": 56,
            "yes_bid_cents": 54,
            "yes_ask_cents": 56,
            "no_bid_cents": 44,
            "no_ask_cents": 46,
            "buy_yes_price_cents": 56,
            "buy_no_price_cents": 44,
            "quote_mid_prob": 0.55,
            "quote_spread_cents": 2,
            "quote_age_seconds": 0.2,
            "tau_bucket": "6-8",
            "price_bucket": "50-60",
            "chosen_side_probability_bucket": "70-80",
            "chosen_side_edge_bucket": "10-20",
            "regime_label": "uptrend",
            "market_event_id": market_event_id,
            "feature_row_id": feature_row_id,
            "received_at": _iso(logged_at + timedelta(milliseconds=200)),
            "source": "ticker",
        },
    }


def _research_settled_event(
    *,
    logged_at: datetime,
    sample_id: str,
    ticker: str,
    settlement_result: str,
    realized_pnl_dollars: float,
    side: str = "YES",
) -> dict:
    return {
        "logged_at": _iso(logged_at),
        "event_type": "research_sample_settled",
        "payload": {
            "sample_id": sample_id,
            "ticker": ticker,
            "side": side,
            "settlement_result": settlement_result,
            "is_win": realized_pnl_dollars > 0.0,
            "realized_pnl_dollars": realized_pnl_dollars,
            "cumulative_realized_pnl_dollars": realized_pnl_dollars,
            "cash_required_dollars": 0.59,
        },
    }


def _write_live_run(
    base: Path,
    *,
    run_name: str,
    model: str,
    trade_rows: list[dict],
    signal_rows: list[dict],
    model_outputs: list[dict],
    feature_rows: list[dict],
    mode: str = "live",
    extra_execution_rows: list[dict] | None = None,
) -> Path:
    run_dir = base / "output" / "live" / run_name
    execution_started = {
        "logged_at": _iso(datetime(2026, 4, 1, 11, 59, tzinfo=UTC)),
        "event_type": "execution_started",
        "payload": {"mode": mode},
    }
    _write_jsonl(
        run_dir / "signal" / model / "production" / "date=2026-04-01" / "events.jsonl",
        signal_rows,
    )
    _write_jsonl(
        run_dir / "execution" / model / "production" / "date=2026-04-01" / "events.jsonl",
        [execution_started, *(extra_execution_rows or []), *trade_rows],
    )
    _write_parquet(
        run_dir / "archive" / "model_outputs" / "date=2026-04-01" / "environment=production" / "part-0.parquet",
        model_outputs,
    )
    _write_parquet(
        run_dir / "archive" / "feature_rows" / "date=2026-04-01" / "environment=production" / "part-0.parquet",
        feature_rows,
    )
    return run_dir


def _write_research_run(
    base: Path,
    *,
    run_name: str,
    model: str,
    raw_rows: list[dict],
    model_outputs: list[dict],
    feature_rows: list[dict],
) -> Path:
    run_dir = base / "output" / "live_research" / run_name
    _write_jsonl(
        run_dir / "research" / model / "production" / "date=2026-04-01" / "events.jsonl",
        raw_rows,
    )
    _write_parquet(
        run_dir / "archive" / "model_outputs" / "date=2026-04-01" / "environment=production" / "part-0.parquet",
        model_outputs,
    )
    _write_parquet(
        run_dir / "archive" / "feature_rows" / "date=2026-04-01" / "environment=production" / "part-0.parquet",
        feature_rows,
    )
    return run_dir


def _base_archive_rows(
    *,
    run_name: str,
    model: str,
    event_time: datetime,
    ticker: str,
    feature_row_id: str,
    market_event_id: str,
    model_file: str,
    schema_version: str = "kalshi_feature_row_v1",
) -> tuple[list[dict], list[dict]]:
    model_output = {
        "run_name": run_name,
        "environment": "production",
        "model_label": model,
        "model_family": model,
        "model_output_id": f"model_output:{model}:{feature_row_id}",
        "feature_row_id": feature_row_id,
        "market_event_id": market_event_id,
        "ticker": ticker,
        "event_time": _iso(event_time),
        "received_at": _iso(event_time + timedelta(milliseconds=200)),
        "model_file": model_file,
        "predicted_yes_probability": 0.76,
        "market_prob": 0.55,
        "tau_minutes": 7.0,
        "model_edge": 0.21,
        "yes_bid_cents": 54,
        "yes_ask_cents": 56,
        "no_bid_cents": 44,
        "no_ask_cents": 46,
        "buy_yes_price_cents": 56,
        "buy_no_price_cents": 44,
        "quote_mid_prob": 0.55,
        "quote_spread_cents": 2,
        "quote_age_seconds": 0.2,
        "regime_label": "uptrend",
    }
    feature_row = {
        "run_name": run_name,
        "environment": "production",
        "feature_row_id": feature_row_id,
        "market_event_id": market_event_id,
        "ticker": ticker,
        "event_time": _iso(event_time),
        "schema_version": schema_version,
    }
    return [model_output], [feature_row]


def test_parity_report_builds_recorded_replay_and_segmented_outputs(tmp_path: Path) -> None:
    base_time = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    model = "bagged_lasso"
    live_run_name = "btc_live_run"
    feature_ids = [f"feature:event-{index}" for index in range(1, 7)]
    market_ids = [f"event-{index}" for index in range(1, 7)]
    tickers = [f"KXBTC15M-T{index}" for index in range(1, 7)]
    live_signal_rows: list[dict] = []
    live_trade_rows: list[dict] = []
    extra_execution_rows: list[dict] = []
    live_model_outputs: list[dict] = []
    live_feature_rows: list[dict] = []
    research_raw_rows: list[dict] = [_research_started_event(base_time - timedelta(minutes=5))]
    research_model_outputs: list[dict] = []
    research_feature_rows: list[dict] = []

    for index, (feature_row_id, market_event_id, ticker) in enumerate(zip(feature_ids, market_ids, tickers), start=1):
        decision_time = base_time + timedelta(days=index - 1)
        settlement_time = decision_time + timedelta(minutes=10)
        signal = _live_signal_event(
            logged_at=decision_time,
            decision_id=f"decision-{index}",
            ticker=ticker,
            side="YES",
            feature_row_id=None if index != 5 else feature_row_id,
            market_event_id=market_event_id if index == 4 else None,
            contracts=2 if index == 3 else 1,
        )
        settlement = _live_settlement_event(
            logged_at=settlement_time,
            decision_id=f"decision-{index}",
            ticker=ticker,
            side="YES",
            settlement_result="YES" if index % 2 == 1 else "NO",
            realized_pnl_dollars=0.40 if index == 6 else (0.42 if index % 2 == 1 else -0.58),
            cash_required_dollars=0.58 if index != 6 else 0.29,
            contracts=1,
            filled_contracts=1 if index != 6 else 1,
        )
        live_signal_rows.append(signal)
        live_trade_rows.append(settlement)
        if index == 3:
            extra_execution_rows.extend(
                [
                    {
                        "logged_at": _iso(decision_time + timedelta(seconds=1)),
                        "event_type": "submit_requested",
                        "payload": {
                            "decision_id": "decision-3",
                            "ticker": ticker,
                            "side": "YES",
                            "contracts": 2,
                            "limit_price_cents": 56,
                        },
                    },
                    {
                        "logged_at": _iso(decision_time + timedelta(seconds=2)),
                        "event_type": "submit_response",
                        "payload": {
                            "decision_id": "decision-3",
                            "response": {
                                "order": {
                                    "order_id": "order-3",
                                    "client_order_id": "decision-3",
                                    "ticker": ticker,
                                    "side": "yes",
                                    "status": "canceled",
                                    "yes_price_dollars": "0.5600",
                                    "no_price_dollars": "0.4400",
                                    "fill_count_fp": "1.00",
                                    "remaining_count_fp": "1.00",
                                    "initial_count_fp": "2.00",
                                    "created_time": _iso(decision_time + timedelta(seconds=1)),
                                    "last_update_time": _iso(decision_time + timedelta(seconds=2)),
                                }
                            },
                        },
                    },
                ]
            )

        model_file = "artifacts/kalshi/live_model.joblib" if index != 6 else "artifacts/kalshi/live_model_v2.joblib"
        schema_version = "kalshi_feature_row_v1" if index != 6 else "kalshi_feature_row_v2"
        model_output_rows, feature_rows = _base_archive_rows(
            run_name=live_run_name,
            model=model,
            event_time=decision_time,
            ticker=ticker,
            feature_row_id=feature_row_id,
            market_event_id=market_event_id,
            model_file=model_file,
            schema_version=schema_version,
        )
        if index == 5:
            model_output_rows[0]["event_time"] = _iso(decision_time - timedelta(milliseconds=500))
        live_model_outputs.extend(model_output_rows)
        live_feature_rows.extend(feature_rows)

        if index in {1, 2, 3, 4, 6}:
            research_model_file = "artifacts/kalshi/live_model.joblib" if index != 6 else "artifacts/kalshi/other_model.joblib"
            research_schema_version = "kalshi_feature_row_v1" if index != 6 else "kalshi_feature_row_v0"
            model_rows, feature_rows = _base_archive_rows(
                run_name="research_run",
                model=model,
                event_time=decision_time,
                ticker=ticker,
                feature_row_id=feature_row_id,
                market_event_id=market_event_id,
                model_file=research_model_file,
                schema_version=research_schema_version,
            )
            research_model_outputs.extend(model_rows)
            research_feature_rows.extend(feature_rows)

        if index == 1:
            research_raw_rows.append(
                _research_recorded_event(
                    logged_at=decision_time,
                    sample_id="sample-1",
                    ticker=ticker,
                    side="YES",
                    feature_row_id=feature_row_id,
                    market_event_id=market_event_id,
                )
            )
            research_raw_rows.append(
                _research_settled_event(
                    logged_at=settlement_time,
                    sample_id="sample-1",
                    ticker=ticker,
                    settlement_result="YES",
                    realized_pnl_dollars=0.41,
                )
            )
        elif index == 2:
            research_raw_rows.append(
                _research_skipped_event(
                    logged_at=decision_time,
                    ticker=ticker,
                    feature_row_id=feature_row_id,
                    market_event_id=market_event_id,
                )
            )
        elif index == 3:
            research_raw_rows.append(
                _research_recorded_event(
                    logged_at=decision_time,
                    sample_id="sample-3",
                    ticker=ticker,
                    side="YES",
                    feature_row_id=feature_row_id,
                    market_event_id=market_event_id,
                    contracts=1,
                )
            )
            research_raw_rows.append(
                _research_settled_event(
                    logged_at=settlement_time,
                    sample_id="sample-3",
                    ticker=ticker,
                    settlement_result="YES",
                    realized_pnl_dollars=0.41,
                )
            )
        elif index == 4:
            research_raw_rows.append(
                _research_recorded_event(
                    logged_at=decision_time,
                    sample_id="sample-4",
                    ticker=ticker,
                    side="NO",
                    feature_row_id=feature_row_id,
                    market_event_id=market_event_id,
                )
            )
            research_raw_rows.append(
                _research_settled_event(
                    logged_at=settlement_time,
                    sample_id="sample-4",
                    ticker=ticker,
                    settlement_result="NO",
                    realized_pnl_dollars=0.41,
                    side="NO",
                )
            )
        elif index == 6:
            research_raw_rows.append(
                _research_recorded_event(
                    logged_at=decision_time,
                    sample_id="sample-6",
                    ticker=ticker,
                    side="YES",
                    feature_row_id=feature_row_id,
                    market_event_id=market_event_id,
                )
            )
            research_raw_rows.append(
                _research_settled_event(
                    logged_at=settlement_time,
                    sample_id="sample-6",
                    ticker=ticker,
                    settlement_result="YES",
                    realized_pnl_dollars=0.41,
                )
            )

    live_run = _write_live_run(
        tmp_path,
        run_name=live_run_name,
        model=model,
        trade_rows=live_trade_rows,
        signal_rows=live_signal_rows,
        model_outputs=live_model_outputs,
        feature_rows=live_feature_rows,
        extra_execution_rows=extra_execution_rows,
    )
    research_run = _write_research_run(
        tmp_path,
        run_name="research_run",
        model=model,
        raw_rows=research_raw_rows,
        model_outputs=research_model_outputs,
        feature_rows=research_feature_rows,
    )

    artifacts = build_parity_artifacts(
        live_root=tmp_path / "output" / "live",
        research_root=tmp_path / "output" / "live_research",
        as_of_date=date.fromisoformat("2026-04-06"),
        live_runs=(live_run,),
        research_runs=(research_run,),
    )

    paired = artifacts.paired_trades.set_index("decision_id")
    assert paired.loc["decision-1", "research_source"] == "recorded"
    assert paired.loc["decision-1", "research_action"] == "entered_same"
    assert bool(paired.loc["decision-1", "primary_cohort"]) is True

    assert paired.loc["decision-2", "research_action"] == "declined"
    assert float(paired.loc["decision-2", "research_theoretical_pnl_dollars"]) == 0.0
    assert bool(paired.loc["decision-2", "primary_cohort"]) is False

    assert paired.loc["decision-3", "research_action"] == "entered_different_size"
    assert bool(paired.loc["decision-3", "primary_cohort"]) is False

    assert paired.loc["decision-4", "research_action"] == "entered_different_side"
    assert bool(paired.loc["decision-4", "primary_cohort"]) is False

    assert paired.loc["decision-5", "research_source"] == "replayed"
    assert bool(paired.loc["decision-5", "primary_cohort"]) is False

    assert paired.loc["decision-6", "match_quality"] == "model_version_mismatch"
    assert bool(paired.loc["decision-6", "primary_cohort"]) is False

    assert not artifacts.daily_rollup.empty
    assert "shadow_minus_research_pnl_dollars_ci_status" in artifacts.daily_rollup.columns
    assert set(artifacts.daily_segment_rollup["segment"]) >= {
        "declined",
        "entered_different_side",
        "entered_different_size",
        "partial_fill_live",
    }
    assert artifacts.summary["schema_version"] == PARITY_SCHEMA_VERSION
    assert artifacts.summary["matched_snapshot_count"] >= 5


def test_parity_report_handles_single_cluster_day_and_positive_breach(tmp_path: Path) -> None:
    base_time = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    model = "bagged_lasso"
    live_signal_rows: list[dict] = []
    live_trade_rows: list[dict] = []
    live_model_outputs: list[dict] = []
    live_feature_rows: list[dict] = []
    research_raw_rows: list[dict] = [_research_started_event(base_time - timedelta(minutes=5))]
    research_model_outputs: list[dict] = []
    research_feature_rows: list[dict] = []

    for index in range(10):
        event_time = base_time + timedelta(days=index)
        feature_row_id = f"feature:pos-{index}"
        market_event_id = f"pos-{index}"
        ticker = f"KXBTC15M-P{index}"
        live_signal_rows.append(
            _live_signal_event(
                logged_at=event_time,
                decision_id=f"pos-{index}",
                ticker=ticker,
                side="YES",
            )
        )
        live_trade_rows.append(
            _live_settlement_event(
                logged_at=event_time + timedelta(minutes=10),
                decision_id=f"pos-{index}",
                ticker=ticker,
                side="YES",
                settlement_result="YES",
                realized_pnl_dollars=0.44,
                cash_required_dollars=0.56,
            )
        )
        model_rows, feature_rows = _base_archive_rows(
            run_name="pos_live",
            model=model,
            event_time=event_time,
            ticker=ticker,
            feature_row_id=feature_row_id,
            market_event_id=market_event_id,
            model_file="artifacts/kalshi/live_model.joblib",
        )
        live_model_outputs.extend(model_rows)
        live_feature_rows.extend(feature_rows)
        research_model_outputs.extend(model_rows)
        research_feature_rows.extend(feature_rows)
        research_raw_rows.append(
            _research_recorded_event(
                logged_at=event_time,
                sample_id=f"sample-pos-{index}",
                ticker=ticker,
                side="YES",
                feature_row_id=feature_row_id,
                market_event_id=market_event_id,
            )
        )
        research_raw_rows.append(
            _research_settled_event(
                logged_at=event_time + timedelta(minutes=10),
                sample_id=f"sample-pos-{index}",
                ticker=ticker,
                settlement_result="YES",
                realized_pnl_dollars=0.30,
            )
        )

    _write_live_run(
        tmp_path,
        run_name="pos_live",
        model=model,
        trade_rows=live_trade_rows,
        signal_rows=live_signal_rows,
        model_outputs=live_model_outputs,
        feature_rows=live_feature_rows,
    )
    _write_research_run(
        tmp_path,
        run_name="pos_research",
        model=model,
        raw_rows=research_raw_rows,
        model_outputs=research_model_outputs,
        feature_rows=research_feature_rows,
    )

    artifacts = build_parity_artifacts(
        live_root=tmp_path / "output" / "live",
        research_root=tmp_path / "output" / "live_research",
        as_of_date=date.fromisoformat("2026-04-10"),
        economic_floor_per_trade=0.01,
    )
    rolling = artifacts.rolling_rollup.set_index("report_day")
    latest = rolling.loc["2026-04-10"]
    assert bool(latest["parity_breach"]) is True
    assert latest["leak_label"] == "live_execution_favorable"

    single_day = artifacts.daily_rollup.set_index("report_day").loc["2026-04-01"]
    assert single_day["shadow_minus_research_pnl_dollars_ci_status"] == "insufficient_clusters"


def test_parity_report_script_writes_timestamped_and_latest_outputs(tmp_path: Path) -> None:
    base_time = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    model = "bagged_lasso"
    feature_row_id = "feature:cli"
    market_event_id = "cli-event"
    ticker = "KXBTC15M-CLI"
    model_rows, feature_rows = _base_archive_rows(
        run_name="cli_live",
        model=model,
        event_time=base_time,
        ticker=ticker,
        feature_row_id=feature_row_id,
        market_event_id=market_event_id,
        model_file="artifacts/kalshi/live_model.joblib",
    )
    _write_live_run(
        tmp_path,
        run_name="cli_live",
        model=model,
        trade_rows=[
            _live_settlement_event(
                logged_at=base_time + timedelta(minutes=10),
                decision_id="cli-1",
                ticker=ticker,
                side="YES",
                settlement_result="YES",
                realized_pnl_dollars=0.44,
                cash_required_dollars=0.56,
            )
        ],
        signal_rows=[
            _live_signal_event(
                logged_at=base_time,
                decision_id="cli-1",
                ticker=ticker,
                side="YES",
            )
        ],
        model_outputs=model_rows,
        feature_rows=feature_rows,
    )
    _write_research_run(
        tmp_path,
        run_name="cli_research",
        model=model,
        raw_rows=[
            _research_started_event(base_time - timedelta(minutes=5)),
            _research_recorded_event(
                logged_at=base_time,
                sample_id="sample-cli",
                ticker=ticker,
                side="YES",
                feature_row_id=feature_row_id,
                market_event_id=market_event_id,
            ),
            _research_settled_event(
                logged_at=base_time + timedelta(minutes=10),
                sample_id="sample-cli",
                ticker=ticker,
                settlement_result="YES",
                realized_pnl_dollars=0.41,
            ),
        ],
        model_outputs=model_rows,
        feature_rows=feature_rows,
    )

    output_root = tmp_path / "artifacts" / "kalshi" / "parity_reports"
    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts") / "parity_report.py"),
            "--live-root",
            str(tmp_path / "output" / "live"),
            "--research-root",
            str(tmp_path / "output" / "live_research"),
            "--output-root",
            str(output_root),
            "--as-of-date",
            "2026-04-01",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    dated_dir = output_root / "2026-04-01"
    latest_dir = output_root / "latest"
    assert dated_dir.exists()
    assert latest_dir.exists()
    assert (dated_dir / "paired_trades.parquet").exists()
    assert (dated_dir / "summary.json").exists()
    assert (dated_dir / "parity_report.md").exists()
    assert (dated_dir / "parity_gaps.png").exists()
    assert "14d research/shadow/live net PnL:" in result.stdout

    second = subprocess.run(
        [
            sys.executable,
            str(Path("scripts") / "parity_report.py"),
            "--live-root",
            str(tmp_path / "output" / "live"),
            "--research-root",
            str(tmp_path / "output" / "live_research"),
            "--output-root",
            str(output_root),
            "--as-of-date",
            "2026-04-01",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Parity report written to" in second.stdout
