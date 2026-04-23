from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.compare_walkforward_fixed_policy import (
    _load_run_bundle,
    run_fixed_policy_walkforward_comparison,
)


class _IdentityCalibration:
    def apply(self, values):
        return np.asarray(values, dtype=np.float64)


def _write_run_dir(
    run_dir: Path,
    *,
    feature_schema: str,
    best_params: dict[str, object],
    policy_config: dict[str, object],
    ordered_tickers: list[str],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "datasets" / "all").mkdir(parents=True, exist_ok=True)
    payload = {
        "series": "KXBTC15M",
        "feature_schema": feature_schema,
        "best_params": best_params,
        "model_family": "lightgbm",
    }
    (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "policy.json").write_text(json.dumps({"config": policy_config}), encoding="utf-8")
    split_manifest = {
        "series": "KXBTC15M",
        "train_tickers": ordered_tickers[:2],
        "validation_tickers": ordered_tickers[2:4],
        "test_tickers": ordered_tickers[4:6],
    }
    (run_dir / "split_manifest.json").write_text(json.dumps(split_manifest), encoding="utf-8")


def test_run_fixed_policy_walkforward_comparison_builds_cross_model_policy_matrix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    ordered_tickers = ["A", "B", "C", "D", "E", "F"]
    _write_run_dir(
        left_dir,
        feature_schema="default",
        best_params={"tag": "left"},
        policy_config={
            "edge_threshold_cents": 1.0,
            "min_tau_minutes": 0.0,
            "max_tau_minutes": 12.0,
            "price_band_min_cents": 10,
            "price_band_max_cents": 90,
            "reserve_cash_pct": 30.0,
        },
        ordered_tickers=ordered_tickers,
    )
    _write_run_dir(
        right_dir,
        feature_schema="spot_v1",
        best_params={"tag": "right"},
        policy_config={
            "edge_threshold_cents": 4.0,
            "min_tau_minutes": 0.0,
            "max_tau_minutes": 12.0,
            "price_band_min_cents": 20,
            "price_band_max_cents": 80,
            "reserve_cash_pct": 30.0,
        },
        ordered_tickers=ordered_tickers,
    )

    module = __import__("scripts.compare_walkforward_fixed_policy", fromlist=["unused"])

    folds = [
        SimpleNamespace(fold_index=1, train_tickers=("A",), validation_tickers=("B",), test_tickers=("C",)),
        SimpleNamespace(fold_index=2, train_tickers=("A", "B"), validation_tickers=("C",), test_tickers=("D",)),
    ]
    monkeypatch.setattr(module, "build_walk_forward_folds", lambda _markets_df: folds)

    def fake_load_feature_dataset(_cache_dir, tickers, *, progress_desc=None, feature_schema="default"):
        rows = []
        for index, ticker in enumerate(tickers):
            rows.append(
                {
                    "ticker": ticker,
                    "trade_id": f"{ticker}-{feature_schema}-{index}",
                    "created_time": pd.Timestamp("2026-03-10T00:00:00Z"),
                    "close_time": pd.Timestamp("2026-03-10T00:15:00Z"),
                    "market_prob": 0.5,
                    "tau_minutes": 5.0,
                    "actual_outcome": index % 2,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(module, "load_feature_dataset", fake_load_feature_dataset)
    monkeypatch.setattr(module, "train_lightgbm_model", lambda train_df, validation_df, params: (params["tag"], {"rows": len(train_df) + len(validation_df)}))

    def fake_raw_predictions(model, frame):
        base = 0.8 if model == "left" else 0.6
        return np.full(len(frame), base, dtype=np.float64)

    monkeypatch.setattr(module, "raw_predictions", fake_raw_predictions)
    monkeypatch.setattr(module, "fit_platt_scaler", lambda raw, actuals: _IdentityCalibration())

    def fake_evaluate_policy(frame, probabilities, config, overall_log_loss, progress_desc=None):
        is_left_model = float(probabilities[0]) > 0.7
        base_pnl = 10.0 if is_left_model else 7.0
        net_pnl = base_pnl - float(config.edge_threshold_cents)
        return {
            "objective": net_pnl - overall_log_loss,
            "trades": int(len(frame)),
            "net_pnl_dollars": net_pnl,
            "max_drawdown_dollars": abs(net_pnl) / 2.0,
            "max_drawdown_pct": abs(net_pnl) / 100.0,
            "return_pct": net_pnl / 100.0,
            "log_loss": overall_log_loss,
            "skipped_due_open_ticker": 0,
            "skipped_due_price_band": 0,
            "skipped_due_regime": 0,
            "skipped_due_post_cost_edge": 0,
        }

    monkeypatch.setattr(module, "evaluate_policy", fake_evaluate_policy)

    left_bundle = _load_run_bundle(left_dir, label="default")
    right_bundle = _load_run_bundle(right_dir, label="spot_v1")

    comparison = run_fixed_policy_walkforward_comparison(left_bundle, right_bundle)

    assert len(comparison["fold_results"]) == 8
    aggregate_rows = comparison["aggregate_results"]
    assert len(aggregate_rows) == 4

    left_under_left_policy = next(
        row
        for row in aggregate_rows
        if row["model_label"] == "default" and row["policy_label"] == "default_policy"
    )
    right_under_left_policy = next(
        row
        for row in aggregate_rows
        if row["model_label"] == "spot_v1" and row["policy_label"] == "default_policy"
    )
    assert left_under_left_policy["total_test_pnl_dollars"] == 18.0
    assert right_under_left_policy["total_test_pnl_dollars"] == 12.0

    pairwise = comparison["pairwise_policy_summary"]
    default_policy_summary = next(row for row in pairwise if row["policy_label"] == "default_policy")
    assert default_policy_summary["left_minus_right_total_test_pnl_dollars"] == 6.0
    assert default_policy_summary["left_label"] == "default"
    assert default_policy_summary["right_label"] == "spot_v1"

