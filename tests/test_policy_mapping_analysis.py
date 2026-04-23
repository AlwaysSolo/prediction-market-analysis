from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.live.kalshi.offline_training import PolicyConfig
from src.live.kalshi.policy_mapping_analysis import (
    annotate_policy_gate_status,
    bootstrap_bucket_delta_correlation,
    build_common_prediction_frame,
    compute_slice_log_loss_matrix,
    summarize_threshold_crossings,
)


def test_build_common_prediction_frame_aligns_common_rows_and_trade_pnl() -> None:
    left_predictions = pd.DataFrame(
        {
            "trade_id": ["a", "b", "c"],
            "ticker": ["T1", "T2", "T3"],
            "created_time": pd.to_datetime(
                ["2026-03-10T00:00:00Z", "2026-03-10T00:01:00Z", "2026-03-10T00:02:00Z"],
                utc=True,
            ),
            "close_time": pd.to_datetime(
                ["2026-03-10T00:15:00Z", "2026-03-10T00:16:00Z", "2026-03-10T00:17:00Z"],
                utc=True,
            ),
            "actual_outcome": [1, 0, 1],
            "market_prob": [0.50, 0.45, 0.55],
            "tau_minutes": [5.0, 6.0, 7.0],
            "calibrated_probability": [0.70, 0.40, 0.65],
            "realized_vol_regime_source": ["spot", "spot", "spot"],
            "realized_vol_regime_bucket": ["medium", "medium", "high"],
            "realized_vol_regime_bucket_version": ["spot_2026q1_backfill_v1"] * 3,
            "realized_vol_regime_metric": [55.0, 58.0, 91.0],
            "hour_of_day_et": [0, 0, 1],
            "hour_of_day_et_label": ["00", "00", "01"],
            "session_block_et": ["asia", "asia", "asia"],
        }
    )
    right_predictions = pd.DataFrame(
        {
            "trade_id": ["b", "c", "d"],
            "ticker": ["T2", "T3", "T4"],
            "created_time": pd.to_datetime(
                ["2026-03-10T00:01:00Z", "2026-03-10T00:02:00Z", "2026-03-10T00:03:00Z"],
                utc=True,
            ),
            "close_time": pd.to_datetime(
                ["2026-03-10T00:16:00Z", "2026-03-10T00:17:00Z", "2026-03-10T00:18:00Z"],
                utc=True,
            ),
            "actual_outcome": [0, 1, 0],
            "market_prob": [0.45, 0.55, 0.40],
            "tau_minutes": [6.0, 7.0, 8.0],
            "calibrated_probability": [0.42, 0.68, 0.35],
            "realized_vol_regime_source": ["spot", "spot", "spot"],
            "realized_vol_regime_bucket": ["medium", "high", "medium"],
            "realized_vol_regime_bucket_version": ["spot_2026q1_backfill_v1"] * 3,
            "realized_vol_regime_metric": [58.0, 91.0, 59.0],
            "hour_of_day_et": [0, 1, 1],
            "hour_of_day_et_label": ["00", "01", "01"],
            "session_block_et": ["asia", "asia", "asia"],
        }
    )
    left_trade_records = pd.DataFrame({"trade_id": ["b"], "net_pnl_dollars": [1.25]})
    right_trade_records = pd.DataFrame({"trade_id": ["c"], "net_pnl_dollars": [2.50]})

    frame = build_common_prediction_frame(
        left_predictions,
        right_predictions,
        left_label="default",
        right_label="spot_v1",
        left_trade_records=left_trade_records,
        right_trade_records=right_trade_records,
    )

    assert frame["trade_id"].tolist() == ["b", "c"]
    assert frame["default_calibrated_probability"].tolist() == [0.40, 0.65]
    assert frame["spot_v1_calibrated_probability"].tolist() == [0.42, 0.68]
    assert frame["default_selected"].tolist() == [True, False]
    assert frame["spot_v1_selected"].tolist() == [False, True]
    assert frame["default_net_pnl_dollars"].tolist() == [1.25, 0.0]
    assert frame["spot_v1_net_pnl_dollars"].tolist() == [0.0, 2.50]


def test_annotate_policy_gate_status_marks_trade_eligibility_and_threshold_bands() -> None:
    frame = pd.DataFrame(
        {
            "trade_id": ["tight_pass", "tight_fail", "price_fail", "tau_fail"],
            "ticker": ["A", "B", "C", "D"],
            "created_time": pd.to_datetime(["2026-03-10T00:00:00Z"] * 4, utc=True),
            "close_time": pd.to_datetime(["2026-03-10T00:15:00Z"] * 4, utc=True),
            "actual_outcome": [1, 0, 1, 0],
            "market_prob": [0.50, 0.50, 0.95, 0.50],
            "tau_minutes": [5.0, 5.0, 5.0, 13.0],
            "default_calibrated_probability": [0.54, 0.53, 0.97, 0.60],
        }
    )
    config = PolicyConfig(
        edge_threshold_cents=1.0,
        min_tau_minutes=0.0,
        max_tau_minutes=12.0,
        price_band_min_cents=10,
        price_band_max_cents=90,
        reserve_cash_pct=30.0,
    )

    annotated = annotate_policy_gate_status(
        frame,
        probability_column="default_calibrated_probability",
        policy_config=config,
        output_prefix="default",
        near_threshold_band_cents=2.0,
    )

    tight_pass = annotated.loc[annotated["trade_id"] == "tight_pass"].iloc[0]
    tight_fail = annotated.loc[annotated["trade_id"] == "tight_fail"].iloc[0]
    price_fail = annotated.loc[annotated["trade_id"] == "price_fail"].iloc[0]
    tau_fail = annotated.loc[annotated["trade_id"] == "tau_fail"].iloc[0]

    assert bool(tight_pass["default_pre_path_trade_eligible"]) is True
    assert bool(tight_pass["default_near_threshold"]) is True
    assert bool(tight_pass["default_threshold_crossed"]) is True

    assert bool(tight_fail["default_pre_path_trade_eligible"]) is False
    assert bool(tight_fail["default_near_threshold"]) is True
    assert bool(tight_fail["default_threshold_crossed"]) is False

    assert bool(price_fail["default_in_price_band"]) is False
    assert bool(price_fail["default_pre_path_trade_eligible"]) is False
    assert bool(price_fail["default_near_threshold"]) is False

    assert bool(tau_fail["default_in_tau_band"]) is False
    assert bool(tau_fail["default_pre_path_trade_eligible"]) is False


def test_compute_slice_log_loss_matrix_uses_shared_masks_for_both_models() -> None:
    frame = pd.DataFrame(
        {
            "trade_id": ["a", "b", "c", "d"],
            "actual_outcome": [1, 0, 1, 0],
            "default_calibrated_probability": [0.80, 0.25, 0.55, 0.45],
            "spot_v1_calibrated_probability": [0.75, 0.35, 0.70, 0.40],
            "default_selected": [True, False, True, False],
            "spot_v1_selected": [False, True, True, False],
            "default_pre_path_trade_eligible": [True, False, True, False],
            "spot_v1_pre_path_trade_eligible": [False, True, True, False],
            "default_near_threshold": [False, True, False, True],
            "spot_v1_near_threshold": [True, False, True, False],
        }
    )

    rows = compute_slice_log_loss_matrix(frame, left_label="default", right_label="spot_v1")
    by_slice = {row["slice_name"]: row for row in rows}

    assert by_slice["all_rows"]["rows"] == 4
    assert by_slice["default_selected_rows"]["rows"] == 2
    assert by_slice["spot_v1_selected_rows"]["rows"] == 2
    assert by_slice["default_near_threshold"]["rows"] == 2

    expected_default_selected = log_loss([1, 1], [0.80, 0.55], labels=[0, 1])
    expected_spot_on_default_selected = log_loss([1, 1], [0.75, 0.70], labels=[0, 1])
    assert by_slice["default_selected_rows"]["left_log_loss"] == expected_default_selected
    assert by_slice["default_selected_rows"]["right_log_loss"] == expected_spot_on_default_selected


def test_summarize_threshold_crossings_counts_crossers_by_bucket() -> None:
    frame = pd.DataFrame(
        {
            "realized_vol_regime_bucket": ["medium", "medium", "high", "high"],
            "default_pre_path_trade_eligible": [True, False, True, False],
            "spot_v1_pre_path_trade_eligible": [True, True, True, False],
            "default_threshold_crossed": [True, False, True, False],
            "spot_v1_threshold_crossed": [True, True, True, False],
        }
    )

    rows = summarize_threshold_crossings(
        frame,
        group_columns=["realized_vol_regime_bucket"],
        left_label="default",
        right_label="spot_v1",
    )
    by_bucket = {row["realized_vol_regime_bucket"]: row for row in rows}

    assert by_bucket["medium"]["default_threshold_crossings"] == 1
    assert by_bucket["medium"]["spot_v1_threshold_crossings"] == 2
    assert by_bucket["medium"]["crossing_delta"] == 1
    assert by_bucket["high"]["default_threshold_crossings"] == 1
    assert by_bucket["high"]["spot_v1_threshold_crossings"] == 1


def test_bootstrap_bucket_delta_correlation_returns_confidence_band() -> None:
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(7)
    for bucket, base in enumerate((0.2, 0.5, 0.8), start=1):
        for _ in range(60):
            ll_delta = float(base + rng.normal(0.0, 0.05))
            pnl_delta = float(10.0 * base + rng.normal(0.0, 0.5))
            records.append(
                {
                    "bucket": f"b{bucket}",
                    "logloss_delta_contribution": ll_delta,
                    "pnl_delta_dollars": pnl_delta,
                }
            )
    frame = pd.DataFrame(records)

    summary = bootstrap_bucket_delta_correlation(
        frame,
        bucket_column="bucket",
        logloss_delta_column="logloss_delta_contribution",
        pnl_delta_column="pnl_delta_dollars",
        iterations=200,
        seed=11,
    )

    assert summary["bucket_count"] == 3
    assert summary["point_estimate"] > 0.8
    assert summary["bootstrap_p05"] > 0.5
    assert summary["bootstrap_p95"] <= 1.0
