from __future__ import annotations

import pandas as pd

from src.live.kalshi.offline_training import PolicyConfig
from src.live.kalshi.policy_threshold_retune import (
    evaluate_standard_policy_with_threshold_overrides,
    summarize_tail_calibration,
    summarize_trade_pnl_distribution,
    search_regime_threshold_offsets,
    split_time_ordered_frame,
    summarize_probability_distribution,
)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": ["m1", "m2", "h1", "h2"],
            "ticker": ["M1", "M2", "H1", "H2"],
            "created_time": pd.to_datetime(
                [
                    "2026-03-10T00:00:00Z",
                    "2026-03-10T00:01:00Z",
                    "2026-03-10T00:02:00Z",
                    "2026-03-10T00:03:00Z",
                ],
                utc=True,
            ),
            "close_time": pd.to_datetime(
                [
                    "2026-03-10T00:15:00Z",
                    "2026-03-10T00:16:00Z",
                    "2026-03-10T00:17:00Z",
                    "2026-03-10T00:18:00Z",
                ],
                utc=True,
            ),
            "actual_outcome": [0, 0, 1, 1],
            "market_prob": [0.50, 0.50, 0.50, 0.50],
            "tau_minutes": [5.0, 5.0, 5.0, 5.0],
            "realized_vol_regime_bucket": ["medium", "medium", "high", "high"],
            "spot_v1_calibrated_probability": [0.54, 0.54, 0.60, 0.60],
        }
    )


def test_split_time_ordered_frame_returns_non_overlapping_halves() -> None:
    frame = _base_frame()

    fit_df, eval_df = split_time_ordered_frame(frame, time_column="created_time", fit_fraction=0.5)

    assert fit_df["trade_id"].tolist() == ["m1", "m2"]
    assert eval_df["trade_id"].tolist() == ["h1", "h2"]


def test_summarize_probability_distribution_reports_quantiles_and_crossing_rate() -> None:
    frame = _base_frame()
    frame["spot_v1_threshold_crossed"] = [True, True, True, True]
    frame["default_threshold_crossed"] = [True, False, True, False]

    rows = summarize_probability_distribution(
        frame,
        group_columns=["realized_vol_regime_bucket"],
        probability_columns={"default": "market_prob", "spot_v1": "spot_v1_calibrated_probability"},
        crossing_columns={"default": "default_threshold_crossed", "spot_v1": "spot_v1_threshold_crossed"},
    )
    medium = next(row for row in rows if row["realized_vol_regime_bucket"] == "medium")

    assert medium["rows"] == 2
    assert medium["spot_v1_probability_q50"] == 0.54
    assert medium["default_crossing_rate"] == 0.5
    assert medium["spot_v1_crossing_rate"] == 1.0


def test_evaluate_standard_policy_with_threshold_overrides_skips_medium_candidates() -> None:
    frame = _base_frame()
    config = PolicyConfig(
        edge_threshold_cents=1.0,
        min_tau_minutes=0.0,
        max_tau_minutes=12.0,
        price_band_min_cents=10,
        price_band_max_cents=90,
        reserve_cash_pct=30.0,
    )

    baseline_metrics, _baseline_trades = evaluate_standard_policy_with_threshold_overrides(
        frame,
        probability_column="spot_v1_calibrated_probability",
        config=config,
        overall_log_loss=0.0,
        threshold_cents_by_bucket={"medium": 1.0, "high": 1.0},
    )
    stricter_metrics, stricter_trades = evaluate_standard_policy_with_threshold_overrides(
        frame,
        probability_column="spot_v1_calibrated_probability",
        config=config,
        overall_log_loss=0.0,
        threshold_cents_by_bucket={"medium": 2.0, "high": 1.0},
    )

    assert baseline_metrics["trades"] == 4
    assert stricter_metrics["trades"] == 2
    assert set(stricter_trades["trade_id"]) == {"h1", "h2"}


def test_search_regime_threshold_offsets_prefers_stricter_medium_threshold_on_fit_half() -> None:
    frame = _base_frame()
    config = PolicyConfig(
        edge_threshold_cents=1.0,
        min_tau_minutes=0.0,
        max_tau_minutes=12.0,
        price_band_min_cents=10,
        price_band_max_cents=90,
        reserve_cash_pct=30.0,
    )

    fit_df, _eval_df = split_time_ordered_frame(frame, time_column="created_time", fit_fraction=0.5)

    best = search_regime_threshold_offsets(
        fit_df,
        probability_column="spot_v1_calibrated_probability",
        config=config,
        overall_log_loss=0.0,
        global_threshold_grid=[1.0],
        medium_offsets=[0.0, 1.0],
        high_offsets=[0.0],
    )

    assert best["best_candidate"]["threshold_cents_by_bucket"]["medium"] == 2.0


def test_summarize_tail_calibration_reports_low_and_high_tails() -> None:
    frame = pd.DataFrame(
        {
            "actual_outcome": [0, 0, 1, 1, 1],
            "market_prob": [0.08, 0.09, 0.91, 0.93, 0.95],
            "spot_v1_calibrated_probability": [0.05, 0.08, 0.91, 0.94, 0.97],
        }
    )

    summary = summarize_tail_calibration(
        frame,
        probability_column="spot_v1_calibrated_probability",
        low_cutoff=0.10,
        high_cutoff=0.90,
    )

    assert summary["low_tail"]["rows"] == 2
    assert summary["high_tail"]["rows"] == 3
    assert summary["low_tail"]["absolute_calibration_error"] >= 0.0
    assert summary["high_tail"]["absolute_calibration_error"] >= 0.0


def test_summarize_trade_pnl_distribution_reports_shape_stats() -> None:
    frame = pd.DataFrame(
        {
            "trade_id": ["a", "b", "c", "d"],
            "net_pnl_dollars": [-1.0, 0.5, 1.0, -0.25],
        }
    )

    summary = summarize_trade_pnl_distribution(frame, pnl_column="net_pnl_dollars")

    assert summary["rows"] == 4
    assert summary["wins"] == 2
    assert summary["win_rate"] == 0.5
    assert summary["mean_net_pnl_dollars"] == 0.0625
