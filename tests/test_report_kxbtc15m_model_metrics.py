from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from scripts.report_kxbtc15m_model_metrics import (
    PredictionMetrics,
    RunReport,
    TradeMetrics,
    _compute_prediction_metrics,
    _compute_trade_metrics,
    _paired_base_and_hourly_reports,
    _yes_precision_bucket_table,
)


def test_compute_prediction_metrics_uses_calibrated_probabilities_and_threshold() -> None:
    frame = pd.DataFrame(
        [
            {"actual_outcome": 1, "raw_probability": 0.2, "calibrated_probability": 0.8},
            {"actual_outcome": 0, "raw_probability": 0.9, "calibrated_probability": 0.6},
            {"actual_outcome": 0, "raw_probability": 0.1, "calibrated_probability": 0.4},
            {"actual_outcome": 1, "raw_probability": 0.7, "calibrated_probability": 0.3},
        ]
    )

    metrics = _compute_prediction_metrics(frame, threshold=0.5)

    assert metrics.probability_column == "calibrated_probability"
    assert metrics.true_positive == 1
    assert metrics.false_positive == 1
    assert metrics.true_negative == 1
    assert metrics.false_negative == 1
    assert math.isclose(metrics.accuracy or 0.0, 0.5)
    assert math.isclose(metrics.yes_precision or 0.0, 0.5)
    assert math.isclose(metrics.yes_recall or 0.0, 0.5)
    assert math.isclose(metrics.no_precision or 0.0, 0.5)
    assert math.isclose(metrics.no_recall or 0.0, 0.5)
    assert math.isclose(metrics.predicted_yes_rate or 0.0, 0.5)
    assert math.isclose(metrics.actual_yes_rate or 0.0, 0.5)


def test_compute_trade_metrics_breaks_out_yes_and_no_hit_rates() -> None:
    trade_records = pd.DataFrame(
        [
            {"side": "YES", "is_win": True, "net_pnl_dollars": 1.5},
            {"side": "YES", "is_win": False, "net_pnl_dollars": -0.5},
            {"side": "NO", "is_win": True, "net_pnl_dollars": 0.8},
        ]
    )

    metrics = _compute_trade_metrics(trade_records)

    assert metrics.settled_trade_count == 3
    assert metrics.yes_trade_count == 2
    assert metrics.no_trade_count == 1
    assert math.isclose(metrics.trade_hit_rate or 0.0, 2 / 3)
    assert math.isclose(metrics.yes_trade_hit_rate or 0.0, 0.5)
    assert math.isclose(metrics.no_trade_hit_rate or 0.0, 1.0)
    assert math.isclose(metrics.net_pnl_dollars or 0.0, 1.8)
    assert math.isclose(metrics.average_trade_pnl_dollars or 0.0, 0.6)


def test_yes_precision_bucket_table_uses_10c_price_buckets(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        [
            {"market_prob": 0.14, "actual_outcome": 1, "calibrated_probability": 0.60},
            {"market_prob": 0.16, "actual_outcome": 0, "calibrated_probability": 0.80},
            {"market_prob": 0.16, "actual_outcome": 1, "calibrated_probability": 0.20},
            {"market_prob": 0.84, "actual_outcome": 1, "calibrated_probability": 0.90},
        ]
    )
    predictions.to_parquet(tmp_path / "test_predictions.parquet")

    table = _yes_precision_bucket_table(tmp_path, probability_column="calibrated_probability", threshold=0.5)

    assert "| 10-20c | 3 | 2 | 1 | 50.00% |" in table
    assert "| 80-90c | 1 | 1 | 1 | 100.00% |" in table


def test_pairing_prefers_closest_base_run_name_for_hourly_context() -> None:
    metrics = PredictionMetrics(
        row_count=1,
        test_log_loss=0.5,
        brier_score=0.2,
        accuracy=0.5,
        yes_precision=0.5,
        yes_recall=0.5,
        no_precision=0.5,
        no_recall=0.5,
        predicted_yes_rate=0.5,
        actual_yes_rate=0.5,
        true_positive=1,
        false_positive=1,
        true_negative=1,
        false_negative=1,
        probability_column="calibrated_probability",
    )
    trade_metrics = TradeMetrics(
        settled_trade_count=1,
        trade_hit_rate=0.5,
        yes_trade_count=1,
        no_trade_count=0,
        yes_trade_hit_rate=0.5,
        no_trade_hit_rate=None,
        net_pnl_dollars=1.0,
        average_trade_pnl_dollars=1.0,
    )
    reports = [
        RunReport(
            run_dir=Path("artifacts/base1"),
            run_name="kxbtc15m_l1_compare_v1_lasso_l1r100",
            model_family="lasso",
            context_label="base 15m features",
            feature_count=29,
            prediction_metrics=metrics,
            trade_metrics=trade_metrics,
        ),
        RunReport(
            run_dir=Path("artifacts/base2"),
            run_name="kxbtc15m_lasso_v1",
            model_family="lasso",
            context_label="base 15m features",
            feature_count=29,
            prediction_metrics=metrics,
            trade_metrics=trade_metrics,
        ),
        RunReport(
            run_dir=Path("artifacts/hourly"),
            run_name="kxbtc15m_lasso_kxbtcd_hourly_ctx_v1",
            model_family="lasso",
            context_label="KXBTCD hourly context",
            feature_count=41,
            prediction_metrics=metrics,
            trade_metrics=trade_metrics,
        ),
    ]

    pairs = _paired_base_and_hourly_reports(reports)

    assert len(pairs) == 1
    base_report, hourly_report = pairs[0]
    assert base_report.run_name == "kxbtc15m_lasso_v1"
    assert hourly_report.run_name == "kxbtc15m_lasso_kxbtcd_hourly_ctx_v1"
