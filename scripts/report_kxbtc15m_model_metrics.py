from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


DEFAULT_ARTIFACTS_ROOT = Path("artifacts") / "kalshi"
DEFAULT_OUTPUT_PATH = DEFAULT_ARTIFACTS_ROOT / "kxbtc15m_model_metrics_report.md"


@dataclass(frozen=True)
class PredictionMetrics:
    row_count: int
    test_log_loss: float | None
    brier_score: float | None
    accuracy: float | None
    yes_precision: float | None
    yes_recall: float | None
    no_precision: float | None
    no_recall: float | None
    predicted_yes_rate: float | None
    actual_yes_rate: float | None
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    probability_column: str


@dataclass(frozen=True)
class TradeMetrics:
    settled_trade_count: int
    trade_hit_rate: float | None
    yes_trade_count: int
    no_trade_count: int
    yes_trade_hit_rate: float | None
    no_trade_hit_rate: float | None
    net_pnl_dollars: float | None
    average_trade_pnl_dollars: float | None


@dataclass(frozen=True)
class RunReport:
    run_dir: Path
    run_name: str
    model_family: str
    context_label: str
    feature_count: int | None
    prediction_metrics: PredictionMetrics
    trade_metrics: TradeMetrics


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _format_pct(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "-"
    return f"{value * 100.0:.2f}%"


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None or np.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def _format_money(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "-"
    return f"${value:,.2f}"


def _format_signed_float(value: float | None, digits: int = 4) -> str:
    if value is None or np.isnan(value):
        return "-"
    return f"{value:+.{digits}f}"


def _format_delta_pct_points(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "-"
    return f"{value * 100.0:+.2f} pp"


def _format_signed_money(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "-"
    return f"{value:+,.2f}$"


def _format_signed_int(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:+d}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _probability_column(predictions: pd.DataFrame) -> str:
    if "calibrated_probability" in predictions.columns:
        return "calibrated_probability"
    if "raw_probability" in predictions.columns:
        return "raw_probability"
    raise ValueError("Predictions parquet is missing both calibrated_probability and raw_probability.")


def _compute_prediction_metrics(predictions: pd.DataFrame, *, threshold: float) -> PredictionMetrics:
    probability_column = _probability_column(predictions)
    probabilities = predictions[probability_column].to_numpy(dtype=np.float64, copy=True)
    actual = predictions["actual_outcome"].to_numpy(dtype=np.int8, copy=True)
    predicted_yes = probabilities >= float(threshold)
    actual_yes = actual == 1

    true_positive = int(np.count_nonzero(predicted_yes & actual_yes))
    false_positive = int(np.count_nonzero(predicted_yes & ~actual_yes))
    true_negative = int(np.count_nonzero(~predicted_yes & ~actual_yes))
    false_negative = int(np.count_nonzero(~predicted_yes & actual_yes))

    test_log_loss: float | None
    try:
        test_log_loss = float(log_loss(actual, probabilities, labels=[0, 1]))
    except ValueError:
        test_log_loss = None

    brier_score = float(np.mean((probabilities - actual.astype(np.float64)) ** 2)) if len(actual) else None
    accuracy = _safe_divide(true_positive + true_negative, len(actual))
    yes_precision = _safe_divide(true_positive, true_positive + false_positive)
    yes_recall = _safe_divide(true_positive, true_positive + false_negative)
    no_precision = _safe_divide(true_negative, true_negative + false_negative)
    no_recall = _safe_divide(true_negative, true_negative + false_positive)
    predicted_yes_rate = float(np.mean(predicted_yes)) if len(actual) else None
    actual_yes_rate = float(np.mean(actual_yes)) if len(actual) else None

    return PredictionMetrics(
        row_count=int(len(actual)),
        test_log_loss=test_log_loss,
        brier_score=brier_score,
        accuracy=accuracy,
        yes_precision=yes_precision,
        yes_recall=yes_recall,
        no_precision=no_precision,
        no_recall=no_recall,
        predicted_yes_rate=predicted_yes_rate,
        actual_yes_rate=actual_yes_rate,
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        probability_column=probability_column,
    )


def _compute_trade_metrics(trade_records: pd.DataFrame | None) -> TradeMetrics:
    if trade_records is None or trade_records.empty:
        return TradeMetrics(
            settled_trade_count=0,
            trade_hit_rate=None,
            yes_trade_count=0,
            no_trade_count=0,
            yes_trade_hit_rate=None,
            no_trade_hit_rate=None,
            net_pnl_dollars=None,
            average_trade_pnl_dollars=None,
        )

    trade_count = int(len(trade_records))
    yes_trades = trade_records.loc[trade_records["side"] == "YES"]
    no_trades = trade_records.loc[trade_records["side"] == "NO"]
    net_pnl = float(trade_records["net_pnl_dollars"].sum()) if "net_pnl_dollars" in trade_records.columns else None
    average_trade_pnl = float(trade_records["net_pnl_dollars"].mean()) if "net_pnl_dollars" in trade_records.columns else None

    return TradeMetrics(
        settled_trade_count=trade_count,
        trade_hit_rate=float(trade_records["is_win"].mean()) if "is_win" in trade_records.columns else None,
        yes_trade_count=int(len(yes_trades)),
        no_trade_count=int(len(no_trades)),
        yes_trade_hit_rate=float(yes_trades["is_win"].mean()) if len(yes_trades) and "is_win" in yes_trades.columns else None,
        no_trade_hit_rate=float(no_trades["is_win"].mean()) if len(no_trades) and "is_win" in no_trades.columns else None,
        net_pnl_dollars=net_pnl,
        average_trade_pnl_dollars=average_trade_pnl,
    )


def _context_label(run_dir: Path, summary: dict[str, Any]) -> str:
    summary_context = summary.get("hourly_context_series")
    if isinstance(summary_context, str) and summary_context:
        return f"{summary_context} hourly context"

    feature_manifest_path = run_dir / "feature_manifest.json"
    if feature_manifest_path.exists():
        feature_manifest = _read_json(feature_manifest_path)
        metadata = feature_manifest.get("metadata")
        if isinstance(metadata, dict):
            hourly_series = metadata.get("hourly_context_series")
            if isinstance(hourly_series, str) and hourly_series:
                return f"{hourly_series} hourly context"
    return "base 15m features"


def _model_family(run_dir: Path, summary: dict[str, Any]) -> str:
    model_family = summary.get("model_family")
    if isinstance(model_family, str) and model_family:
        if model_family.startswith("kxbtc15m_"):
            return model_family.removeprefix("kxbtc15m_")
        return model_family
    parent_name = run_dir.parent.name
    if parent_name.startswith("kxbtc15m_"):
        return parent_name.removeprefix("kxbtc15m_")
    return parent_name


def _feature_count(run_dir: Path, summary: dict[str, Any]) -> int | None:
    feature_count_value = summary.get("feature_count")
    if isinstance(feature_count_value, int):
        return feature_count_value
    feature_manifest_path = run_dir / "feature_manifest.json"
    if feature_manifest_path.exists():
        feature_manifest = _read_json(feature_manifest_path)
        feature_order = feature_manifest.get("feature_order")
        if isinstance(feature_order, list):
            return len(feature_order)
    return None


def _discover_run_dirs(artifacts_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for summary_path in artifacts_root.rglob("summary.json"):
        run_dir = summary_path.parent
        if "latest" in run_dir.parts:
            continue
        if not (run_dir / "test_predictions.parquet").exists():
            continue
        run_dirs.append(run_dir)
    return sorted(set(run_dirs))


def _load_run_report(run_dir: Path, *, threshold: float) -> RunReport:
    summary = _read_json(run_dir / "summary.json")
    predictions = pd.read_parquet(run_dir / "test_predictions.parquet")
    trade_records_path = run_dir / "test_trade_records.parquet"
    trade_records = pd.read_parquet(trade_records_path) if trade_records_path.exists() else None

    return RunReport(
        run_dir=run_dir,
        run_name=run_dir.name,
        model_family=_model_family(run_dir, summary),
        context_label=_context_label(run_dir, summary),
        feature_count=_feature_count(run_dir, summary),
        prediction_metrics=_compute_prediction_metrics(predictions, threshold=threshold),
        trade_metrics=_compute_trade_metrics(trade_records),
    )


def _prediction_sort_key(report: RunReport) -> tuple[float, float, float]:
    accuracy = report.prediction_metrics.accuracy
    yes_precision = report.prediction_metrics.yes_precision
    test_log_loss = report.prediction_metrics.test_log_loss
    return (
        float(accuracy) if accuracy is not None else float("-inf"),
        float(yes_precision) if yes_precision is not None else float("-inf"),
        -(float(test_log_loss) if test_log_loss is not None else float("inf")),
    )


def _trade_sort_key(report: RunReport) -> tuple[float, float, float]:
    pnl = report.trade_metrics.net_pnl_dollars
    hit_rate = report.trade_metrics.trade_hit_rate
    settled = report.trade_metrics.settled_trade_count
    return (
        float(pnl) if pnl is not None else float("-inf"),
        float(hit_rate) if hit_rate is not None else float("-inf"),
        float(settled),
    )


def _prediction_metrics_table(reports: list[RunReport]) -> str:
    rows = sorted(reports, key=_prediction_sort_key, reverse=True)
    header = [
        "Rank",
        "Run",
        "Family",
        "Context",
        "Features",
        "Rows",
        "Log-loss",
        "Brier",
        "Accuracy",
        "YES Precision",
        "YES Recall",
        "NO Precision",
        "Pred YES Rate",
        "Actual YES Rate",
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for index, report in enumerate(rows, start=1):
        metrics = report.prediction_metrics
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    report.run_name,
                    report.model_family,
                    report.context_label,
                    str(report.feature_count) if report.feature_count is not None else "-",
                    f"{metrics.row_count:,}",
                    _format_float(metrics.test_log_loss, digits=6),
                    _format_float(metrics.brier_score, digits=6),
                    _format_pct(metrics.accuracy),
                    _format_pct(metrics.yes_precision),
                    _format_pct(metrics.yes_recall),
                    _format_pct(metrics.no_precision),
                    _format_pct(metrics.predicted_yes_rate),
                    _format_pct(metrics.actual_yes_rate),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _trade_metrics_table(reports: list[RunReport]) -> str:
    rows = sorted(reports, key=_trade_sort_key, reverse=True)
    header = [
        "Rank",
        "Run",
        "Family",
        "Context",
        "Settled Trades",
        "Net PnL",
        "Avg Trade PnL",
        "Trade Hit Rate",
        "YES Trades",
        "YES Hit Rate",
        "NO Trades",
        "NO Hit Rate",
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for index, report in enumerate(rows, start=1):
        metrics = report.trade_metrics
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    report.run_name,
                    report.model_family,
                    report.context_label,
                    str(metrics.settled_trade_count),
                    _format_money(metrics.net_pnl_dollars),
                    _format_money(metrics.average_trade_pnl_dollars),
                    _format_pct(metrics.trade_hit_rate),
                    str(metrics.yes_trade_count),
                    _format_pct(metrics.yes_trade_hit_rate),
                    str(metrics.no_trade_count),
                    _format_pct(metrics.no_trade_hit_rate),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _artifact_paths_section(reports: list[RunReport]) -> str:
    lines = ["## Included Runs", ""]
    for report in sorted(reports, key=lambda row: (row.model_family, row.run_name)):
        lines.append(f"- `{report.run_name}`")
        lines.append(f"  - family: `{report.model_family}`")
        lines.append(f"  - context: `{report.context_label}`")
        lines.append(f"  - path: `{report.run_dir}`")
    return "\n".join(lines)


def _yes_precision_bucket_table(run_dir: Path, *, probability_column: str, threshold: float) -> str:
    predictions = pd.read_parquet(
        run_dir / "test_predictions.parquet",
        columns=["market_prob", "actual_outcome", probability_column],
    )
    predicted_yes = predictions[probability_column].to_numpy(dtype=np.float64, copy=True) >= float(threshold)
    price_cents = (
        predictions["market_prob"].to_numpy(dtype=np.float64, copy=True) * 100.0
    ).round().astype(np.int64)
    price_cents = np.clip(price_cents, 0, 100)
    actual_outcome = predictions["actual_outcome"].to_numpy(dtype=np.int8, copy=True)

    bucket_start = (price_cents // 10) * 10
    bucket_start = np.minimum(bucket_start, 90)
    bucket_labels = [f"{start:02d}-{start + 10:02d}c" for start in range(0, 100, 10)]

    lines = ["| Bucket | Test Rows | Predicted YES | Actual YES Within Predicted YES | YES Precision |", "| --- | --- | --- | --- | --- |"]

    for start, label in zip(range(0, 100, 10), bucket_labels, strict=True):
        in_bucket = bucket_start == start
        bucket_rows = int(np.count_nonzero(in_bucket))
        bucket_pred_yes = predicted_yes[in_bucket]
        predicted_yes_count = int(np.count_nonzero(bucket_pred_yes))
        bucket_actual_yes = actual_outcome[in_bucket][bucket_pred_yes]
        actual_yes_count = int(np.count_nonzero(bucket_actual_yes == 1))
        yes_precision = _safe_divide(actual_yes_count, predicted_yes_count)
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{bucket_rows:,}",
                    f"{predicted_yes_count:,}",
                    f"{actual_yes_count:,}",
                    _format_pct(yes_precision),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _hourly_series_token(report: RunReport) -> str | None:
    context = report.context_label.lower()
    if not context.endswith(" hourly context"):
        return None
    return context.removesuffix(" hourly context").strip() or None


def _hourly_base_stem(report: RunReport) -> str:
    hourly_series = _hourly_series_token(report)
    if hourly_series:
        pattern = rf"_{re.escape(hourly_series)}_hourly_ctx_v\d+$"
        stem, count = re.subn(pattern, "", report.run_name)
        if count:
            return stem
    stem, count = re.subn(r"_hourly_ctx_v\d+$", "", report.run_name)
    if count:
        return stem
    return report.run_name


def _run_name_similarity(candidate_run_name: str, target_stem: str) -> tuple[int, int, int]:
    candidate_tokens = set(candidate_run_name.split("_"))
    target_tokens = set(target_stem.split("_"))
    token_overlap = len(candidate_tokens & target_tokens)
    starts_with_stem = 1 if candidate_run_name.startswith(target_stem) else 0
    return (starts_with_stem, token_overlap, -len(candidate_run_name))


def _paired_base_and_hourly_reports(reports: list[RunReport]) -> list[tuple[RunReport, RunReport]]:
    pairs: list[tuple[RunReport, RunReport]] = []
    hourly_reports = [report for report in reports if "hourly context" in report.context_label.lower()]
    for hourly_report in sorted(hourly_reports, key=lambda row: (row.model_family, row.run_name)):
        candidates = [
            report
            for report in reports
            if report.model_family == hourly_report.model_family and report.context_label == "base 15m features"
        ]
        if not candidates:
            continue
        target_stem = _hourly_base_stem(hourly_report)
        best_candidate = max(candidates, key=lambda candidate: _run_name_similarity(candidate.run_name, target_stem))
        pairs.append((best_candidate, hourly_report))
    return pairs


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return float(current) - float(baseline)


def _base_vs_hourly_delta_section(reports: list[RunReport]) -> str:
    pairs = _paired_base_and_hourly_reports(reports)
    if not pairs:
        return ""

    prediction_lines = [
        "## Base vs Hourly Delta Summary",
        "",
        "### Pairing Rule",
        "",
        "- Each hourly-context run is paired to the base run in the same model family whose run name most closely matches the hourly run name after removing the hourly-context suffix.",
        "- Deltas are calculated as `hourly - base`.",
        "- For `Log-loss`, a negative delta is better.",
        "- For `Accuracy`, `YES Precision`, and `Trade Hit Rate`, a positive delta is better.",
        "- For `Net PnL`, a positive delta is better.",
        "",
        "### Prediction Deltas",
        "",
        "| Family | Base Run | Hourly Run | Base Log-loss | Hourly Log-loss | Δ Log-loss | Base Accuracy | Hourly Accuracy | Δ Accuracy | Base YES Precision | Hourly YES Precision | Δ YES Precision |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    trade_lines = [
        "### Trade Deltas",
        "",
        "| Family | Base Run | Hourly Run | Base Net PnL | Hourly Net PnL | Δ Net PnL | Base Trade Hit Rate | Hourly Trade Hit Rate | Δ Trade Hit Rate | Base Settled | Hourly Settled | Δ Settled |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for base_report, hourly_report in pairs:
        base_pred = base_report.prediction_metrics
        hourly_pred = hourly_report.prediction_metrics
        prediction_lines.append(
            "| "
            + " | ".join(
                [
                    hourly_report.model_family,
                    base_report.run_name,
                    hourly_report.run_name,
                    _format_float(base_pred.test_log_loss, digits=6),
                    _format_float(hourly_pred.test_log_loss, digits=6),
                    _format_signed_float(_delta(hourly_pred.test_log_loss, base_pred.test_log_loss), digits=6),
                    _format_pct(base_pred.accuracy),
                    _format_pct(hourly_pred.accuracy),
                    _format_delta_pct_points(_delta(hourly_pred.accuracy, base_pred.accuracy)),
                    _format_pct(base_pred.yes_precision),
                    _format_pct(hourly_pred.yes_precision),
                    _format_delta_pct_points(_delta(hourly_pred.yes_precision, base_pred.yes_precision)),
                ]
            )
            + " |"
        )

        base_trade = base_report.trade_metrics
        hourly_trade = hourly_report.trade_metrics
        trade_lines.append(
            "| "
            + " | ".join(
                [
                    hourly_report.model_family,
                    base_report.run_name,
                    hourly_report.run_name,
                    _format_money(base_trade.net_pnl_dollars),
                    _format_money(hourly_trade.net_pnl_dollars),
                    _format_signed_money(_delta(hourly_trade.net_pnl_dollars, base_trade.net_pnl_dollars)),
                    _format_pct(base_trade.trade_hit_rate),
                    _format_pct(hourly_trade.trade_hit_rate),
                    _format_delta_pct_points(_delta(hourly_trade.trade_hit_rate, base_trade.trade_hit_rate)),
                    str(base_trade.settled_trade_count),
                    str(hourly_trade.settled_trade_count),
                    _format_signed_int(hourly_trade.settled_trade_count - base_trade.settled_trade_count),
                ]
            )
            + " |"
        )
    return "\n".join(prediction_lines + [""] + trade_lines)


def _all_models_bucket_report(reports: list[RunReport], *, threshold: float) -> str:
    lines = [
        "## YES Precision By 10c Contract Bucket",
        "",
        "### How This Was Calculated",
        "",
        "- Bucket price = test-time market YES price, computed as `market_prob * 100` and rounded to cents.",
        "- Bucket assignment = floor that rounded price into 10-cent buckets: `00-10c`, `10-20c`, ..., `90-100c`.",
        "- A row counts as a model `YES` prediction when the run's probability column meets the classification threshold.",
        f"- YES prediction rule = `probability >= {threshold:.2f}` using `calibrated_probability` when available, otherwise `raw_probability`.",
        "- YES precision in each bucket = `actual YES within predicted YES / predicted YES count in that bucket`.",
        "- `Test Rows` = all prediction rows that landed in that price bucket, even if the model did not predict YES there.",
        "",
    ]
    for report in sorted(reports, key=lambda row: (row.model_family, row.context_label, row.run_name)):
        lines.extend(
            [
                f"### {report.run_name}",
                "",
                f"- Family: `{report.model_family}`",
                f"- Context: `{report.context_label}`",
                f"- Probability column: `{report.prediction_metrics.probability_column}`",
                "",
                _yes_precision_bucket_table(
                    report.run_dir,
                    probability_column=report.prediction_metrics.probability_column,
                    threshold=threshold,
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _explanation_section(threshold: float, probability_column: str) -> str:
    return "\n".join(
        [
            "## Metric Explanations",
            "",
            f"- **Probability column used**: `{probability_column}`. The script uses calibrated probabilities when available, otherwise raw probabilities.",
            f"- **Overall accuracy**: We call the model `YES` when probability is at least `{threshold:.2f}` and `NO` otherwise. Accuracy tells us how often that simple YES/NO call matched reality.",
            "- **YES precision**: Out of all rows where the model predicted `YES`, how many actually resolved `YES`.",
            "- **YES recall**: Out of all markets that really resolved `YES`, how many the model successfully identified as `YES`.",
            "- **NO precision**: Out of all rows where the model predicted `NO`, how many actually resolved `NO`.",
            "- **Predicted YES rate**: How often the model leans `YES` at the chosen threshold.",
            "- **Actual YES rate**: How often the test set truly resolved `YES`. This is useful context for interpreting accuracy.",
            "- **Log-loss**: A probability-quality metric. Lower is better. It rewards well-calibrated confidence and punishes confident mistakes.",
            "- **Brier score**: Another probability-quality metric. Lower is better. It measures how far the predicted probability was from the actual outcome on average.",
            "- **Trade hit rate**: Out of the trades the policy actually took, how often those trades settled as wins.",
            "- **YES hit rate / NO hit rate**: The trade win rate broken out by side, so we can see whether a model is stronger on YES trades or NO trades.",
            "",
            "### How To Read This In Simple Terms",
            "",
            "- If you want to know **\"when the model says YES, is it usually right?\"**, look at **YES precision**.",
            "- If you want to know **\"how often does the model get the final direction right overall?\"**, look at **Overall accuracy**.",
            "- If you want to know **\"how good are the probabilities themselves, not just the yes/no labels?\"**, look at **Log-loss** and **Brier score**.",
            "- If you want to know **\"did the actual trading policy make correct bets?\"**, look at **Trade hit rate**.",
        ]
    )


def _build_report(reports: list[RunReport], *, artifacts_root: Path, output_path: Path, threshold: float) -> str:
    if not reports:
        raise RuntimeError(f"No completed runs with test_predictions.parquet were found under {artifacts_root}")

    example_probability_column = reports[0].prediction_metrics.probability_column
    lines = [
        "# KXBTC15M Model Metrics Report",
        "",
        f"- Generated at: `{datetime.now(UTC).isoformat()}`",
        f"- Artifacts root: `{artifacts_root}`",
        f"- Output file: `{output_path}`",
        f"- Included runs: `{len(reports)}`",
        f"- Classification threshold: `{threshold:.2f}`",
        "",
        "## Prediction Metrics",
        "",
        _prediction_metrics_table(reports),
        "",
        "## Trade Metrics",
        "",
        _trade_metrics_table(reports),
        "",
        _base_vs_hourly_delta_section(reports),
        "",
        _artifact_paths_section(reports),
        "",
        _all_models_bucket_report(reports, threshold=threshold),
        "",
        _explanation_section(threshold, example_probability_column),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute KXBTC15M classification and trade metrics across trained model runs and write a Markdown report."
    )
    parser.add_argument(
        "--artifacts-root",
        default=str(DEFAULT_ARTIFACTS_ROOT),
        help="Root folder containing model run artifacts. Defaults to artifacts/kalshi.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Optional specific run directory to include. Pass multiple times to restrict the report.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Markdown output path. Defaults to artifacts/kalshi/kxbtc15m_model_metrics_report.md.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for YES vs NO metrics. Defaults to 0.5.",
    )
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root).expanduser()
    output_path = Path(args.output).expanduser()
    run_dirs = [Path(value).expanduser() for value in args.run_dir] if args.run_dir else _discover_run_dirs(artifacts_root)
    reports = [_load_run_report(run_dir, threshold=float(args.threshold)) for run_dir in run_dirs]

    markdown = _build_report(reports, artifacts_root=artifacts_root, output_path=output_path, threshold=float(args.threshold))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote report to {output_path}")
    print(f"Included {len(reports)} runs.")


if __name__ == "__main__":
    main()
