from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.live.kalshi.bucket_policy import (
    label_edge_bucket,
    label_price_bucket,
    label_probability_bucket,
    label_tau_bucket,
)

DEFAULT_REPORTS_ROOT = Path("artifacts") / "kalshi" / "performance_reports"
DEFAULT_MIN_COMBO_COUNT = 5

TAU_BUCKET_ORDER = (
    "0-2",
    "2-4",
    "4-6",
    "6-8",
    "8-10",
    "10-12",
    "12-14",
    "14-16",
    "16+",
    "unknown",
)
PRICE_BUCKET_ORDER = tuple(f"{value}-{value + 10}" for value in range(0, 100, 10)) + ("unknown",)
PROBABILITY_BUCKET_ORDER = tuple(f"{value}-{value + 10}" for value in range(0, 100, 10)) + ("unknown",)
EDGE_BUCKET_ORDER = ("<0", "0-5", "5-10", "10-20", "20-40", "40-60", "60+", "unknown")
REGIME_ORDER = ("downtrend", "neutral", "uptrend", "unknown")
SPREAD_BUCKET_ORDER = ("0-2", "2-5", "5-10", "10-20", "20+", "unknown")
AGE_BUCKET_ORDER = ("0-1s", "1-3s", "3-10s", "10s+", "unknown")
BOOL_BUCKET_ORDER = ("true", "false", "unknown")

KNOWN_OFFLINE_FILENAMES = {
    "summary.json",
    "test_metrics.json",
    "validation_metrics.json",
    "test_diagnostics.json",
    "validation_diagnostics.json",
    "test_trade_records.parquet",
    "validation_trade_records.parquet",
    "test_predictions.parquet",
}
KNOWN_EVENT_FILENAMES = {"events.jsonl"}


@dataclass(frozen=True)
class AnalysisTarget:
    source_type: str
    run_dir: Path
    run_name: str


@dataclass(frozen=True)
class TargetDetection:
    detected_mode: str
    detected_name: str
    targets: list[AnalysisTarget]


@dataclass
class SourceLoadResult:
    source_type: str
    run_dir: Path
    run_name: str
    canonical_rows: list[dict[str, Any]]
    skipped_lines_by_file: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_totals_override: list[dict[str, Any]] = field(default_factory=list)
    prediction_metrics_rows: list[dict[str, Any]] = field(default_factory=list)
    trade_metrics_rows: list[dict[str, Any]] = field(default_factory=list)
    calibration_summary_rows: list[dict[str, Any]] = field(default_factory=list)
    skip_reason_rows: list[dict[str, Any]] = field(default_factory=list)
    quote_quality_rows: list[dict[str, Any]] = field(default_factory=list)
    thesis_summary_rows: list[dict[str, Any]] = field(default_factory=list)
    metadata_extras: dict[str, Any] = field(default_factory=dict)


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _timestamp_iso(value: Any) -> str | None:
    parsed = _parse_timestamp(value)
    return None if parsed is None else parsed.isoformat()


def _slugify_path_name(path: Path) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in path.name)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug or "kalshi-performance"


def _probability_bucket_label(probability: float | None) -> str:
    return label_probability_bucket(probability)


def _price_bucket_label(price_cents: int | None) -> str:
    return label_price_bucket(price_cents)


def _edge_bucket_label(edge_cents: float | None) -> str:
    return label_edge_bucket(edge_cents)


def _tau_bucket_label(tau_minutes: float | None) -> str:
    if tau_minutes is None:
        return "unknown"
    if tau_minutes < 0.0:
        return "unknown"
    if tau_minutes < 2.0:
        return "0-2"
    if tau_minutes < 4.0:
        return "2-4"
    if tau_minutes < 6.0:
        return "4-6"
    if tau_minutes < 8.0:
        return "6-8"
    if tau_minutes < 10.0:
        return "8-10"
    if tau_minutes < 12.0:
        return "10-12"
    if tau_minutes < 14.0:
        return "12-14"
    if tau_minutes < 16.0:
        return "14-16"
    return "16+"


def _spread_bucket_label(spread_cents: int | None) -> str:
    if spread_cents is None:
        return "unknown"
    if spread_cents < 2:
        return "0-2"
    if spread_cents < 5:
        return "2-5"
    if spread_cents < 10:
        return "5-10"
    if spread_cents < 20:
        return "10-20"
    return "20+"


def _age_bucket_label(age_seconds: float | None) -> str:
    if age_seconds is None:
        return "unknown"
    if age_seconds < 1.0:
        return "0-1s"
    if age_seconds < 3.0:
        return "1-3s"
    if age_seconds < 10.0:
        return "3-10s"
    return "10s+"


def _bool_bucket(value: Any) -> str:
    if value is None:
        return "unknown"
    return "true" if bool(value) else "false"


def _iter_jsonl_tolerant(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    if not path.exists():
        return rows, skipped
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
            else:
                skipped += 1
    return rows, skipped


def _format_currency(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.2f}%"


def _format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def _is_live_execution_run(path: Path) -> bool:
    return path.is_dir() and (path / "signal").exists() and (path / "execution").exists()


def _is_live_research_run(path: Path) -> bool:
    return path.is_dir() and (path / "research").exists()


def _is_offline_artifact_run(path: Path) -> bool:
    return path.is_dir() and any((path / filename).exists() for filename in KNOWN_OFFLINE_FILENAMES)


def _resolve_live_execution_run(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if _is_live_execution_run(candidate):
            return candidate
    return None


def _resolve_live_research_run(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if _is_live_research_run(candidate):
            return candidate
    return None


def _resolve_offline_artifact_run(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if _is_offline_artifact_run(candidate):
            return candidate
    return None


def _discover_live_execution_targets(path: Path) -> list[AnalysisTarget]:
    resolved = _resolve_live_execution_run(path)
    if resolved is not None:
        return [AnalysisTarget(source_type="live_execution", run_dir=resolved, run_name=resolved.name)]
    if path.is_dir():
        targets = [
            AnalysisTarget(source_type="live_execution", run_dir=child, run_name=child.name)
            for child in sorted(path.iterdir())
            if _is_live_execution_run(child)
        ]
        if targets:
            return targets
    return []


def _discover_live_research_targets(path: Path) -> list[AnalysisTarget]:
    resolved = _resolve_live_research_run(path)
    if resolved is not None:
        return [AnalysisTarget(source_type="live_research", run_dir=resolved, run_name=resolved.name)]
    if path.is_dir():
        targets = [
            AnalysisTarget(source_type="live_research", run_dir=child, run_name=child.name)
            for child in sorted(path.iterdir())
            if _is_live_research_run(child)
        ]
        if targets:
            return targets
    return []


def _discover_offline_artifact_targets(path: Path) -> list[AnalysisTarget]:
    resolved = _resolve_offline_artifact_run(path)
    if resolved is not None:
        return [AnalysisTarget(source_type="offline_artifacts", run_dir=resolved, run_name=resolved.name)]
    if not path.is_dir():
        return []
    direct_targets = [
        AnalysisTarget(source_type="offline_artifacts", run_dir=child, run_name=child.name)
        for child in sorted(path.iterdir())
        if _is_offline_artifact_run(child)
    ]
    if direct_targets:
        return direct_targets
    found_runs: dict[Path, AnalysisTarget] = {}
    for summary_path in sorted(path.rglob("summary.json")):
        run_dir = summary_path.parent
        if _is_offline_artifact_run(run_dir):
            found_runs[run_dir] = AnalysisTarget(
                source_type="offline_artifacts",
                run_dir=run_dir,
                run_name=run_dir.name,
            )
    return sorted(found_runs.values(), key=lambda item: str(item.run_dir))


def detect_analysis_targets(input_path: str | Path, mode: str = "auto") -> TargetDetection:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if mode not in {"auto", "offline_artifacts", "live_execution", "live_research"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode == "live_execution":
        targets = _discover_live_execution_targets(path)
    elif mode == "live_research":
        targets = _discover_live_research_targets(path)
    elif mode == "offline_artifacts":
        targets = _discover_offline_artifact_targets(path)
    else:
        targets = []
        for candidate_mode, discover in (
            ("live_research", _discover_live_research_targets),
            ("live_execution", _discover_live_execution_targets),
            ("offline_artifacts", _discover_offline_artifact_targets),
        ):
            targets = discover(path)
            if targets:
                mode = candidate_mode
                break
    if not targets:
        raise ValueError(f"Could not detect a supported Kalshi performance source from: {path}")

    detected_name = targets[0].run_name if len(targets) == 1 else _slugify_path_name(path)
    return TargetDetection(detected_mode=mode, detected_name=detected_name, targets=targets)


def _filter_by_lookback(rows: list[dict[str, Any]], lookback_days: int | None) -> list[dict[str, Any]]:
    if lookback_days is None:
        return rows
    latest: datetime | None = None
    for row in rows:
        for key in ("settled_at", "recorded_at"):
            value = _parse_timestamp(row.get(key))
            if value is not None and (latest is None or value > latest):
                latest = value
    if latest is None:
        return rows
    cutoff = latest - timedelta(days=lookback_days)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        row_time = _parse_timestamp(row.get("settled_at")) or _parse_timestamp(row.get("recorded_at"))
        if row_time is None or row_time >= cutoff:
            filtered.append(row)
    return filtered


def _apply_model_filters(rows: list[dict[str, Any]], model_filters: tuple[str, ...]) -> list[dict[str, Any]]:
    if not model_filters:
        return rows
    normalized = {value.lower() for value in model_filters}
    return [row for row in rows if str(row.get("model", "")).lower() in normalized]


def _same_ticker_side_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_model: dict[str, int] = {}
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        ticker = _clean_string(row.get("ticker"))
        model = _clean_string(row.get("model"))
        side = _clean_string(row.get("side"))
        if not ticker or not model or not side:
            continue
        grouped[(model, ticker)].add(side)
    for model_ticker, sides in grouped.items():
        if len(sides) > 1:
            per_model[model_ticker[0]] = per_model.get(model_ticker[0], 0) + 1
    return {
        "tickers_with_both_sides_by_model": per_model,
        "tickers_with_both_sides_total": sum(per_model.values()),
    }


def _time_lag_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lags_minutes: list[float] = []
    for row in rows:
        recorded_at = _parse_timestamp(row.get("recorded_at"))
        settled_at = _parse_timestamp(row.get("settled_at"))
        if recorded_at is None or settled_at is None:
            continue
        lags_minutes.append((settled_at - recorded_at).total_seconds() / 60.0)
    if not lags_minutes:
        return {}
    ordered = sorted(lags_minutes)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 == 1 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "settlement_lag_avg_minutes": sum(ordered) / len(ordered),
        "settlement_lag_median_minutes": median,
        "settlement_lag_max_minutes": max(ordered),
    }


def _load_live_execution_run(run_dir: Path, environment: str | None = None) -> SourceLoadResult:
    approvals: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    settlements: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    skipped_lines_by_file: dict[str, int] = {}
    skip_reason_counter: Counter[tuple[str, str, str | None, str | None, str | None]] = Counter()
    approval_counts: Counter[str] = Counter()
    quoted_rows: list[dict[str, Any]] = []

    signal_root = run_dir / "signal"
    execution_root = run_dir / "execution"
    models = sorted(
        {path.name for path in signal_root.iterdir() if path.is_dir()}
        & {path.name for path in execution_root.iterdir() if path.is_dir()}
    )

    for model in models:
        model_signal_root = signal_root / model
        env_paths = [model_signal_root / environment] if environment else [path for path in model_signal_root.iterdir() if path.is_dir()]
        for env_path in sorted(env_paths):
            for events_path in sorted(env_path.rglob("events.jsonl")):
                rows, skipped = _iter_jsonl_tolerant(events_path)
                if skipped:
                    skipped_lines_by_file[str(events_path)] = skipped_lines_by_file.get(str(events_path), 0) + skipped
                for event in rows:
                    if event.get("event_type") != "signal_decision":
                        continue
                    payload = event.get("payload", {})
                    decision_id = _clean_string(payload.get("decision_id"))
                    if decision_id is None:
                        continue
                    if not payload.get("approved"):
                        reason = _clean_string(payload.get("block_reason")) or "unknown"
                        skip_reason_counter[(
                            model,
                            reason,
                            _clean_string(payload.get("bucket_policy_side")),
                            _clean_string(payload.get("bucket_policy_dimension")),
                            _clean_string(payload.get("bucket_policy_bucket")),
                        )] += 1
                        continue
                    approval_counts[model] += 1
                    approvals[(model, decision_id)] = (_timestamp_iso(event.get("logged_at")) or "", payload)

    for model in models:
        model_execution_root = execution_root / model
        env_paths = (
            [model_execution_root / environment]
            if environment
            else [path for path in model_execution_root.iterdir() if path.is_dir()]
        )
        for env_path in sorted(env_paths):
            for events_path in sorted(env_path.rglob("events.jsonl")):
                rows, skipped = _iter_jsonl_tolerant(events_path)
                if skipped:
                    skipped_lines_by_file[str(events_path)] = skipped_lines_by_file.get(str(events_path), 0) + skipped
                for event in rows:
                    if event.get("event_type") != "simulated_position_settled":
                        continue
                    payload = event.get("payload", {})
                    decision_id = _clean_string(payload.get("decision_id"))
                    if decision_id is None:
                        continue
                    settlements[(model, decision_id)] = (_timestamp_iso(event.get("logged_at")) or "", payload)

    canonical_rows: list[dict[str, Any]] = []
    for (model, decision_id), (approved_at, payload) in sorted(approvals.items()):
        settlement = settlements.get((model, decision_id))
        settled_payload = None if settlement is None else settlement[1]
        settled_at = None if settlement is None else settlement[0]
        side = _clean_string((settled_payload or {}).get("side")) or _clean_string(payload.get("side"))
        predicted_yes_probability = _safe_float(payload.get("predicted_yes_probability"))
        feature_basis_market_prob = _safe_float(payload.get("feature_basis_market_prob"))
        chosen_probability = None
        if side == "YES":
            chosen_probability = predicted_yes_probability
        elif side == "NO" and predicted_yes_probability is not None:
            chosen_probability = 1.0 - predicted_yes_probability
        yes_edge = _safe_float(payload.get("yes_post_cost_edge"))
        no_edge = _safe_float(payload.get("no_post_cost_edge"))
        chosen_edge = yes_edge if side == "YES" else no_edge if side == "NO" else None
        reference_price_cents = _safe_int(payload.get("reference_price_cents"))
        settlement_result = _clean_string((settled_payload or {}).get("settlement_result"))
        realized_pnl = _safe_float((settled_payload or {}).get("realized_pnl_dollars"))
        cumulative_pnl = _safe_float((settled_payload or {}).get("cumulative_realized_pnl_dollars"))
        thesis_id = _clean_string((settled_payload or {}).get("thesis_id")) or _clean_string(payload.get("thesis_id"))
        offline_rule_side = None
        same_as_offline_rule = None
        if predicted_yes_probability is not None and feature_basis_market_prob is not None:
            offline_rule_side = "YES" if predicted_yes_probability > feature_basis_market_prob else "NO"
            if side is not None:
                same_as_offline_rule = side == offline_rule_side
        yes_bid_cents = _safe_int(payload.get("yes_bid_cents"))
        yes_ask_cents = _safe_int(payload.get("yes_ask_cents"))
        one_sided_quote = (yes_bid_cents == 0) or (yes_ask_cents == 100) if yes_bid_cents is not None and yes_ask_cents is not None else None

        row = {
            "source_type": "live_execution",
            "run_name": run_dir.name,
            "model": model,
            "ticker": _clean_string((settled_payload or {}).get("ticker")) or _clean_string(payload.get("ticker")),
            "decision_id": decision_id,
            "thesis_id": thesis_id,
            "sample_id": None,
            "side": side,
            "recorded_at": approved_at,
            "settled_at": settled_at,
            "status": "settled" if settled_payload is not None else "open",
            "settlement_result": settlement_result,
            "is_win": None if settled_payload is None else bool(realized_pnl is not None and realized_pnl > 0.0),
            "realized_pnl_dollars": realized_pnl,
            "cumulative_realized_pnl_dollars": cumulative_pnl,
            "price_bucket": _clean_string(payload.get("price_bucket")) or _price_bucket_label(reference_price_cents),
            "probability_bucket": _clean_string(payload.get("chosen_side_probability_bucket")) or _probability_bucket_label(chosen_probability),
            "edge_bucket": _clean_string(payload.get("chosen_side_edge_bucket")) or _edge_bucket_label(None if chosen_edge is None else chosen_edge * 100.0),
            "tau_bucket": _clean_string(payload.get("tau_bucket")) or _tau_bucket_label(_safe_float(payload.get("tau_minutes"))),
            "reference_price_cents": reference_price_cents,
            "chosen_side_probability": chosen_probability,
            "chosen_post_cost_edge_cents": None if chosen_edge is None else chosen_edge * 100.0,
            "tau_minutes": _safe_float(payload.get("tau_minutes")),
            "cash_required_dollars": _safe_float((settled_payload or {}).get("cash_required_dollars")),
            "quote_spread_cents": _safe_int(payload.get("quote_spread_cents")),
            "quote_spread_bucket": _spread_bucket_label(_safe_int(payload.get("quote_spread_cents"))),
            "quote_age_seconds": _safe_float(payload.get("quote_age_seconds")),
            "quote_age_bucket": _age_bucket_label(_safe_float(payload.get("quote_age_seconds"))),
            "quote_mid_prob": _safe_float(payload.get("quote_mid_prob")),
            "buy_yes_price_cents": _safe_int(payload.get("buy_yes_price_cents")),
            "buy_no_price_cents": _safe_int(payload.get("buy_no_price_cents")),
            "predicted_yes_probability": predicted_yes_probability,
            "predicted_no_probability": _safe_float(payload.get("predicted_no_probability")),
            "feature_basis_market_prob": feature_basis_market_prob,
            "raw_model_edge": _safe_float(payload.get("raw_model_edge")),
            "yes_post_cost_edge_cents": None if yes_edge is None else yes_edge * 100.0,
            "no_post_cost_edge_cents": None if no_edge is None else no_edge * 100.0,
            "regime_label": _clean_string(payload.get("regime_label")) or "unknown",
            "bearish_vote_count": _safe_int(payload.get("bearish_vote_count")),
            "bullish_vote_count": _safe_int(payload.get("bullish_vote_count")),
            "regime_price_momentum_bearish": payload.get("regime_price_momentum_bearish"),
            "regime_signed_flow_bearish": payload.get("regime_signed_flow_bearish"),
            "regime_yes_share_bearish": payload.get("regime_yes_share_bearish"),
            "regime_price_momentum_bullish": payload.get("regime_price_momentum_bullish"),
            "regime_signed_flow_bullish": payload.get("regime_signed_flow_bullish"),
            "regime_yes_share_bullish": payload.get("regime_yes_share_bullish"),
            "bucket_policy_side": _clean_string(payload.get("bucket_policy_side")),
            "bucket_policy_dimension": _clean_string(payload.get("bucket_policy_dimension")),
            "bucket_policy_bucket": _clean_string(payload.get("bucket_policy_bucket")),
            "tranche_index": _safe_int((settled_payload or {}).get("tranche_index"))
            if (settled_payload or {}).get("tranche_index") is not None
            else _safe_int(payload.get("tranche_index")),
            "tranche_window": _clean_string((settled_payload or {}).get("tranche_window"))
            or _clean_string(payload.get("tranche_window")),
            "tranche_reason": _clean_string((settled_payload or {}).get("tranche_reason"))
            or _clean_string(payload.get("tranche_reason")),
            "lifecycle_state": _clean_string((settled_payload or {}).get("lifecycle_state"))
            or _clean_string(payload.get("lifecycle_state")),
            "total_thesis_budget_dollars": _safe_float((settled_payload or {}).get("total_thesis_budget_dollars"))
            if (settled_payload or {}).get("total_thesis_budget_dollars") is not None
            else _safe_float(payload.get("total_thesis_budget_dollars")),
            "payout_if_yes_dollars": _safe_float((settled_payload or {}).get("payout_if_yes_dollars"))
            if (settled_payload or {}).get("payout_if_yes_dollars") is not None
            else _safe_float(payload.get("payout_if_yes_dollars")),
            "payout_if_no_dollars": _safe_float((settled_payload or {}).get("payout_if_no_dollars"))
            if (settled_payload or {}).get("payout_if_no_dollars") is not None
            else _safe_float(payload.get("payout_if_no_dollars")),
            "expected_value_dollars": _safe_float((settled_payload or {}).get("expected_value_dollars"))
            if (settled_payload or {}).get("expected_value_dollars") is not None
            else _safe_float(payload.get("expected_value_dollars")),
            "worst_case_loss_dollars": _safe_float((settled_payload or {}).get("worst_case_loss_dollars"))
            if (settled_payload or {}).get("worst_case_loss_dollars") is not None
            else _safe_float(payload.get("worst_case_loss_dollars")),
            "offline_rule_side": offline_rule_side,
            "same_as_offline_rule": same_as_offline_rule,
            "one_sided_quote": one_sided_quote,
        }
        canonical_rows.append(row)
        quoted_rows.append(row)

    skip_reason_rows = [
        {
            "model": model,
            "reason": reason,
            "count": count,
            "source_type": "live_execution",
            "bucket_policy_side": bucket_policy_side,
            "bucket_policy_dimension": bucket_policy_dimension,
            "bucket_policy_bucket": bucket_policy_bucket,
        }
        for (model, reason, bucket_policy_side, bucket_policy_dimension, bucket_policy_bucket), count in sorted(skip_reason_counter.items())
    ]

    metadata_extras = {
        "approval_counts_by_model": dict(approval_counts),
        "settlement_counts_by_model": dict(
            Counter(row["model"] for row in canonical_rows if row.get("status") == "settled")
        ),
    }
    metadata_extras.update(_same_ticker_side_stats(canonical_rows))
    metadata_extras.update(_time_lag_summary([row for row in canonical_rows if row.get("status") == "settled"]))

    quote_quality_rows = _quote_quality_summary_rows(canonical_rows)
    thesis_summary_rows = _thesis_summary_rows(canonical_rows)
    return SourceLoadResult(
        source_type="live_execution",
        run_dir=run_dir,
        run_name=run_dir.name,
        canonical_rows=canonical_rows,
        skipped_lines_by_file=skipped_lines_by_file,
        skip_reason_rows=skip_reason_rows,
        quote_quality_rows=quote_quality_rows,
        thesis_summary_rows=thesis_summary_rows,
        metadata_extras=metadata_extras,
    )


def _load_live_research_run(run_dir: Path, environment: str | None = None) -> SourceLoadResult:
    recorded: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    settled: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    skipped_lines_by_file: dict[str, int] = {}
    skip_reason_counter: Counter[tuple[str, str, str | None, str | None, str | None]] = Counter()
    summary_snapshots: dict[str, dict[str, Any]] = {}

    research_root = run_dir / "research"
    models = sorted(path.name for path in research_root.iterdir() if path.is_dir())
    for model in models:
        model_root = research_root / model
        env_paths = [model_root / environment] if environment else [path for path in model_root.iterdir() if path.is_dir()]
        for env_path in sorted(env_paths):
            for events_path in sorted(env_path.rglob("events.jsonl")):
                rows, skipped = _iter_jsonl_tolerant(events_path)
                if skipped:
                    skipped_lines_by_file[str(events_path)] = skipped_lines_by_file.get(str(events_path), 0) + skipped
                for event in rows:
                    event_type = event.get("event_type")
                    payload = event.get("payload", {})
                    if event_type == "research_sample_recorded":
                        sample_id = _clean_string(payload.get("sample_id"))
                        if sample_id:
                            recorded[(model, sample_id)] = (_timestamp_iso(event.get("logged_at")) or "", payload)
                    elif event_type == "research_sample_settled":
                        sample_id = _clean_string(payload.get("sample_id"))
                        if sample_id:
                            settled[(model, sample_id)] = (_timestamp_iso(event.get("logged_at")) or "", payload)
                    elif event_type == "research_sample_skipped":
                        reason = _clean_string(payload.get("reason")) or "unknown"
                        skip_reason_counter[(
                            model,
                            reason,
                            _clean_string(payload.get("bucket_policy_side")),
                            _clean_string(payload.get("bucket_policy_dimension")),
                            _clean_string(payload.get("bucket_policy_bucket")),
                        )] += 1
                    elif event_type == "research_summary_snapshot":
                        summary_snapshots[model] = payload

    canonical_rows: list[dict[str, Any]] = []
    for (model, sample_id), (recorded_at, payload) in sorted(recorded.items()):
        settlement = settled.get((model, sample_id))
        settled_payload = None if settlement is None else settlement[1]
        settled_at = None if settlement is None else settlement[0]
        chosen_probability = _safe_float(payload.get("chosen_side_probability"))
        chosen_edge = _safe_float(payload.get("chosen_post_cost_edge"))
        row = {
            "source_type": "live_research",
            "run_name": run_dir.name,
            "model": model,
            "ticker": _clean_string(payload.get("ticker")),
            "decision_id": None,
            "sample_id": sample_id,
            "side": _clean_string(payload.get("side")),
            "recorded_at": recorded_at,
            "settled_at": settled_at,
            "status": "settled" if settled_payload is not None else "open",
            "settlement_result": _clean_string((settled_payload or {}).get("settlement_result")),
            "is_win": None if settled_payload is None else bool((settled_payload or {}).get("is_win")),
            "realized_pnl_dollars": _safe_float((settled_payload or {}).get("realized_pnl_dollars")),
            "cumulative_realized_pnl_dollars": _safe_float((settled_payload or {}).get("cumulative_realized_pnl_dollars")),
            "price_bucket": _clean_string(payload.get("price_bucket")) or _price_bucket_label(_safe_int(payload.get("reference_price_cents"))),
            "probability_bucket": _clean_string(payload.get("chosen_side_probability_bucket")) or _probability_bucket_label(chosen_probability),
            "edge_bucket": _clean_string(payload.get("chosen_side_edge_bucket")) or _edge_bucket_label(None if chosen_edge is None else chosen_edge * 100.0),
            "tau_bucket": _clean_string(payload.get("tau_bucket")) or _tau_bucket_label(_safe_float(payload.get("tau_minutes"))),
            "reference_price_cents": _safe_int(payload.get("reference_price_cents")),
            "chosen_side_probability": chosen_probability,
            "chosen_post_cost_edge_cents": None if chosen_edge is None else chosen_edge * 100.0,
            "tau_minutes": _safe_float(payload.get("tau_minutes")),
            "cash_required_dollars": _safe_float((settled_payload or {}).get("cash_required_dollars"))
            or _safe_float(payload.get("estimated_cash_required_dollars")),
            "quote_spread_cents": _safe_int(payload.get("quote_spread_cents")),
            "quote_spread_bucket": _spread_bucket_label(_safe_int(payload.get("quote_spread_cents"))),
            "quote_age_seconds": _safe_float(payload.get("quote_age_seconds")),
            "quote_age_bucket": _age_bucket_label(_safe_float(payload.get("quote_age_seconds"))),
            "quote_mid_prob": _safe_float(payload.get("quote_mid_prob")),
            "buy_yes_price_cents": _safe_int(payload.get("buy_yes_price_cents")),
            "buy_no_price_cents": _safe_int(payload.get("buy_no_price_cents")),
            "predicted_yes_probability": _safe_float(payload.get("predicted_yes_probability")),
            "predicted_no_probability": _safe_float(payload.get("predicted_no_probability")),
            "feature_basis_market_prob": _safe_float(payload.get("feature_basis_market_prob")),
            "raw_model_edge": _safe_float(payload.get("raw_model_edge")),
            "yes_post_cost_edge_cents": None
            if _safe_float(payload.get("yes_post_cost_edge")) is None
            else _safe_float(payload.get("yes_post_cost_edge")) * 100.0,
            "no_post_cost_edge_cents": None
            if _safe_float(payload.get("no_post_cost_edge")) is None
            else _safe_float(payload.get("no_post_cost_edge")) * 100.0,
            "regime_label": _clean_string(payload.get("regime_label")) or "unknown",
            "bearish_vote_count": _safe_int(payload.get("bearish_vote_count")),
            "bullish_vote_count": _safe_int(payload.get("bullish_vote_count")),
            "regime_price_momentum_bearish": payload.get("regime_price_momentum_bearish"),
            "regime_signed_flow_bearish": payload.get("regime_signed_flow_bearish"),
            "regime_yes_share_bearish": payload.get("regime_yes_share_bearish"),
            "regime_price_momentum_bullish": payload.get("regime_price_momentum_bullish"),
            "regime_signed_flow_bullish": payload.get("regime_signed_flow_bullish"),
            "regime_yes_share_bullish": payload.get("regime_yes_share_bullish"),
            "bucket_policy_side": None,
            "bucket_policy_dimension": None,
            "bucket_policy_bucket": None,
            "offline_rule_side": None,
            "same_as_offline_rule": None,
            "one_sided_quote": None,
        }
        canonical_rows.append(row)

    skip_reason_rows = [
        {
            "model": model,
            "reason": reason,
            "count": count,
            "source_type": "live_research",
            "bucket_policy_side": bucket_policy_side,
            "bucket_policy_dimension": bucket_policy_dimension,
            "bucket_policy_bucket": bucket_policy_bucket,
        }
        for (model, reason, bucket_policy_side, bucket_policy_dimension, bucket_policy_bucket), count in sorted(skip_reason_counter.items())
    ]
    metadata_extras = {"summary_snapshots": summary_snapshots}
    metadata_extras.update(_same_ticker_side_stats(canonical_rows))
    metadata_extras.update(_time_lag_summary([row for row in canonical_rows if row.get("status") == "settled"]))

    return SourceLoadResult(
        source_type="live_research",
        run_dir=run_dir,
        run_name=run_dir.name,
        canonical_rows=canonical_rows,
        skipped_lines_by_file=skipped_lines_by_file,
        skip_reason_rows=skip_reason_rows,
        metadata_extras=metadata_extras,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_prediction_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {}
    prob_column = "calibrated_probability" if "calibrated_probability" in predictions.columns else "raw_probability"
    probability = predictions[prob_column].astype(float).clip(1e-9, 1.0 - 1e-9)
    actual_yes = predictions["actual_outcome"].astype(str).eq("YES")
    predicted_yes = probability >= 0.5
    tp = int((predicted_yes & actual_yes).sum())
    tn = int((~predicted_yes & ~actual_yes).sum())
    fp = int((predicted_yes & ~actual_yes).sum())
    fn = int((~predicted_yes & actual_yes).sum())
    total = len(predictions)

    def ratio(numerator: int, denominator: int) -> float | None:
        if denominator == 0:
            return None
        return numerator / denominator

    log_loss = float(
        -(
            actual_yes.astype(float) * probability.map(math.log)
            + (1.0 - actual_yes.astype(float)) * (1.0 - probability).map(math.log)
        ).mean()
    )
    brier = float(((probability - actual_yes.astype(float)) ** 2).mean())
    accuracy = ratio(tp + tn, total)
    yes_precision = ratio(tp, tp + fp)
    yes_recall = ratio(tp, tp + fn)
    no_precision = ratio(tn, tn + fn)
    no_recall = ratio(tn, tn + fp)
    return {
        "rows": total,
        "log_loss": log_loss,
        "brier_score": brier,
        "accuracy": accuracy,
        "yes_precision": yes_precision,
        "yes_recall": yes_recall,
        "no_precision": no_precision,
        "no_recall": no_recall,
        "predicted_yes_rate": float(predicted_yes.mean()),
        "actual_yes_rate": float(actual_yes.mean()),
        "probability_column": prob_column,
    }


def _context_label(summary: dict[str, Any], feature_manifest: dict[str, Any]) -> str:
    feature_order = feature_manifest.get("feature_order") or []
    if any("kxbtcd" in str(feature).lower() for feature in feature_order):
        return "hourly_context"
    if "context" in summary.get("model_family", "").lower():
        return "hourly_context"
    return "base"


def _trade_metrics_row(
    run_name: str,
    model_family: str,
    phase: str,
    summary_metrics: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": run_name,
        "model_family": model_family,
        "phase": phase,
        "trades": _safe_int(summary_metrics.get("trades")),
        "net_pnl_dollars": _safe_float(summary_metrics.get("net_pnl_dollars")),
        "return_pct": _safe_float(summary_metrics.get("return_pct")),
        "max_drawdown_dollars": _safe_float(summary_metrics.get("max_drawdown_dollars")),
        "max_drawdown_pct": _safe_float(summary_metrics.get("max_drawdown_pct")),
        "log_loss": _safe_float(summary_metrics.get("log_loss")),
        "selection_mode": _clean_string(summary_metrics.get("selection_mode")),
        "yes_trades": next((row.get("trades") for row in diagnostics.get("side_breakdown", []) if row.get("side") == "YES"), None),
        "no_trades": next((row.get("trades") for row in diagnostics.get("side_breakdown", []) if row.get("side") == "NO"), None),
        "yes_win_rate": next((row.get("win_rate") for row in diagnostics.get("side_breakdown", []) if row.get("side") == "YES"), None),
        "no_win_rate": next((row.get("win_rate") for row in diagnostics.get("side_breakdown", []) if row.get("side") == "NO"), None),
    }


def _canonical_rows_from_trade_records(
    df: pd.DataFrame,
    *,
    run_name: str,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    frame = df.copy()
    if "close_time" in frame.columns:
        frame["__sort_time"] = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")
    elif "created_time" in frame.columns:
        frame["__sort_time"] = pd.to_datetime(frame["created_time"], utc=True, errors="coerce")
    else:
        frame["__sort_time"] = pd.NaT
    frame = frame.sort_values(["__sort_time", "ticker", "trade_id"], kind="stable").reset_index(drop=True)
    if "net_pnl_dollars" in frame.columns:
        frame["__cumulative_net_pnl"] = frame["net_pnl_dollars"].fillna(0.0).cumsum()
    else:
        frame["__cumulative_net_pnl"] = None

    rows: list[dict[str, Any]] = []
    for _, trade in frame.iterrows():
        side = _clean_string(trade.get("side"))
        predicted_yes_probability = _safe_float(trade.get("predicted_yes_probability"))
        chosen_probability = None
        if side == "YES":
            chosen_probability = predicted_yes_probability
        elif side == "NO" and predicted_yes_probability is not None:
            chosen_probability = 1.0 - predicted_yes_probability
        post_cost_edge = _safe_float(trade.get("post_cost_edge"))
        rows.append(
            {
                "source_type": "offline_artifacts",
                "run_name": run_name,
                "model": run_name,
                "ticker": _clean_string(trade.get("ticker")),
                "decision_id": _clean_string(trade.get("trade_id")),
                "sample_id": None,
                "side": side,
                "recorded_at": _timestamp_iso(trade.get("created_time")),
                "settled_at": _timestamp_iso(trade.get("close_time")),
                "status": "settled",
                "settlement_result": _clean_string(trade.get("actual_outcome")),
                "is_win": bool(trade.get("is_win")) if trade.get("is_win") is not None else None,
                "realized_pnl_dollars": _safe_float(trade.get("net_pnl_dollars")),
                "cumulative_realized_pnl_dollars": _safe_float(trade.get("__cumulative_net_pnl")),
                "price_bucket": _clean_string(trade.get("price_bucket")) or _price_bucket_label(_safe_int(trade.get("reference_price_cents"))),
                "probability_bucket": _probability_bucket_label(chosen_probability),
                "edge_bucket": _edge_bucket_label(None if post_cost_edge is None else post_cost_edge * 100.0),
                "tau_bucket": _clean_string(trade.get("tau_bucket")) or _tau_bucket_label(_safe_float(trade.get("tau_minutes"))),
                "reference_price_cents": _safe_int(trade.get("reference_price_cents")),
                "chosen_side_probability": chosen_probability,
                "chosen_post_cost_edge_cents": None if post_cost_edge is None else post_cost_edge * 100.0,
                "tau_minutes": _safe_float(trade.get("tau_minutes")),
                "cash_required_dollars": _safe_float(trade.get("cash_required_dollars")),
                "quote_spread_cents": None,
                "quote_spread_bucket": "unknown",
                "quote_age_seconds": None,
                "quote_age_bucket": "unknown",
                "quote_mid_prob": None,
                "buy_yes_price_cents": None,
                "buy_no_price_cents": None,
                "predicted_yes_probability": predicted_yes_probability,
                "predicted_no_probability": None if predicted_yes_probability is None else 1.0 - predicted_yes_probability,
                "feature_basis_market_prob": _safe_float(trade.get("market_prob")),
                "raw_model_edge": None if predicted_yes_probability is None else predicted_yes_probability - _safe_float(trade.get("market_prob")),
                "yes_post_cost_edge_cents": None,
                "no_post_cost_edge_cents": None,
                "regime_label": _clean_string(trade.get("regime_label")) or "unknown",
                "bearish_vote_count": _safe_int(trade.get("bearish_vote_count")),
                "bullish_vote_count": _safe_int(trade.get("bullish_vote_count")),
                "regime_price_momentum_bearish": trade.get("regime_price_momentum_bearish"),
                "regime_signed_flow_bearish": trade.get("regime_signed_flow_bearish"),
                "regime_yes_share_bearish": trade.get("regime_yes_share_bearish"),
                "regime_price_momentum_bullish": trade.get("regime_price_momentum_bullish"),
                "regime_signed_flow_bullish": trade.get("regime_signed_flow_bullish"),
                "regime_yes_share_bullish": trade.get("regime_yes_share_bullish"),
                "offline_rule_side": None,
                "same_as_offline_rule": None,
                "one_sided_quote": None,
            }
        )
    return rows


def _load_offline_artifact_run(run_dir: Path) -> SourceLoadResult:
    summary = _read_json(run_dir / "summary.json")
    test_metrics = _read_json(run_dir / "test_metrics.json")
    validation_metrics = _read_json(run_dir / "validation_metrics.json")
    test_diagnostics = _read_json(run_dir / "test_diagnostics.json")
    validation_diagnostics = _read_json(run_dir / "validation_diagnostics.json")
    feature_manifest = _read_json(run_dir / "feature_manifest.json")

    model_family = _clean_string(summary.get("model_family")) or run_dir.parent.name
    context_label = _context_label(summary, feature_manifest)
    warnings: list[str] = []
    notes: list[str] = []

    prediction_metrics_rows: list[dict[str, Any]] = []
    predictions_path = run_dir / "test_predictions.parquet"
    if predictions_path.exists():
        prediction_metrics = _compute_prediction_metrics(pd.read_parquet(predictions_path))
        if prediction_metrics:
            prediction_metrics_rows.append(
                {
                    "model": run_dir.name,
                    "model_family": model_family,
                    "context_label": context_label,
                    **prediction_metrics,
                }
            )
    else:
        warnings.append("Missing test_predictions.parquet; prediction leaderboard fields are incomplete.")

    trade_metrics_rows: list[dict[str, Any]] = []
    if validation_metrics:
        trade_metrics_rows.append(
            _trade_metrics_row(run_dir.name, model_family, "validation", validation_metrics, validation_diagnostics)
        )
    if test_metrics:
        row = _trade_metrics_row(run_dir.name, model_family, "test", test_metrics, test_diagnostics)
        validation_net = _safe_float(validation_metrics.get("net_pnl_dollars"))
        test_net = _safe_float(test_metrics.get("net_pnl_dollars"))
        row["net_pnl_drift_dollars"] = (
            None if validation_net is None or test_net is None else test_net - validation_net
        )
        trade_metrics_rows.append(row)

    calibration_summary_rows: list[dict[str, Any]] = []
    for calibration_source, bucket_rows in (test_diagnostics.get("calibration") or {}).items():
        if not isinstance(bucket_rows, list):
            continue
        for bucket_row in bucket_rows:
            calibration_summary_rows.append(
                {
                    "model": run_dir.name,
                    "context_label": context_label,
                    "calibration_source": calibration_source,
                    **bucket_row,
                }
            )

    canonical_rows: list[dict[str, Any]] = []
    model_totals_override: list[dict[str, Any]] = []
    trade_records_path = run_dir / "test_trade_records.parquet"
    if trade_records_path.exists():
        canonical_rows = _canonical_rows_from_trade_records(pd.read_parquet(trade_records_path), run_name=run_dir.name)
    else:
        warnings.append("Missing test_trade_records.parquet; canonical trade rows and bucket summaries may be limited.")
        model_totals_override.append(
            {
                "model": run_dir.name,
                "source_type": "offline_artifacts",
                "recorded_count": _safe_int(test_metrics.get("trades")),
                "settled_count": _safe_int(test_metrics.get("trades")),
                "open_count": 0,
                "wins": None,
                "losses": None,
                "win_rate": None,
                "net_pnl_dollars": _safe_float(test_metrics.get("net_pnl_dollars")),
                "avg_pnl_dollars": None,
            }
        )

    metadata_extras = {
        "feature_count": _safe_int(summary.get("feature_count")),
        "model_family": model_family,
        "model_type": _clean_string(summary.get("model_type")),
        "context_label": context_label,
        "train_rows": _safe_int(summary.get("train_rows")),
        "validation_rows": _safe_int(summary.get("validation_rows")),
        "test_rows": _safe_int(summary.get("test_rows")),
    }
    if validation_metrics and test_metrics:
        metadata_extras["validation_vs_test_trade_drift_dollars"] = (
            _safe_float(test_metrics.get("net_pnl_dollars")) or 0.0
        ) - (_safe_float(validation_metrics.get("net_pnl_dollars")) or 0.0)
    if summary.get("policy"):
        notes.append("Policy selection metadata loaded from summary.json.")

    return SourceLoadResult(
        source_type="offline_artifacts",
        run_dir=run_dir,
        run_name=run_dir.name,
        canonical_rows=canonical_rows,
        notes=notes,
        warnings=warnings,
        model_totals_override=model_totals_override,
        prediction_metrics_rows=prediction_metrics_rows,
        trade_metrics_rows=trade_metrics_rows,
        calibration_summary_rows=calibration_summary_rows,
        metadata_extras=metadata_extras,
    )


def _quote_quality_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settled_rows = [row for row in rows if row.get("status") == "settled"]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in settled_rows:
        grouped[(str(row.get("model")), "quote_spread_bucket", str(row.get("quote_spread_bucket")))].append(row)
        grouped[(str(row.get("model")), "quote_age_bucket", str(row.get("quote_age_bucket")))].append(row)
        grouped[(str(row.get("model")), "one_sided_quote", _bool_bucket(row.get("one_sided_quote")))].append(row)
    summaries: list[dict[str, Any]] = []
    for (model, dimension, bucket), group in sorted(grouped.items()):
        pnl = sum(_safe_float(row.get("realized_pnl_dollars")) or 0.0 for row in group)
        wins = sum(1 for row in group if row.get("is_win") is True)
        summaries.append(
            {
                "model": model,
                "dimension": dimension,
                "bucket": bucket,
                "settled_count": len(group),
                "wins": wins,
                "losses": len(group) - wins,
                "win_rate": None if not group else wins / len(group),
                "net_pnl_dollars": pnl,
                "avg_pnl_dollars": None if not group else pnl / len(group),
            }
        )
    return summaries


def _thesis_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        thesis_id = _clean_string(row.get("thesis_id"))
        model = _clean_string(row.get("model"))
        if thesis_id is None or model is None:
            continue
        grouped[(model, thesis_id)].append(row)

    summaries: list[dict[str, Any]] = []
    for (model, thesis_id), group in sorted(grouped.items()):
        settled_rows = [row for row in group if row.get("status") == "settled"]
        recorded_times = [_parse_timestamp(row.get("recorded_at")) for row in group]
        settled_times = [_parse_timestamp(row.get("settled_at")) for row in settled_rows]
        budget_values = [_safe_float(row.get("total_thesis_budget_dollars")) for row in group]
        loss_values = [_safe_float(row.get("worst_case_loss_dollars")) for row in group]
        pnl_values = [_safe_float(row.get("realized_pnl_dollars")) for row in settled_rows]
        tranche_indices = [_safe_int(row.get("tranche_index")) for row in group]
        summaries.append(
            {
                "model": model,
                "thesis_id": thesis_id,
                "ticker": _clean_string(group[0].get("ticker")),
                "status": "settled" if len(settled_rows) == len(group) and group else "open",
                "tranche_count": len(group),
                "max_tranche_index": max((value for value in tranche_indices if value is not None), default=None),
                "settled_tranche_count": len(settled_rows),
                "net_pnl_dollars": sum(value or 0.0 for value in pnl_values),
                "total_cash_required_dollars": sum(_safe_float(row.get("cash_required_dollars")) or 0.0 for row in group),
                "total_thesis_budget_dollars": max((value for value in budget_values if value is not None), default=None),
                "max_worst_case_loss_dollars": max((value for value in loss_values if value is not None), default=None),
                "opened_at": min((value for value in recorded_times if value is not None), default=None),
                "settled_at": max((value for value in settled_times if value is not None), default=None),
            }
        )
    return summaries


def _model_totals_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row.get("model"))].append(row)
    totals: list[dict[str, Any]] = []
    for model, model_rows in sorted(by_model.items()):
        settled_rows = [row for row in model_rows if row.get("status") == "settled"]
        open_rows = [row for row in model_rows if row.get("status") != "settled"]
        wins = sum(1 for row in settled_rows if row.get("is_win") is True)
        net_pnl = sum(_safe_float(row.get("realized_pnl_dollars")) or 0.0 for row in settled_rows)
        totals.append(
            {
                "model": model,
                "source_type": str(model_rows[0].get("source_type")),
                "recorded_count": len(model_rows),
                "settled_count": len(settled_rows),
                "open_count": len(open_rows),
                "wins": wins if settled_rows else None,
                "losses": (len(settled_rows) - wins) if settled_rows else None,
                "win_rate": None if not settled_rows else wins / len(settled_rows),
                "net_pnl_dollars": net_pnl if settled_rows else None,
                "avg_pnl_dollars": None if not settled_rows else net_pnl / len(settled_rows),
            }
        )
    return totals


def _bucket_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settled_rows = [row for row in rows if row.get("status") == "settled"]
    bucket_specs = (
        ("tau", "tau_bucket", TAU_BUCKET_ORDER),
        ("price", "price_bucket", PRICE_BUCKET_ORDER),
        ("probability", "probability_bucket", PROBABILITY_BUCKET_ORDER),
        ("edge", "edge_bucket", EDGE_BUCKET_ORDER),
        ("regime", "regime_label", REGIME_ORDER),
    )
    summaries: list[dict[str, Any]] = []
    for dimension, key, ordering in bucket_specs:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in settled_rows:
            grouped[str(row.get(key, "unknown"))].append(row)
        for bucket in ordering:
            group = grouped.get(bucket, [])
            wins = sum(1 for row in group if row.get("is_win") is True)
            pnl = sum(_safe_float(row.get("realized_pnl_dollars")) or 0.0 for row in group)
            summaries.append(
                {
                    "scope": "all_models",
                    "dimension": dimension,
                    "bucket": bucket,
                    "settled_count": len(group),
                    "wins": wins,
                    "losses": len(group) - wins,
                    "win_rate": None if not group else wins / len(group),
                    "net_pnl_dollars": pnl,
                    "avg_pnl_dollars": None if not group else pnl / len(group),
                }
            )
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in settled_rows:
            by_model[str(row.get("model"))].append(row)
        for model, model_rows in sorted(by_model.items()):
            grouped = defaultdict(list)
            for row in model_rows:
                grouped[str(row.get(key, "unknown"))].append(row)
            for bucket in ordering:
                group = grouped.get(bucket, [])
                wins = sum(1 for row in group if row.get("is_win") is True)
                pnl = sum(_safe_float(row.get("realized_pnl_dollars")) or 0.0 for row in group)
                summaries.append(
                    {
                        "scope": model,
                        "dimension": dimension,
                        "bucket": bucket,
                        "settled_count": len(group),
                        "wins": wins,
                        "losses": len(group) - wins,
                        "win_rate": None if not group else wins / len(group),
                        "net_pnl_dollars": pnl,
                        "avg_pnl_dollars": None if not group else pnl / len(group),
                    }
                )
    return summaries


def _side_bucket_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settled_rows = [row for row in rows if row.get("status") == "settled"]
    summaries: list[dict[str, Any]] = []
    bucket_specs = (
        ("tau", "tau_bucket", TAU_BUCKET_ORDER),
        ("price", "price_bucket", PRICE_BUCKET_ORDER),
        ("probability", "probability_bucket", PROBABILITY_BUCKET_ORDER),
        ("edge", "edge_bucket", EDGE_BUCKET_ORDER),
        ("regime", "regime_label", REGIME_ORDER),
    )
    for side in ("YES", "NO"):
        side_rows = [row for row in settled_rows if row.get("side") == side]
        for dimension, key, ordering in bucket_specs:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in side_rows:
                grouped[str(row.get(key, "unknown"))].append(row)
            for bucket in ordering:
                group = grouped.get(bucket, [])
                wins = sum(1 for row in group if row.get("is_win") is True)
                pnl = sum(_safe_float(row.get("realized_pnl_dollars")) or 0.0 for row in group)
                summaries.append(
                    {
                        "side": side,
                        "scope": "all_models",
                        "dimension": dimension,
                        "bucket": bucket,
                        "settled_count": len(group),
                        "wins": wins,
                        "losses": len(group) - wins,
                        "win_rate": None if not group else wins / len(group),
                        "net_pnl_dollars": pnl,
                        "avg_pnl_dollars": None if not group else pnl / len(group),
                    }
                )
    return summaries


def _regime_bucket_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settled_rows = [row for row in rows if row.get("status") == "settled"]
    summaries: list[dict[str, Any]] = []
    bucket_specs = (
        ("tau", "tau_bucket", TAU_BUCKET_ORDER),
        ("price", "price_bucket", PRICE_BUCKET_ORDER),
        ("probability", "probability_bucket", PROBABILITY_BUCKET_ORDER),
        ("edge", "edge_bucket", EDGE_BUCKET_ORDER),
    )
    for regime_label in REGIME_ORDER:
        regime_rows = [row for row in settled_rows if str(row.get("regime_label") or "unknown") == regime_label]
        for dimension, key, ordering in bucket_specs:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in regime_rows:
                grouped[str(row.get(key, "unknown"))].append(row)
            for bucket in ordering:
                group = grouped.get(bucket, [])
                wins = sum(1 for row in group if row.get("is_win") is True)
                pnl = sum(_safe_float(row.get("realized_pnl_dollars")) or 0.0 for row in group)
                summaries.append(
                    {
                        "scope": "all_models",
                        "regime_label": regime_label,
                        "dimension": dimension,
                        "bucket": bucket,
                        "settled_count": len(group),
                        "wins": wins,
                        "losses": len(group) - wins,
                        "win_rate": None if not group else wins / len(group),
                        "net_pnl_dollars": pnl,
                        "avg_pnl_dollars": None if not group else pnl / len(group),
                    }
                )
    return summaries


def _combo_summary_rows(rows: list[dict[str, Any]], min_combo_count: int) -> list[dict[str, Any]]:
    settled_rows = [row for row in rows if row.get("status") == "settled"]
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in settled_rows:
        grouped[
            (
                str(row.get("model")),
                str(row.get("tau_bucket")),
                str(row.get("price_bucket")),
                str(row.get("probability_bucket")),
                str(row.get("edge_bucket")),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (model, tau_bucket, price_bucket, probability_bucket, edge_bucket), group in sorted(grouped.items()):
        if len(group) < min_combo_count:
            continue
        wins = sum(1 for row in group if row.get("is_win") is True)
        pnl = sum(_safe_float(row.get("realized_pnl_dollars")) or 0.0 for row in group)
        summaries.append(
            {
                "model": model,
                "tau_bucket": tau_bucket,
                "price_bucket": price_bucket,
                "probability_bucket": probability_bucket,
                "edge_bucket": edge_bucket,
                "settled_count": len(group),
                "wins": wins,
                "losses": len(group) - wins,
                "win_rate": wins / len(group),
                "net_pnl_dollars": pnl,
                "avg_pnl_dollars": pnl / len(group),
            }
        )
    return summaries


def _timeline_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settled_rows = [row for row in rows if row.get("status") == "settled"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in settled_rows:
        ts = _parse_timestamp(row.get("settled_at")) or _parse_timestamp(row.get("recorded_at"))
        if ts is None:
            continue
        day = ts.date().isoformat()
        grouped[("all_models", day)].append(row)
        grouped[(str(row.get("model")), day)].append(row)
    summaries: list[dict[str, Any]] = []
    for (scope, day), group in sorted(grouped.items()):
        wins = sum(1 for row in group if row.get("is_win") is True)
        pnl = sum(_safe_float(row.get("realized_pnl_dollars")) or 0.0 for row in group)
        summaries.append(
            {
                "scope": scope,
                "day": day,
                "settled_count": len(group),
                "wins": wins,
                "losses": len(group) - wins,
                "win_rate": wins / len(group),
                "net_pnl_dollars": pnl,
                "avg_pnl_dollars": pnl / len(group),
            }
        )
    return summaries


def _top_bucket_rows(bucket_rows: list[dict[str, Any]], *, count: int, profitable: bool) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for row in bucket_rows:
        if row.get("scope") != "all_models":
            continue
        if _safe_int(row.get("settled_count")) in (None, 0):
            continue
        pnl = _safe_float(row.get("net_pnl_dollars")) or 0.0
        if profitable and pnl <= 0.0:
            continue
        if not profitable and pnl >= 0.0:
            continue
        eligible.append(row)
    return sorted(
        eligible,
        key=lambda row: (_safe_float(row.get("net_pnl_dollars")) or 0.0),
        reverse=profitable,
    )[:count]


def _best_and_worst_models(model_totals: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    eligible = [row for row in model_totals if _safe_float(row.get("net_pnl_dollars")) is not None]
    if not eligible:
        return None, None
    best = max(eligible, key=lambda row: _safe_float(row.get("net_pnl_dollars")) or float("-inf"))
    worst = min(eligible, key=lambda row: _safe_float(row.get("net_pnl_dollars")) or float("inf"))
    return best, worst


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _report_markdown(
    result: SourceLoadResult,
    model_totals: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    regime_bucket_rows: list[dict[str, Any]],
    combo_rows: list[dict[str, Any]],
    timeline_rows: list[dict[str, Any]],
) -> str:
    best_model, worst_model = _best_and_worst_models(model_totals)
    total_recorded = sum(_safe_int(row.get("recorded_count")) or 0 for row in model_totals)
    total_settled = sum(_safe_int(row.get("settled_count")) or 0 for row in model_totals)
    total_open = sum(_safe_int(row.get("open_count")) or 0 for row in model_totals)
    aggregate_pnl = sum(_safe_float(row.get("net_pnl_dollars")) or 0.0 for row in model_totals if _safe_float(row.get("net_pnl_dollars")) is not None)
    profitable_buckets = _top_bucket_rows(bucket_rows, count=6, profitable=True)
    losing_buckets = _top_bucket_rows(bucket_rows, count=6, profitable=False)
    strongest_combos = sorted(combo_rows, key=lambda row: (_safe_float(row.get("net_pnl_dollars")) or 0.0), reverse=True)[:8]
    weakest_combos = sorted(combo_rows, key=lambda row: (_safe_float(row.get("net_pnl_dollars")) or 0.0))[:8]
    regime_summary_rows = [row for row in bucket_rows if row.get("scope") == "all_models" and row.get("dimension") == "regime"]
    downtrend_bucket_rows = [
        row for row in regime_bucket_rows
        if row.get("scope") == "all_models" and row.get("regime_label") == "downtrend"
    ]
    strongest_downtrend_buckets = sorted(
        [row for row in downtrend_bucket_rows if (_safe_float(row.get("net_pnl_dollars")) or 0.0) > 0.0],
        key=lambda row: (_safe_float(row.get("net_pnl_dollars")) or 0.0),
        reverse=True,
    )[:6]
    weakest_downtrend_buckets = sorted(
        [row for row in downtrend_bucket_rows if (_safe_float(row.get("net_pnl_dollars")) or 0.0) < 0.0],
        key=lambda row: (_safe_float(row.get("net_pnl_dollars")) or 0.0),
    )[:6]

    lines = [
        f"# Kalshi Performance Analysis: {result.run_name}",
        "",
        f"- Source type: `{result.source_type}`",
        f"- Analyzed path: `{result.run_dir}`",
        f"- Recorded rows: `{total_recorded:,}`",
        f"- Settled rows: `{total_settled:,}`",
        f"- Open rows: `{total_open:,}`",
        f"- Aggregate settled PnL: `{_format_currency(aggregate_pnl)}`",
        "",
        "## Model Totals",
        "",
        "| Model | Recorded | Settled | Open | Win Rate | Net PnL | Avg PnL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in model_totals:
        lines.append(
            f"| {row['model']} | {_format_number(_safe_int(row.get('recorded_count')), 0)} | {_format_number(_safe_int(row.get('settled_count')), 0)} | "
            f"{_format_number(_safe_int(row.get('open_count')), 0)} | {_format_pct(_safe_float(row.get('win_rate')))} | "
            f"{_format_currency(_safe_float(row.get('net_pnl_dollars')))} | {_format_currency(_safe_float(row.get('avg_pnl_dollars')))} |"
        )

    lines.extend(["", "## Highlights", ""])
    if best_model is not None:
        lines.append(
            f"- Best model by net PnL: `{best_model['model']}` at "
            f"`{_format_currency(_safe_float(best_model.get('net_pnl_dollars')))}`"
        )
    if worst_model is not None:
        lines.append(
            f"- Weakest model by net PnL: `{worst_model['model']}` at "
            f"`{_format_currency(_safe_float(worst_model.get('net_pnl_dollars')))}`"
        )
    if result.metadata_extras.get("tickers_with_both_sides_total") is not None:
        lines.append(
            f"- Tickers with both sides observed: `{_format_number(result.metadata_extras.get('tickers_with_both_sides_total'), 0)}`"
        )
    if result.metadata_extras.get("settlement_lag_avg_minutes") is not None:
        lines.append(
            f"- Average settlement lag: `{_format_number(result.metadata_extras.get('settlement_lag_avg_minutes'))} minutes`"
        )

    if regime_summary_rows:
        lines.extend(["", "## Regime Summary", "", "| Regime | Settled | Win Rate | Net PnL |", "| --- | ---: | ---: | ---: |"])
        for row in regime_summary_rows:
            lines.append(
                f"| {row['bucket']} | {_format_number(_safe_int(row.get('settled_count')), 0)} | "
                f"{_format_pct(_safe_float(row.get('win_rate')))} | {_format_currency(_safe_float(row.get('net_pnl_dollars')))} |"
            )

    lines.extend(["", "## Profitable Buckets", "", "| Dimension | Bucket | Settled | Win Rate | Net PnL |", "| --- | --- | ---: | ---: | ---: |"])
    for row in profitable_buckets:
        lines.append(
            f"| {row['dimension']} | {row['bucket']} | {_format_number(_safe_int(row.get('settled_count')), 0)} | "
            f"{_format_pct(_safe_float(row.get('win_rate')))} | {_format_currency(_safe_float(row.get('net_pnl_dollars')))} |"
        )

    lines.extend(["", "## Losing Buckets", "", "| Dimension | Bucket | Settled | Win Rate | Net PnL |", "| --- | --- | ---: | ---: | ---: |"])
    for row in losing_buckets:
        lines.append(
            f"| {row['dimension']} | {row['bucket']} | {_format_number(_safe_int(row.get('settled_count')), 0)} | "
            f"{_format_pct(_safe_float(row.get('win_rate')))} | {_format_currency(_safe_float(row.get('net_pnl_dollars')))} |"
        )

    lines.extend(["", "## Strongest Bucket Combos", "", "| Model | Tau | Price | Prob | Edge | Settled | Win Rate | Net PnL |", "| --- | --- | --- | --- | --- | ---: | ---: | ---: |"])
    for row in strongest_combos:
        lines.append(
            f"| {row['model']} | {row['tau_bucket']} | {row['price_bucket']} | {row['probability_bucket']} | {row['edge_bucket']} | "
            f"{_format_number(_safe_int(row.get('settled_count')), 0)} | {_format_pct(_safe_float(row.get('win_rate')))} | {_format_currency(_safe_float(row.get('net_pnl_dollars')))} |"
        )

    lines.extend(["", "## Weakest Bucket Combos", "", "| Model | Tau | Price | Prob | Edge | Settled | Win Rate | Net PnL |", "| --- | --- | --- | --- | --- | ---: | ---: | ---: |"])
    for row in weakest_combos:
        lines.append(
            f"| {row['model']} | {row['tau_bucket']} | {row['price_bucket']} | {row['probability_bucket']} | {row['edge_bucket']} | "
            f"{_format_number(_safe_int(row.get('settled_count')), 0)} | {_format_pct(_safe_float(row.get('win_rate')))} | {_format_currency(_safe_float(row.get('net_pnl_dollars')))} |"
        )

    if downtrend_bucket_rows:
        lines.extend(["", "## Downtrend Bucket Detail", "", "| Dimension | Bucket | Settled | Win Rate | Net PnL |", "| --- | --- | ---: | ---: | ---: |"])
        for row in strongest_downtrend_buckets + weakest_downtrend_buckets:
            lines.append(
                f"| {row['dimension']} | {row['bucket']} | {_format_number(_safe_int(row.get('settled_count')), 0)} | "
                f"{_format_pct(_safe_float(row.get('win_rate')))} | {_format_currency(_safe_float(row.get('net_pnl_dollars')))} |"
            )

    if timeline_rows:
        lines.extend(["", "## Timeline", "", "| Scope | Day | Settled | Win Rate | Net PnL |", "| --- | --- | ---: | ---: | ---: |"])
        for row in timeline_rows[:40]:
            lines.append(
                f"| {row['scope']} | {row['day']} | {_format_number(_safe_int(row.get('settled_count')), 0)} | "
                f"{_format_pct(_safe_float(row.get('win_rate')))} | {_format_currency(_safe_float(row.get('net_pnl_dollars')))} |"
            )

    if result.prediction_metrics_rows:
        lines.extend(["", "## Prediction Metrics", "", "| Model | Rows | Log Loss | Brier | Accuracy | YES Precision | YES Recall | Pred YES Rate | Actual YES Rate |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in result.prediction_metrics_rows:
            lines.append(
                f"| {row['model']} | {_format_number(_safe_int(row.get('rows')), 0)} | {_format_number(_safe_float(row.get('log_loss')), 4)} | "
                f"{_format_number(_safe_float(row.get('brier_score')), 4)} | {_format_pct(_safe_float(row.get('accuracy')))} | "
                f"{_format_pct(_safe_float(row.get('yes_precision')))} | {_format_pct(_safe_float(row.get('yes_recall')))} | "
                f"{_format_pct(_safe_float(row.get('predicted_yes_rate')))} | {_format_pct(_safe_float(row.get('actual_yes_rate')))} |"
            )

    if result.trade_metrics_rows:
        lines.extend(["", "## Trade Metrics", "", "| Model | Phase | Trades | Net PnL | Return | Max DD | Log Loss |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
        for row in result.trade_metrics_rows:
            lines.append(
                f"| {row['model']} | {row['phase']} | {_format_number(_safe_int(row.get('trades')), 0)} | {_format_currency(_safe_float(row.get('net_pnl_dollars')))} | "
                f"{_format_pct(_safe_float(row.get('return_pct')))} | {_format_currency(_safe_float(row.get('max_drawdown_dollars')))} | {_format_number(_safe_float(row.get('log_loss')), 4)} |"
            )

    if result.skip_reason_rows:
        lines.extend(["", "## Skip Reasons", "", "| Model | Reason | Count |", "| --- | --- | ---: |"])
        for row in result.skip_reason_rows[:30]:
            lines.append(f"| {row['model']} | {row['reason']} | {_format_number(_safe_int(row.get('count')), 0)} |")

    if result.quote_quality_rows:
        lines.extend(["", "## Quote Quality Effects", "", "| Model | Dimension | Bucket | Settled | Win Rate | Net PnL |", "| --- | --- | --- | ---: | ---: | ---: |"])
        for row in result.quote_quality_rows[:40]:
            lines.append(
                f"| {row['model']} | {row['dimension']} | {row['bucket']} | {_format_number(_safe_int(row.get('settled_count')), 0)} | "
                f"{_format_pct(_safe_float(row.get('win_rate')))} | {_format_currency(_safe_float(row.get('net_pnl_dollars')))} |"
            )

    if result.thesis_summary_rows:
        lines.extend(["", "## Thesis Summary", "", "| Model | Thesis | Ticker | Tranches | Status | Net PnL | Worst Loss Cap |", "| --- | --- | --- | ---: | --- | ---: | ---: |"])
        for row in result.thesis_summary_rows[:20]:
            thesis_id = str(row["thesis_id"])
            lines.append(
                f"| {row['model']} | {thesis_id[:12]} | {row.get('ticker') or 'n/a'} | "
                f"{_format_number(_safe_int(row.get('tranche_count')), 0)} | {row.get('status') or 'unknown'} | "
                f"{_format_currency(_safe_float(row.get('net_pnl_dollars')))} | "
                f"{_format_currency(_safe_float(row.get('max_worst_case_loss_dollars')))} |"
            )

    if result.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in result.notes)
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.skipped_lines_by_file:
        lines.extend(["", "## Data Quality", ""])
        for file_path, count in sorted(result.skipped_lines_by_file.items()):
            lines.append(f"- Skipped `{count}` malformed JSONL line(s) in `{file_path}`")

    return "\n".join(lines).strip() + "\n"


def _write_run_outputs(
    result: SourceLoadResult,
    *,
    output_dir: Path,
    min_combo_count: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_totals = result.model_totals_override or _model_totals_from_rows(result.canonical_rows)
    bucket_rows = _bucket_summary_rows(result.canonical_rows)
    side_bucket_rows = _side_bucket_summary_rows(result.canonical_rows)
    regime_bucket_rows = _regime_bucket_summary_rows(result.canonical_rows)
    combo_rows = _combo_summary_rows(result.canonical_rows, min_combo_count=min_combo_count)
    timeline_rows = _timeline_summary_rows(result.canonical_rows)

    artifacts: dict[str, str] = {}
    files_to_write = {
        "canonical_rows.csv": result.canonical_rows,
        "model_totals.csv": model_totals,
        "bucket_summary.csv": bucket_rows,
        "side_bucket_summary.csv": side_bucket_rows,
        "regime_bucket_summary.csv": regime_bucket_rows,
        "combo_summary.csv": combo_rows,
        "timeline_summary.csv": timeline_rows,
    }
    optional_files = {
        "skip_reason_summary.csv": result.skip_reason_rows,
        "prediction_metrics.csv": result.prediction_metrics_rows,
        "trade_metrics.csv": result.trade_metrics_rows,
        "calibration_summary.csv": result.calibration_summary_rows,
        "quote_quality_summary.csv": result.quote_quality_rows,
        "thesis_summary.csv": result.thesis_summary_rows,
    }
    for filename, rows in files_to_write.items():
        path = output_dir / filename
        _write_csv(path, rows)
        artifacts[filename] = str(path)
    for filename, rows in optional_files.items():
        if rows:
            path = output_dir / filename
            _write_csv(path, rows)
            artifacts[filename] = str(path)

    metadata = {
        "detected_source_type": result.source_type,
        "run_name": result.run_name,
        "analyzed_path": str(result.run_dir),
        "skipped_malformed_lines_by_file": result.skipped_lines_by_file,
        "generated_artifacts": artifacts,
        "summary_kpis": {
            "model_count": len(model_totals),
            "recorded_count": sum(_safe_int(row.get("recorded_count")) or 0 for row in model_totals),
            "settled_count": sum(_safe_int(row.get("settled_count")) or 0 for row in model_totals),
            "open_count": sum(_safe_int(row.get("open_count")) or 0 for row in model_totals),
            "aggregate_net_pnl_dollars": sum(_safe_float(row.get("net_pnl_dollars")) or 0.0 for row in model_totals if _safe_float(row.get("net_pnl_dollars")) is not None),
        },
        "metadata_extras": result.metadata_extras,
        "warnings": result.warnings,
        "notes": result.notes,
    }

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    artifacts["run_metadata.json"] = str(metadata_path)

    report_path = output_dir / "report.md"
    report_path.write_text(
        _report_markdown(result, model_totals, bucket_rows, regime_bucket_rows, combo_rows, timeline_rows),
        encoding="utf-8",
    )
    artifacts["report.md"] = str(report_path)
    return artifacts


def _load_target(target: AnalysisTarget, environment: str | None = None) -> SourceLoadResult:
    if target.source_type == "live_execution":
        return _load_live_execution_run(target.run_dir, environment=environment)
    if target.source_type == "live_research":
        return _load_live_research_run(target.run_dir, environment=environment)
    if target.source_type == "offline_artifacts":
        return _load_offline_artifact_run(target.run_dir)
    raise ValueError(f"Unsupported source type: {target.source_type}")


def generate_kalshi_performance_reports(
    input_path: str | Path,
    *,
    mode: str = "auto",
    output_dir: str | Path | None = None,
    environment: str | None = None,
    model_filters: tuple[str, ...] = (),
    lookback_days: int | None = None,
    min_combo_count: int = DEFAULT_MIN_COMBO_COUNT,
) -> dict[str, Any]:
    detection = detect_analysis_targets(input_path, mode=mode)
    base_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_REPORTS_ROOT / detection.detected_name
    )
    results: list[dict[str, Any]] = []

    for target in detection.targets:
        loaded = _load_target(target, environment=environment)
        loaded.canonical_rows = _apply_model_filters(loaded.canonical_rows, model_filters)
        loaded.canonical_rows = _filter_by_lookback(loaded.canonical_rows, lookback_days)
        output_path = base_output_dir if len(detection.targets) == 1 else base_output_dir / target.run_name
        artifacts = _write_run_outputs(loaded, output_dir=output_path, min_combo_count=min_combo_count)
        results.append(
            {
                "run_name": target.run_name,
                "source_type": target.source_type,
                "input_path": str(target.run_dir),
                "output_dir": str(output_path),
                "artifacts": artifacts,
            }
        )

    return {
        "detected_mode": detection.detected_mode,
        "detected_name": detection.detected_name,
        "output_dir": str(base_output_dir),
        "runs": results,
    }
