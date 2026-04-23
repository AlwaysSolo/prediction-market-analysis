from __future__ import annotations

import io
import json
import math
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zstandard as zstd
from scipy import stats

from src.live.kalshi.bucket_policy import (
    build_chosen_side_buckets,
    evaluate_bucket_ban_policy,
)
from src.live.kalshi.performance_analysis import (
    _filter_paths_by_lookback,
    _iter_jsonl_tolerant,
    _iter_jsonl_zst_tolerant,
    _load_live_execution_run,
)
from src.live.kalshi.research.engine import KalshiResearchSamplerConfig
from src.live.kalshi.signal_risk import (
    calculate_cost_metrics,
    calculate_realized_cash_metrics,
    find_max_acceptable_entry_price_cents,
)

PARITY_SCHEMA_VERSION = "kalshi_parity_report_v1"
BOOTSTRAP_SEED = 0
BOOTSTRAP_RESAMPLES = 2000
DEFAULT_REPORT_TIMEZONE = "America/New_York"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_ROLLING_WINDOW_DAYS = 14
DEFAULT_MAX_SNAPSHOT_LAG_MS = 1000
DEFAULT_ECONOMIC_FLOOR_PER_TRADE = 0.02
DEFAULT_LOW_SNAPSHOT_MATCH_COVERAGE = 0.80
DEFAULT_LOW_PRIMARY_COHORT_COVERAGE = 0.60
PRIMARY_GAP_COLUMNS = (
    "shadow_minus_research_pnl_dollars",
    "live_minus_shadow_pnl_dollars",
)
PRIMARY_SEGMENTS = (
    "declined",
    "entered_different_side",
    "entered_different_size",
    "partial_fill_live",
)


@dataclass(frozen=True)
class ParityArtifacts:
    paired_trades: pd.DataFrame
    daily_rollup: pd.DataFrame
    daily_segment_rollup: pd.DataFrame
    rolling_rollup: pd.DataFrame
    summary: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ReplayOutcome:
    status: str
    reason: str | None
    side: str | None
    contracts: int
    reference_price_cents: int | None
    max_acceptable_entry_price_cents: int | None
    predicted_yes_probability: float
    predicted_no_probability: float
    feature_basis_market_prob: float
    raw_model_edge: float
    yes_post_cost_edge: float | None
    no_post_cost_edge: float | None
    chosen_post_cost_edge: float | None
    estimated_entry_cost_dollars: float | None
    estimated_fees_dollars: float | None
    estimated_cash_required_dollars: float | None
    tau_minutes: float
    regime_label: str
    price_bucket: str | None
    tau_bucket: str | None
    probability_bucket: str | None
    edge_bucket: str | None
    bucket_policy_dimension: str | None
    bucket_policy_bucket: str | None
    bucket_policy_side: str | None
    feature_row_id: str
    market_event_id: str
    model_file: str | None
    schema_version: str | None
    event_time: datetime
    run_name: str
    model: str
    config_origin: str


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    text = str(value).strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _local_date(value: Any, timezone: ZoneInfo) -> date | None:
    parsed = _parse_timestamp(value)
    return None if parsed is None else parsed.astimezone(timezone).date()


def _time_bucket_key(value: Any, timezone: ZoneInfo) -> str | None:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return None
    local = parsed.astimezone(timezone)
    return local.replace(second=0, microsecond=0).isoformat()


def _to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if "time" in column or column.endswith("_at"):
            frame[column] = frame[column].map(_parse_timestamp)
    return frame


def _read_stage_rows(
    run_dir: Path,
    stage_name: str,
    *,
    lookback_days: int | None,
) -> pd.DataFrame:
    archive_root = run_dir / "archive"
    stage_root = archive_root / stage_name
    stage_staging_root = archive_root / f"{stage_name}_staging"
    rows: list[dict[str, Any]] = []

    if stage_root.exists():
        parquet_paths = _filter_paths_by_lookback(sorted(stage_root.rglob("*.parquet")), lookback_days)
        for parquet_path in parquet_paths:
            frame = pd.read_parquet(parquet_path)
            rows.extend(frame.to_dict(orient="records"))

    if stage_staging_root.exists():
        zst_paths = _filter_paths_by_lookback(sorted(stage_staging_root.rglob("*.jsonl.zst")), lookback_days)
        for zst_path in zst_paths:
            parsed_rows, _skipped = _iter_jsonl_zst_tolerant(zst_path)
            rows.extend(parsed_rows)

    return _to_dataframe(rows)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _find_execution_started_payload(run_dir: Path) -> dict[str, Any]:
    execution_root = run_dir / "execution"
    if not execution_root.exists():
        return {}
    for events_path in sorted(execution_root.rglob("events.jsonl")):
        for event in _read_jsonl_rows(events_path)[:50]:
            if event.get("event_type") == "execution_started":
                payload = event.get("payload")
                if isinstance(payload, dict):
                    return payload
    return {}


def _infer_live_run_mode(run_dir: Path) -> str:
    payload = _find_execution_started_payload(run_dir)
    payload_mode = _clean_string(payload.get("mode"))
    if payload_mode is not None:
        return payload_mode.lower()
    run_name_lower = run_dir.name.lower()
    if any(token in run_name_lower for token in ("paper", "shadow", "preview")):
        return "paper"
    if "live" in run_name_lower:
        return "live"
    return "unknown"


def _discover_live_runs(
    live_root: Path,
    *,
    explicit_runs: tuple[Path, ...],
) -> list[Path]:
    if explicit_runs:
        return sorted({path.resolve() for path in explicit_runs if path.exists()})
    if not live_root.exists():
        return []
    discovered: list[Path] = []
    for child in sorted(live_root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "signal").exists() or not (child / "execution").exists():
            continue
        if _infer_live_run_mode(child) == "live":
            discovered.append(child.resolve())
    return discovered


def _discover_research_runs(
    research_root: Path,
    *,
    explicit_runs: tuple[Path, ...],
) -> list[Path]:
    if explicit_runs:
        return sorted({path.resolve() for path in explicit_runs if path.exists()})
    if not research_root.exists():
        return []
    return sorted(
        child.resolve()
        for child in research_root.iterdir()
        if child.is_dir() and (child / "research").exists()
    )


def _load_live_signal_events(run_dir: Path, *, lookback_days: int | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    signal_root = run_dir / "signal"
    if not signal_root.exists():
        return pd.DataFrame()
    event_paths = _filter_paths_by_lookback(sorted(signal_root.rglob("events.jsonl")), lookback_days)
    for events_path in event_paths:
        parts = events_path.parts
        try:
            signal_index = parts.index("signal")
            model = parts[signal_index + 1]
        except (ValueError, IndexError):
            model = "unknown"
        raw_rows, _skipped = _iter_jsonl_tolerant(events_path)
        for event in raw_rows:
            if event.get("event_type") != "signal_decision":
                continue
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                continue
            row = {
                "run_name": run_dir.name,
                "model": model,
                "logged_at": _parse_timestamp(event.get("logged_at")),
                "approved": bool(payload.get("approved")),
                "decision_id": _clean_string(payload.get("decision_id")),
                "ticker": _clean_string(payload.get("ticker")),
                "side": _clean_string(payload.get("side")),
                "feature_row_id": _clean_string(payload.get("feature_row_id")),
                "market_event_id": _clean_string(payload.get("market_event_id")),
                "contracts": _safe_int(payload.get("contracts")),
                "reference_price_cents": _safe_int(payload.get("reference_price_cents")),
                "buy_yes_price_cents": _safe_int(payload.get("buy_yes_price_cents")),
                "buy_no_price_cents": _safe_int(payload.get("buy_no_price_cents")),
                "quote_age_seconds": _safe_float(payload.get("quote_age_seconds")),
                "quote_spread_cents": _safe_int(payload.get("quote_spread_cents")),
                "predicted_yes_probability": _safe_float(payload.get("predicted_yes_probability")),
                "predicted_no_probability": _safe_float(payload.get("predicted_no_probability")),
                "feature_basis_market_prob": _safe_float(payload.get("feature_basis_market_prob")),
                "raw_model_edge": _safe_float(payload.get("raw_model_edge")),
                "yes_post_cost_edge": _safe_float(payload.get("yes_post_cost_edge")),
                "no_post_cost_edge": _safe_float(payload.get("no_post_cost_edge")),
                "tau_minutes": _safe_float(payload.get("tau_minutes")),
                "block_reason": _clean_string(payload.get("block_reason")),
            }
            rows.append(row)
    return _to_dataframe(rows)


def _load_research_configs(
    run_dir: Path,
    *,
    lookback_days: int | None,
) -> dict[str, tuple[KalshiResearchSamplerConfig, str]]:
    configs: dict[str, tuple[datetime, KalshiResearchSamplerConfig, str]] = {}
    research_root = run_dir / "research"
    if not research_root.exists():
        return {}
    event_paths = _filter_paths_by_lookback(sorted(research_root.rglob("events.jsonl")), lookback_days)
    for events_path in event_paths:
        parts = events_path.parts
        try:
            research_index = parts.index("research")
            model = parts[research_index + 1]
        except (ValueError, IndexError):
            model = "unknown"
        for event in _read_jsonl_rows(events_path):
            if event.get("event_type") != "research_started":
                continue
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                continue
            started_at = _parse_timestamp(event.get("logged_at")) or datetime.min.replace(tzinfo=UTC)
            config = KalshiResearchSamplerConfig(
                min_edge_cents=float(payload.get("min_edge_cents", 2.0)),
                min_tau_minutes=float(payload.get("min_tau_minutes", 2.0)),
                max_tau_minutes=float(payload.get("max_tau_minutes", 14.0)),
                apply_regime_hard_gate=bool(payload.get("apply_regime_hard_gate", False)),
                price_band_min_cents=int(payload.get("price_band_min_cents", 20)),
                price_band_max_cents=int(payload.get("price_band_max_cents", 80)),
                quote_max_age_seconds=float(payload.get("quote_max_age_seconds", 3.0)),
                contracts_per_sample=int(payload.get("contracts_per_sample", 1)),
                slippage_pct=float(payload.get("slippage_pct", 1.0)),
                enable_bucket_ban_policy=bool(payload.get("enable_bucket_ban_policy", True)),
                banned_yes_tau_buckets=frozenset(payload.get("banned_yes_tau_buckets", ()) or ()),
                banned_yes_price_buckets=frozenset(payload.get("banned_yes_price_buckets", ()) or ()),
                banned_yes_probability_buckets=frozenset(payload.get("banned_yes_probability_buckets", ()) or ()),
                banned_no_price_buckets=frozenset(payload.get("banned_no_price_buckets", ()) or ()),
            )
            configs[model] = (started_at, config, "logged")
    return {model: (config, origin) for model, (_ts, config, origin) in configs.items()}


def _load_research_event_rows(run_dir: Path, *, lookback_days: int | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    research_root = run_dir / "research"
    if not research_root.exists():
        return pd.DataFrame()
    event_paths = _filter_paths_by_lookback(sorted(research_root.rglob("events.jsonl")), lookback_days)
    for events_path in event_paths:
        parts = events_path.parts
        try:
            research_index = parts.index("research")
            model = parts[research_index + 1]
        except (ValueError, IndexError):
            model = "unknown"
        raw_rows, _skipped = _iter_jsonl_tolerant(events_path)
        for event in raw_rows:
            event_type = event.get("event_type")
            if event_type not in {
                "research_sample_recorded",
                "research_sample_skipped",
                "research_sample_settled",
            }:
                continue
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                continue
            row = {
                "run_name": run_dir.name,
                "model": model,
                "event_type": event_type,
                "logged_at": _parse_timestamp(event.get("logged_at")),
                "sample_id": _clean_string(payload.get("sample_id")),
                "ticker": _clean_string(payload.get("ticker")),
                "side": _clean_string(payload.get("side")),
                "reason": _clean_string(payload.get("reason")),
                "contracts": _safe_int(payload.get("contracts")),
                "reference_price_cents": _safe_int(payload.get("reference_price_cents")),
                "max_acceptable_entry_price_cents": _safe_int(payload.get("max_acceptable_entry_price_cents")),
                "predicted_yes_probability": _safe_float(payload.get("predicted_yes_probability")),
                "predicted_no_probability": _safe_float(payload.get("predicted_no_probability")),
                "chosen_side_probability": _safe_float(payload.get("chosen_side_probability")),
                "feature_basis_market_prob": _safe_float(payload.get("feature_basis_market_prob")),
                "raw_model_edge": _safe_float(payload.get("raw_model_edge")),
                "yes_post_cost_edge": _safe_float(payload.get("yes_post_cost_edge")),
                "no_post_cost_edge": _safe_float(payload.get("no_post_cost_edge")),
                "chosen_post_cost_edge": _safe_float(payload.get("chosen_post_cost_edge")),
                "tau_minutes": _safe_float(payload.get("tau_minutes")),
                "quote_age_seconds": _safe_float(payload.get("quote_age_seconds")),
                "quote_spread_cents": _safe_int(payload.get("quote_spread_cents")),
                "buy_yes_price_cents": _safe_int(payload.get("buy_yes_price_cents")),
                "buy_no_price_cents": _safe_int(payload.get("buy_no_price_cents")),
                "yes_bid_cents": _safe_int(payload.get("yes_bid_cents")),
                "yes_ask_cents": _safe_int(payload.get("yes_ask_cents")),
                "no_bid_cents": _safe_int(payload.get("no_bid_cents")),
                "no_ask_cents": _safe_int(payload.get("no_ask_cents")),
                "regime_label": _clean_string(payload.get("regime_label")),
                "tau_bucket": _clean_string(payload.get("tau_bucket")),
                "price_bucket": _clean_string(payload.get("price_bucket")),
                "chosen_side_probability_bucket": _clean_string(payload.get("chosen_side_probability_bucket")),
                "chosen_side_edge_bucket": _clean_string(payload.get("chosen_side_edge_bucket")),
                "bucket_policy_dimension": _clean_string(payload.get("bucket_policy_dimension")),
                "bucket_policy_bucket": _clean_string(payload.get("bucket_policy_bucket")),
                "bucket_policy_side": _clean_string(payload.get("bucket_policy_side")),
                "estimated_entry_cost_dollars": _safe_float(payload.get("estimated_entry_cost_dollars")),
                "estimated_fees_dollars": _safe_float(payload.get("estimated_fees_dollars")),
                "estimated_cash_required_dollars": _safe_float(payload.get("estimated_cash_required_dollars")),
                "market_event_id": _clean_string(payload.get("market_event_id")),
                "feature_row_id": _clean_string(payload.get("feature_row_id")),
                "raw_event_id": _clean_string(payload.get("raw_event_id")),
                "received_at": _parse_timestamp(payload.get("received_at")),
                "source": _clean_string(payload.get("source")),
                "settlement_result": _clean_string(payload.get("settlement_result")),
                "is_win": payload.get("is_win"),
                "realized_pnl_dollars": _safe_float(payload.get("realized_pnl_dollars")),
                "cumulative_realized_pnl_dollars": _safe_float(payload.get("cumulative_realized_pnl_dollars")),
                "cash_required_dollars": _safe_float(payload.get("cash_required_dollars")),
            }
            rows.append(row)
    return _to_dataframe(rows)


def _merge_research_events_with_archive(
    raw_events: pd.DataFrame,
    model_outputs: pd.DataFrame,
    feature_rows: pd.DataFrame,
) -> pd.DataFrame:
    if raw_events.empty:
        return raw_events.copy()
    merged = raw_events.copy()
    if not model_outputs.empty:
        model_slice = model_outputs.copy()
        if "model_label" in model_slice.columns:
            model_slice["model"] = model_slice["model_label"].map(lambda value: _clean_string(value) or "unknown")
        if "event_time" in model_slice.columns:
            model_slice["event_time"] = model_slice["event_time"].map(_parse_timestamp)
        model_slice = model_slice.rename(
            columns={
                "feature_row_id": "archive_feature_row_id",
                "market_event_id": "archive_market_event_id",
                "model_file": "archive_model_file",
            }
        )
        keep_columns = [
            column
            for column in (
                "model",
                "archive_feature_row_id",
                "archive_market_event_id",
                "archive_model_file",
                "event_time",
            )
            if column in model_slice.columns
        ]
        if keep_columns:
            merged = merged.merge(
                model_slice[keep_columns].drop_duplicates(
                    subset=[
                        column
                        for column in ("model", "archive_feature_row_id", "archive_market_event_id")
                        if column in keep_columns
                    ]
                ),
                how="left",
                left_on=["model", "feature_row_id"],
                right_on=["model", "archive_feature_row_id"],
            )
    if not feature_rows.empty:
        feature_slice = feature_rows.copy()
        if "schema_version" not in feature_slice.columns:
            feature_slice["schema_version"] = None
        feature_slice = feature_slice.rename(columns={"feature_row_id": "archive_feature_row_id"})
        merged = merged.merge(
            feature_slice[["archive_feature_row_id", "schema_version"]].drop_duplicates("archive_feature_row_id"),
            how="left",
            on="archive_feature_row_id",
        )

    merged["model_file"] = merged.get("archive_model_file")
    merged["schema_version"] = merged.get("schema_version")
    merged["event_time"] = merged["logged_at"]

    if "sample_id" not in merged.columns:
        return merged
    recorded_or_skipped = merged.loc[
        merged["event_type"].isin({"research_sample_recorded", "research_sample_skipped"})
    ].copy()
    settlements = merged.loc[merged["event_type"] == "research_sample_settled"].copy()
    if settlements.empty:
        recorded_or_skipped["research_status"] = np.where(
            recorded_or_skipped["event_type"] == "research_sample_recorded",
            "recorded",
            "skipped",
        )
        return recorded_or_skipped

    settlement_columns = {
        "logged_at": "settled_at",
        "settlement_result": "settled_settlement_result",
        "is_win": "settled_is_win",
        "realized_pnl_dollars": "settled_realized_pnl_dollars",
        "cumulative_realized_pnl_dollars": "settled_cumulative_realized_pnl_dollars",
        "cash_required_dollars": "settled_cash_required_dollars",
    }
    settlements = settlements.rename(columns=settlement_columns)
    keep_columns = ["model", "sample_id", *settlement_columns.values()]
    recorded_or_skipped = recorded_or_skipped.merge(
        settlements[keep_columns].drop_duplicates(subset=["model", "sample_id"]),
        how="left",
        on=["model", "sample_id"],
    )
    recorded_or_skipped["research_status"] = np.where(
        recorded_or_skipped["event_type"] == "research_sample_recorded",
        "recorded",
        "skipped",
    )
    recorded_or_skipped["settlement_result"] = recorded_or_skipped["settled_settlement_result"].where(
        recorded_or_skipped["settled_settlement_result"].notna(),
        recorded_or_skipped["settlement_result"],
    )
    recorded_or_skipped["realized_pnl_dollars"] = recorded_or_skipped["settled_realized_pnl_dollars"].where(
        recorded_or_skipped["settled_realized_pnl_dollars"].notna(),
        recorded_or_skipped["realized_pnl_dollars"],
    )
    recorded_or_skipped["cumulative_realized_pnl_dollars"] = recorded_or_skipped[
        "settled_cumulative_realized_pnl_dollars"
    ].where(
        recorded_or_skipped["settled_cumulative_realized_pnl_dollars"].notna(),
        recorded_or_skipped["cumulative_realized_pnl_dollars"],
    )
    recorded_or_skipped["cash_required_dollars"] = recorded_or_skipped["settled_cash_required_dollars"].where(
        recorded_or_skipped["settled_cash_required_dollars"].notna(),
        recorded_or_skipped["cash_required_dollars"],
    )
    return recorded_or_skipped


def _build_replay_outcomes(
    run_name: str,
    model: str,
    model_outputs: pd.DataFrame,
    feature_rows: pd.DataFrame,
    config: KalshiResearchSamplerConfig,
    *,
    config_origin: str,
) -> pd.DataFrame:
    if model_outputs.empty:
        return pd.DataFrame()
    feature_schema_by_id: dict[str, str | None] = {}
    if not feature_rows.empty and "feature_row_id" in feature_rows.columns:
        feature_schema_by_id = {
            _clean_string(row.get("feature_row_id")) or "": _clean_string(row.get("schema_version"))
            for row in feature_rows.to_dict(orient="records")
        }

    seen_signatures: dict[str, set[tuple[str, str, str, str, str]]] = {}
    rows: list[dict[str, Any]] = []
    ordered = model_outputs.copy()
    if "event_time" in ordered.columns:
        ordered["event_time"] = ordered["event_time"].map(_parse_timestamp)
    if "received_at" in ordered.columns:
        ordered["received_at"] = ordered["received_at"].map(_parse_timestamp)
    ordered = ordered.sort_values(["ticker", "event_time", "received_at"], na_position="last")

    for record in ordered.to_dict(orient="records"):
        ticker = _clean_string(record.get("ticker")) or ""
        ticker_signatures = seen_signatures.setdefault(ticker, set())
        predicted_yes_probability = _safe_float(record.get("predicted_yes_probability")) or 0.0
        market_prob = _safe_float(record.get("market_prob")) or _safe_float(record.get("feature_basis_market_prob")) or 0.0
        tau_minutes = _safe_float(record.get("tau_minutes")) or 0.0
        regime_label = _clean_string(record.get("regime_label")) or "unknown"
        buy_yes_price_cents = _safe_int(record.get("buy_yes_price_cents"))
        buy_no_price_cents = _safe_int(record.get("buy_no_price_cents"))
        yes_bid_cents = _safe_int(record.get("yes_bid_cents"))
        yes_ask_cents = _safe_int(record.get("yes_ask_cents"))

        outcome = ReplayOutcome(
            status="skipped",
            reason=None,
            side=None,
            contracts=config.contracts_per_sample,
            reference_price_cents=None,
            max_acceptable_entry_price_cents=None,
            predicted_yes_probability=predicted_yes_probability,
            predicted_no_probability=1.0 - predicted_yes_probability,
            feature_basis_market_prob=market_prob,
            raw_model_edge=_safe_float(record.get("model_edge")) or _safe_float(record.get("raw_model_edge")) or 0.0,
            yes_post_cost_edge=None,
            no_post_cost_edge=None,
            chosen_post_cost_edge=None,
            estimated_entry_cost_dollars=None,
            estimated_fees_dollars=None,
            estimated_cash_required_dollars=None,
            tau_minutes=tau_minutes,
            regime_label=regime_label,
            price_bucket=None,
            tau_bucket=None,
            probability_bucket=None,
            edge_bucket=None,
            bucket_policy_dimension=None,
            bucket_policy_bucket=None,
            bucket_policy_side=None,
            feature_row_id=_clean_string(record.get("feature_row_id")) or "",
            market_event_id=_clean_string(record.get("market_event_id")) or "",
            model_file=_clean_string(record.get("model_file")),
            schema_version=feature_schema_by_id.get(_clean_string(record.get("feature_row_id")) or ""),
            event_time=_parse_timestamp(record.get("event_time")) or _parse_timestamp(record.get("received_at")) or datetime.min.replace(tzinfo=UTC),
            run_name=run_name,
            model=model,
            config_origin=config_origin,
        )

        quote_reason: str | None = None
        if (
            yes_bid_cents is None
            or yes_ask_cents is None
            or buy_yes_price_cents is None
            or buy_no_price_cents is None
        ):
            quote_reason = "missing_quote"
        elif yes_bid_cents >= yes_ask_cents:
            quote_reason = "crossed_quote"
        elif _safe_float(record.get("quote_age_seconds")) is None:
            quote_reason = "missing_quote"
        elif (_safe_float(record.get("quote_age_seconds")) or 0.0) > config.quote_max_age_seconds:
            quote_reason = "stale_quote"
        elif not (config.min_tau_minutes <= tau_minutes <= config.max_tau_minutes):
            quote_reason = "outside_tau_window"

        yes_eval = None
        no_eval = None
        if quote_reason is None and buy_yes_price_cents is not None:
            max_yes = find_max_acceptable_entry_price_cents(
                side="YES",
                predicted_yes_probability=predicted_yes_probability,
                config=config.to_signal_risk_config(),
                contracts=config.contracts_per_sample,
            )
            if config.price_band_min_cents <= buy_yes_price_cents <= config.price_band_max_cents and max_yes is not None:
                yes_edge, yes_entry, yes_fees, yes_cash = calculate_cost_metrics(
                    side="YES",
                    predicted_yes_probability=predicted_yes_probability,
                    displayed_entry_price_cents=buy_yes_price_cents,
                    contracts=config.contracts_per_sample,
                    slippage=config.slippage,
                )
                if buy_yes_price_cents <= max_yes and yes_edge + 1e-12 >= config.min_edge:
                    yes_eval = (buy_yes_price_cents, max_yes, yes_edge, yes_entry, yes_fees, yes_cash)
        if quote_reason is None and buy_no_price_cents is not None:
            max_no = find_max_acceptable_entry_price_cents(
                side="NO",
                predicted_yes_probability=predicted_yes_probability,
                config=config.to_signal_risk_config(),
                contracts=config.contracts_per_sample,
            )
            if config.price_band_min_cents <= buy_no_price_cents <= config.price_band_max_cents and max_no is not None:
                no_edge, no_entry, no_fees, no_cash = calculate_cost_metrics(
                    side="NO",
                    predicted_yes_probability=predicted_yes_probability,
                    displayed_entry_price_cents=buy_no_price_cents,
                    contracts=config.contracts_per_sample,
                    slippage=config.slippage,
                )
                if buy_no_price_cents <= max_no and no_edge + 1e-12 >= config.min_edge:
                    no_eval = (buy_no_price_cents, max_no, no_edge, no_entry, no_fees, no_cash)

        if quote_reason is not None:
            reason = quote_reason
        else:
            candidates = []
            if yes_eval is not None:
                candidates.append(("YES", *yes_eval))
            if no_eval is not None:
                candidates.append(("NO", *no_eval))
            candidates.sort(key=lambda item: (item[3], -item[1]), reverse=True)
            if not candidates:
                if (
                    buy_yes_price_cents is not None
                    and buy_no_price_cents is not None
                    and (
                        (
                            buy_yes_price_cents < config.price_band_min_cents
                            or buy_yes_price_cents > config.price_band_max_cents
                        )
                        and (
                            buy_no_price_cents < config.price_band_min_cents
                            or buy_no_price_cents > config.price_band_max_cents
                        )
                    )
                ):
                    reason = "outside_price_band"
                else:
                    reason = "insufficient_post_cost_edge"
            else:
                side, entry_price_cents, max_price, edge, entry_cost, fees, cash_required = candidates[0]
                chosen_buckets = build_chosen_side_buckets(
                    side=side,
                    tau_minutes=tau_minutes,
                    entry_price_cents=entry_price_cents,
                    predicted_yes_probability=predicted_yes_probability,
                    chosen_edge_cents=edge * 100.0,
                )
                signature = (
                    side,
                    chosen_buckets.tau_bucket,
                    chosen_buckets.price_bucket,
                    chosen_buckets.chosen_side_probability_bucket,
                    chosen_buckets.chosen_side_edge_bucket,
                )
                if signature in ticker_signatures:
                    reason = "duplicate_signature"
                else:
                    bucket_policy = evaluate_bucket_ban_policy(
                        enabled=config.enable_bucket_ban_policy,
                        side=side,
                        buckets=chosen_buckets,
                        banned_yes_tau_buckets=config.banned_yes_tau_buckets,
                        banned_yes_price_buckets=config.banned_yes_price_buckets,
                        banned_yes_probability_buckets=config.banned_yes_probability_buckets,
                        banned_no_price_buckets=config.banned_no_price_buckets,
                    )
                    if bucket_policy.is_blocked:
                        reason = "blocked_by_bucket_policy"
                    elif config.apply_regime_hard_gate and side == "YES" and regime_label == "downtrend":
                        reason = "blocked_by_regime_downtrend"
                    else:
                        reason = None
                        ticker_signatures.add(signature)
                        outcome = ReplayOutcome(
                            status="recorded",
                            reason=None,
                            side=side,
                            contracts=config.contracts_per_sample,
                            reference_price_cents=entry_price_cents,
                            max_acceptable_entry_price_cents=max_price,
                            predicted_yes_probability=predicted_yes_probability,
                            predicted_no_probability=1.0 - predicted_yes_probability,
                            feature_basis_market_prob=market_prob,
                            raw_model_edge=_safe_float(record.get("model_edge")) or _safe_float(record.get("raw_model_edge")) or 0.0,
                            yes_post_cost_edge=None if yes_eval is None else yes_eval[2],
                            no_post_cost_edge=None if no_eval is None else no_eval[2],
                            chosen_post_cost_edge=edge,
                            estimated_entry_cost_dollars=entry_cost,
                            estimated_fees_dollars=fees,
                            estimated_cash_required_dollars=cash_required,
                            tau_minutes=tau_minutes,
                            regime_label=regime_label,
                            price_bucket=chosen_buckets.price_bucket,
                            tau_bucket=chosen_buckets.tau_bucket,
                            probability_bucket=chosen_buckets.chosen_side_probability_bucket,
                            edge_bucket=chosen_buckets.chosen_side_edge_bucket,
                            bucket_policy_dimension=None,
                            bucket_policy_bucket=None,
                            bucket_policy_side=None,
                            feature_row_id=_clean_string(record.get("feature_row_id")) or "",
                            market_event_id=_clean_string(record.get("market_event_id")) or "",
                            model_file=_clean_string(record.get("model_file")),
                            schema_version=feature_schema_by_id.get(_clean_string(record.get("feature_row_id")) or ""),
                            event_time=_parse_timestamp(record.get("event_time")) or _parse_timestamp(record.get("received_at")) or datetime.min.replace(tzinfo=UTC),
                            run_name=run_name,
                            model=model,
                            config_origin=config_origin,
                        )
                if reason is not None:
                    chosen_buckets = build_chosen_side_buckets(
                        side=side,
                        tau_minutes=tau_minutes,
                        entry_price_cents=entry_price_cents,
                        predicted_yes_probability=predicted_yes_probability,
                        chosen_edge_cents=edge * 100.0,
                    )
                    bucket_policy = evaluate_bucket_ban_policy(
                        enabled=config.enable_bucket_ban_policy,
                        side=side,
                        buckets=chosen_buckets,
                        banned_yes_tau_buckets=config.banned_yes_tau_buckets,
                        banned_yes_price_buckets=config.banned_yes_price_buckets,
                        banned_yes_probability_buckets=config.banned_yes_probability_buckets,
                        banned_no_price_buckets=config.banned_no_price_buckets,
                    )
                    outcome = ReplayOutcome(
                        status="skipped",
                        reason=reason,
                        side=side,
                        contracts=config.contracts_per_sample,
                        reference_price_cents=entry_price_cents,
                        max_acceptable_entry_price_cents=max_price,
                        predicted_yes_probability=predicted_yes_probability,
                        predicted_no_probability=1.0 - predicted_yes_probability,
                        feature_basis_market_prob=market_prob,
                        raw_model_edge=_safe_float(record.get("model_edge")) or _safe_float(record.get("raw_model_edge")) or 0.0,
                        yes_post_cost_edge=None if yes_eval is None else yes_eval[2],
                        no_post_cost_edge=None if no_eval is None else no_eval[2],
                        chosen_post_cost_edge=edge,
                        estimated_entry_cost_dollars=entry_cost,
                        estimated_fees_dollars=fees,
                        estimated_cash_required_dollars=cash_required,
                        tau_minutes=tau_minutes,
                        regime_label=regime_label,
                        price_bucket=chosen_buckets.price_bucket,
                        tau_bucket=chosen_buckets.tau_bucket,
                        probability_bucket=chosen_buckets.chosen_side_probability_bucket,
                        edge_bucket=chosen_buckets.chosen_side_edge_bucket,
                        bucket_policy_dimension=None if not bucket_policy.is_blocked else bucket_policy.blocked_dimension,
                        bucket_policy_bucket=None if not bucket_policy.is_blocked else bucket_policy.blocked_bucket,
                        bucket_policy_side=None if not bucket_policy.is_blocked else bucket_policy.blocked_side,
                        feature_row_id=_clean_string(record.get("feature_row_id")) or "",
                        market_event_id=_clean_string(record.get("market_event_id")) or "",
                        model_file=_clean_string(record.get("model_file")),
                        schema_version=feature_schema_by_id.get(_clean_string(record.get("feature_row_id")) or ""),
                        event_time=_parse_timestamp(record.get("event_time")) or _parse_timestamp(record.get("received_at")) or datetime.min.replace(tzinfo=UTC),
                        run_name=run_name,
                        model=model,
                        config_origin=config_origin,
                    )
        rows.append(
            {
                "run_name": outcome.run_name,
                "model": outcome.model,
                "feature_row_id": outcome.feature_row_id,
                "market_event_id": outcome.market_event_id,
                "event_time": outcome.event_time,
                "research_source": "replayed",
                "research_status": outcome.status,
                "reason": outcome.reason,
                "side": outcome.side,
                "contracts": outcome.contracts,
                "reference_price_cents": outcome.reference_price_cents,
                "max_acceptable_entry_price_cents": outcome.max_acceptable_entry_price_cents,
                "predicted_yes_probability": outcome.predicted_yes_probability,
                "predicted_no_probability": outcome.predicted_no_probability,
                "feature_basis_market_prob": outcome.feature_basis_market_prob,
                "raw_model_edge": outcome.raw_model_edge,
                "yes_post_cost_edge": outcome.yes_post_cost_edge,
                "no_post_cost_edge": outcome.no_post_cost_edge,
                "chosen_post_cost_edge": outcome.chosen_post_cost_edge,
                "estimated_entry_cost_dollars": outcome.estimated_entry_cost_dollars,
                "estimated_fees_dollars": outcome.estimated_fees_dollars,
                "estimated_cash_required_dollars": outcome.estimated_cash_required_dollars,
                "tau_minutes": outcome.tau_minutes,
                "regime_label": outcome.regime_label,
                "price_bucket": outcome.price_bucket,
                "tau_bucket": outcome.tau_bucket,
                "chosen_side_probability_bucket": outcome.probability_bucket,
                "chosen_side_edge_bucket": outcome.edge_bucket,
                "bucket_policy_dimension": outcome.bucket_policy_dimension,
                "bucket_policy_bucket": outcome.bucket_policy_bucket,
                "bucket_policy_side": outcome.bucket_policy_side,
                "model_file": outcome.model_file,
                "schema_version": outcome.schema_version,
                "config_origin": outcome.config_origin,
            }
        )
    return _to_dataframe(rows)


def _build_research_index(
    research_events: pd.DataFrame,
) -> dict[str, dict[tuple[str, str], list[dict[str, Any]]]]:
    index = {
        "feature_row_id": {},
        "market_event_id": {},
        "timestamp": {},
    }
    if research_events.empty:
        return index
    for row in research_events.to_dict(orient="records"):
        model = _clean_string(row.get("model")) or "unknown"
        feature_row_id = _clean_string(row.get("feature_row_id"))
        market_event_id = _clean_string(row.get("market_event_id"))
        event_time = _parse_timestamp(row.get("event_time"))
        ticker = _clean_string(row.get("ticker")) or ""
        if feature_row_id:
            index["feature_row_id"].setdefault((model, feature_row_id), []).append(row)
        if market_event_id:
            index["market_event_id"].setdefault((model, market_event_id), []).append(row)
        if event_time is not None:
            index["timestamp"].setdefault((model, ticker, event_time.isoformat()), []).append(row)
    return index


def _pick_candidate(candidates: list[dict[str, Any]], decision_time: datetime | None) -> dict[str, Any] | None:
    if not candidates:
        return None
    if decision_time is None:
        return sorted(
            candidates,
            key=lambda item: (
                0 if item.get("research_status") == "recorded" else 1,
                _clean_string(item.get("run_name")) or "",
            ),
        )[0]
    return sorted(
        candidates,
        key=lambda item: (
            0 if item.get("research_status") == "recorded" else 1,
            abs(((_parse_timestamp(item.get("event_time")) or decision_time) - decision_time).total_seconds()),
            -int((_parse_timestamp(item.get("event_time")) or decision_time).timestamp()),
        ),
    )[0]


def _choose_primary_contracts(row: dict[str, Any]) -> int:
    for key in ("requested_contracts", "desired_contracts", "filled_contracts"):
        contracts = _safe_int(row.get(key))
        if contracts is not None and contracts > 0:
            return contracts
    return 1


def _choose_intended_contracts(live_row: dict[str, Any], signal_row: dict[str, Any] | None) -> int:
    signal_contracts = None if signal_row is None else _safe_int(signal_row.get("contracts"))
    if signal_contracts is not None and signal_contracts > 0:
        return signal_contracts
    return _choose_primary_contracts(live_row)


def _infer_live_fees_dollars(row: dict[str, Any]) -> float | None:
    cash_required = _safe_float(row.get("cash_required_dollars"))
    fill_price_cents = _safe_int(row.get("fill_price_cents"))
    filled_contracts = _safe_int(row.get("filled_contracts"))
    if cash_required is None or fill_price_cents is None or filled_contracts is None:
        return None
    entry_cost = (fill_price_cents / 100.0) * filled_contracts
    fees = cash_required - entry_cost
    return None if fees < -1e-9 else fees


def _settlement_payout_dollars(*, side: str, contracts: int, settlement_result: str | None) -> float:
    if settlement_result is None:
        return 0.0
    normalized = settlement_result.upper()
    if normalized not in {"YES", "NO"}:
        return 0.0
    return float(contracts if normalized == side.upper() else 0.0)


def _research_theoretical_pnl(
    *,
    side: str,
    contracts: int,
    entry_price_cents: int,
    settlement_result: str | None,
    predicted_yes_probability: float,
    slippage: float,
    use_recorded_cash: float | None = None,
) -> float:
    if use_recorded_cash is not None:
        cash_required = use_recorded_cash
    else:
        _edge, _entry_cost, _fees, cash_required = calculate_cost_metrics(
            side=side,
            predicted_yes_probability=predicted_yes_probability,
            displayed_entry_price_cents=entry_price_cents,
            contracts=contracts,
            slippage=slippage,
        )
    payout_dollars = _settlement_payout_dollars(side=side, contracts=contracts, settlement_result=settlement_result)
    return payout_dollars - cash_required


def _shadow_counterfactual_pnl(
    *,
    side: str,
    contracts: int,
    entry_price_cents: int,
    settlement_result: str | None,
) -> tuple[float, float, float]:
    entry_cost, fees, cash_required = calculate_realized_cash_metrics(
        entry_price_cents=entry_price_cents,
        contracts=contracts,
    )
    payout_dollars = _settlement_payout_dollars(side=side, contracts=contracts, settlement_result=settlement_result)
    return payout_dollars - cash_required, fees, cash_required


def _build_live_snapshot_lookup(
    signal_rows: pd.DataFrame,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if signal_rows.empty:
        return lookup
    for row in signal_rows.to_dict(orient="records"):
        decision_id = _clean_string(row.get("decision_id"))
        model = _clean_string(row.get("model")) or "unknown"
        if decision_id is None:
            continue
        lookup.setdefault((model, decision_id), []).append(row)
    return lookup


def _match_model_output(
    *,
    live_row: dict[str, Any],
    signal_row: dict[str, Any] | None,
    model_outputs: pd.DataFrame,
    feature_rows: pd.DataFrame,
    max_snapshot_lag_ms: int,
) -> tuple[str, dict[str, Any] | None]:
    if model_outputs.empty:
        return "unmatched", None
    model = _clean_string(live_row.get("model")) or "unknown"
    ticker = _clean_string(live_row.get("ticker")) or ""
    decision_time = _parse_timestamp((signal_row or {}).get("logged_at")) or _parse_timestamp(live_row.get("recorded_at"))
    direct_feature_row_id = _clean_string((signal_row or {}).get("feature_row_id")) or _clean_string(live_row.get("feature_row_id"))
    direct_market_event_id = _clean_string((signal_row or {}).get("market_event_id")) or _clean_string(live_row.get("market_event_id"))

    subset = model_outputs.copy()
    if "model_label" in subset.columns:
        subset["model"] = subset["model_label"].map(lambda value: _clean_string(value) or "unknown")
    subset["event_time"] = subset["event_time"].map(_parse_timestamp)
    subset["received_at"] = subset["received_at"].map(_parse_timestamp) if "received_at" in subset.columns else None
    subset = subset.loc[(subset["model"] == model) & (subset["ticker"].map(lambda value: _clean_string(value) or "") == ticker)].copy()
    if subset.empty:
        return "unmatched", None

    if direct_feature_row_id:
        exact = subset.loc[subset["feature_row_id"].map(lambda value: _clean_string(value) or "") == direct_feature_row_id]
        if not exact.empty:
            match = exact.sort_values(["event_time", "received_at"], na_position="last").iloc[-1].to_dict()
            return "exact_feature_row", _enrich_model_output_match(match, feature_rows)
    if direct_market_event_id:
        exact = subset.loc[
            subset["market_event_id"].map(lambda value: _clean_string(value) or "") == direct_market_event_id
        ]
        if not exact.empty:
            match = exact.sort_values(["event_time", "received_at"], na_position="last").iloc[-1].to_dict()
            return "exact_market_event", _enrich_model_output_match(match, feature_rows)
    if decision_time is None:
        return "unmatched", None

    exact_time = subset.loc[subset["event_time"] == decision_time]
    if not exact_time.empty:
        match = exact_time.sort_values(["event_time", "received_at"], na_position="last").iloc[-1].to_dict()
        quality = "exact_id" if signal_row is not None else "timestamp_fallback"
        return quality, _enrich_model_output_match(match, feature_rows)

    subset["lag_ms"] = subset["event_time"].map(
        lambda value: math.inf if value is None else (decision_time - value).total_seconds() * 1000.0
    )
    prior = subset.loc[(subset["lag_ms"] >= 0.0) & (subset["lag_ms"] <= float(max_snapshot_lag_ms))]
    if prior.empty:
        return "unmatched", None
    match = prior.sort_values(["lag_ms", "event_time"]).iloc[0].to_dict()
    return "timestamp_fallback", _enrich_model_output_match(match, feature_rows)


def _enrich_model_output_match(match: dict[str, Any], feature_rows: pd.DataFrame) -> dict[str, Any]:
    feature_row_id = _clean_string(match.get("feature_row_id")) or ""
    match["feature_row_id"] = feature_row_id
    match["market_event_id"] = _clean_string(match.get("market_event_id"))
    match["model_file"] = _clean_string(match.get("model_file"))
    if feature_rows.empty or not feature_row_id:
        match["schema_version"] = None
        return match
    subset = feature_rows.loc[
        feature_rows["feature_row_id"].map(lambda value: _clean_string(value) or "") == feature_row_id
    ]
    if subset.empty:
        match["schema_version"] = None
        return match
    feature_row = subset.iloc[-1].to_dict()
    match["schema_version"] = _clean_string(feature_row.get("schema_version"))
    return match


def _compare_version_guard(
    live_snapshot: dict[str, Any] | None,
    research_row: dict[str, Any] | None,
) -> bool:
    if live_snapshot is None or research_row is None:
        return False
    live_model_file = _clean_string(live_snapshot.get("model_file"))
    research_model_file = _clean_string(research_row.get("model_file"))
    live_schema_version = _clean_string(live_snapshot.get("schema_version"))
    research_schema_version = _clean_string(research_row.get("schema_version"))
    if live_model_file and research_model_file and live_model_file != research_model_file:
        return True
    if live_schema_version and research_schema_version and live_schema_version != research_schema_version:
        return True
    return False


def _lookup_research_row(
    *,
    live_row: dict[str, Any],
    live_snapshot: dict[str, Any] | None,
    research_index: dict[str, dict[tuple[str, str], list[dict[str, Any]]]],
    replay_index: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[str, dict[str, Any] | None, str]:
    model = _clean_string(live_row.get("model")) or "unknown"
    ticker = _clean_string(live_row.get("ticker")) or ""
    decision_time = _parse_timestamp(live_row.get("recorded_at"))
    feature_row_id = "" if live_snapshot is None else _clean_string(live_snapshot.get("feature_row_id")) or ""
    market_event_id = "" if live_snapshot is None else _clean_string(live_snapshot.get("market_event_id")) or ""

    candidate: dict[str, Any] | None = None
    match_quality = "unmatched"
    if feature_row_id:
        candidate = _pick_candidate(research_index["feature_row_id"].get((model, feature_row_id), []), decision_time)
    if candidate is None and market_event_id:
        candidate = _pick_candidate(research_index["market_event_id"].get((model, market_event_id), []), decision_time)
        if candidate is not None:
            match_quality = "exact_market_event"
    if candidate is None and decision_time is not None:
        candidate = _pick_candidate(
            research_index["timestamp"].get((model, ticker, decision_time.isoformat()), []),
            decision_time,
        )
        if candidate is not None:
            match_quality = "timestamp_fallback"
    else:
        if candidate is not None and match_quality == "unmatched":
            match_quality = "exact_feature_row"

    if candidate is not None:
        if _compare_version_guard(live_snapshot, candidate):
            return "recorded", None, "model_version_mismatch"
        return "recorded", candidate, match_quality

    if feature_row_id:
        replay_key = (_clean_string(live_row.get("run_name")) or "", model, feature_row_id)
        replay_row = replay_index.get(replay_key)
        if replay_row is not None:
            if _compare_version_guard(live_snapshot, replay_row):
                return "replayed", None, "model_version_mismatch"
            return "replayed", replay_row, "exact_feature_row"

    return "unreplayable", None, "unmatched"


def _cluster_key(row: dict[str, Any], timezone: ZoneInfo) -> str:
    market_event_id = _clean_string(row.get("market_event_id"))
    if market_event_id:
        return f"market:{market_event_id}"
    feature_row_id = _clean_string(row.get("feature_row_id"))
    if feature_row_id:
        return f"feature:{feature_row_id}"
    ticker = _clean_string(row.get("ticker")) or "unknown"
    decision_minute = _time_bucket_key(row.get("decision_time"), timezone) or "unknown"
    return f"time:{ticker}:{decision_minute}"


def _nominal_daily_gap_stats(rows: pd.DataFrame, gap_column: str) -> dict[str, Any]:
    valid = rows.loc[rows[gap_column].notna()].copy()
    if valid.empty:
        return {
            f"{gap_column}_count": 0,
            f"{gap_column}_trade_count": 0,
            f"{gap_column}_total": 0.0,
            f"{gap_column}_mean": None,
            f"{gap_column}_ci_low": None,
            f"{gap_column}_ci_high": None,
            f"{gap_column}_se": None,
            f"{gap_column}_ci_status": "no_data",
        }
    cluster_means = valid.groupby("cluster_key")[gap_column].mean()
    trade_count = int(valid.shape[0])
    total = float(valid[gap_column].sum())
    mean_gap = float(valid[gap_column].mean())
    if cluster_means.shape[0] < 2:
        return {
            f"{gap_column}_count": int(cluster_means.shape[0]),
            f"{gap_column}_trade_count": trade_count,
            f"{gap_column}_total": total,
            f"{gap_column}_mean": mean_gap,
            f"{gap_column}_ci_low": None,
            f"{gap_column}_ci_high": None,
            f"{gap_column}_se": None,
            f"{gap_column}_ci_status": "insufficient_clusters",
        }
    cluster_values = cluster_means.to_numpy(dtype=float)
    se = float(cluster_values.std(ddof=1) / math.sqrt(cluster_values.shape[0]))
    t_crit = float(stats.t.ppf(0.975, df=cluster_values.shape[0] - 1))
    mean_ci_low = mean_gap - (t_crit * se)
    mean_ci_high = mean_gap + (t_crit * se)
    return {
        f"{gap_column}_count": int(cluster_means.shape[0]),
        f"{gap_column}_trade_count": trade_count,
        f"{gap_column}_total": total,
        f"{gap_column}_mean": mean_gap,
        f"{gap_column}_ci_low": mean_ci_low * trade_count,
        f"{gap_column}_ci_high": mean_ci_high * trade_count,
        f"{gap_column}_se": se,
        f"{gap_column}_ci_status": "ok",
    }


def _bootstrap_gap_stats(
    rows: pd.DataFrame,
    gap_column: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    valid = rows.loc[rows[gap_column].notna()].copy()
    distinct_days = sorted(valid["report_day"].dropna().unique())
    trade_count = int(valid.shape[0])
    if len(distinct_days) < 5 or trade_count < 10:
        return {
            f"{gap_column}_trade_count": trade_count,
            f"{gap_column}_distinct_days": len(distinct_days),
            f"{gap_column}_mean": None if valid.empty else float(valid[gap_column].mean()),
            f"{gap_column}_total": None if valid.empty else float(valid[gap_column].sum()),
            f"{gap_column}_nominal_ci_low": None,
            f"{gap_column}_nominal_ci_high": None,
            f"{gap_column}_adjusted_ci_low": None,
            f"{gap_column}_adjusted_ci_high": None,
            f"{gap_column}_nominal_total_ci_low": None,
            f"{gap_column}_nominal_total_ci_high": None,
            f"{gap_column}_adjusted_total_ci_low": None,
            f"{gap_column}_adjusted_total_ci_high": None,
            f"{gap_column}_verdict_status": "insufficient_data",
            f"{gap_column}_direction": "neutral",
            f"{gap_column}_breach": False,
        }
    day_groups = {
        day_value: valid.loc[valid["report_day"] == day_value, gap_column].to_numpy(dtype=float)
        for day_value in distinct_days
    }
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(resamples, dtype=float)
    sampled_totals = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled_days = rng.choice(distinct_days, size=len(distinct_days), replace=True)
        sampled_values = np.concatenate([day_groups[day_value] for day_value in sampled_days])
        sampled_means[index] = float(sampled_values.mean())
        sampled_totals[index] = float(sampled_values.sum())
    nominal_low, nominal_high = np.percentile(sampled_means, [2.5, 97.5])
    adjusted_low, adjusted_high = np.percentile(sampled_means, [1.25, 98.75])
    nominal_total_low, nominal_total_high = np.percentile(sampled_totals, [2.5, 97.5])
    adjusted_total_low, adjusted_total_high = np.percentile(sampled_totals, [1.25, 98.75])
    mean_gap = float(valid[gap_column].mean())
    direction = "neutral"
    if mean_gap < 0:
        direction = "adverse"
    elif mean_gap > 0:
        direction = "favorable"
    return {
        f"{gap_column}_trade_count": trade_count,
        f"{gap_column}_distinct_days": len(distinct_days),
        f"{gap_column}_mean": mean_gap,
        f"{gap_column}_total": float(valid[gap_column].sum()),
        f"{gap_column}_nominal_ci_low": float(nominal_low),
        f"{gap_column}_nominal_ci_high": float(nominal_high),
        f"{gap_column}_adjusted_ci_low": float(adjusted_low),
        f"{gap_column}_adjusted_ci_high": float(adjusted_high),
        f"{gap_column}_nominal_total_ci_low": float(nominal_total_low),
        f"{gap_column}_nominal_total_ci_high": float(nominal_total_high),
        f"{gap_column}_adjusted_total_ci_low": float(adjusted_total_low),
        f"{gap_column}_adjusted_total_ci_high": float(adjusted_total_high),
        f"{gap_column}_verdict_status": "ok",
        f"{gap_column}_direction": direction,
        f"{gap_column}_breach": False,
    }


def _determine_breach(
    row: dict[str, Any],
    gap_column: str,
    *,
    economic_floor_per_trade: float,
) -> dict[str, Any]:
    verdict_status = row.get(f"{gap_column}_verdict_status")
    mean_gap = _safe_float(row.get(f"{gap_column}_mean"))
    adjusted_ci_low = _safe_float(row.get(f"{gap_column}_adjusted_ci_low"))
    adjusted_ci_high = _safe_float(row.get(f"{gap_column}_adjusted_ci_high"))
    if verdict_status != "ok" or mean_gap is None or adjusted_ci_low is None or adjusted_ci_high is None:
        row[f"{gap_column}_breach"] = False
        return row
    ci_excludes_zero = adjusted_ci_low > 0.0 or adjusted_ci_high < 0.0
    economically_material = abs(mean_gap) >= economic_floor_per_trade
    row[f"{gap_column}_breach"] = bool(ci_excludes_zero and economically_material)
    return row


def _window_totals(rows: pd.DataFrame, *, end_date: date, days: int) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    start_date = end_date - timedelta(days=days - 1)
    return rows.loc[(rows["report_day"] >= start_date.isoformat()) & (rows["report_day"] <= end_date.isoformat())].copy()


def _format_currency(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _format_interval(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "n/a"
    return f"[{low:.4f}, {high:.4f}]"


def _build_latest_window_summary(
    paired: pd.DataFrame,
    rolling: pd.DataFrame,
    *,
    end_date: date,
    days: int,
) -> dict[str, Any]:
    window_rows = _window_totals(paired, end_date=end_date, days=days)
    primary_rows = window_rows.loc[window_rows["primary_cohort"]].copy()
    rolling_row = rolling.loc[rolling["report_day"] == end_date.isoformat()]
    rolling_payload = {} if rolling_row.empty else rolling_row.iloc[-1].to_dict()
    return {
        "window_days": days,
        "live_realized_pnl_dollars": float(window_rows["live_realized_pnl_dollars"].sum()) if not window_rows.empty else 0.0,
        "shadow_simulated_pnl_dollars": float(window_rows["shadow_simulated_pnl_dollars"].sum()) if not window_rows.empty else 0.0,
        "research_theoretical_pnl_dollars": float(window_rows["research_theoretical_pnl_dollars"].sum()) if not window_rows.empty else 0.0,
        "primary_trade_count": int(primary_rows.shape[0]),
        "shadow_minus_research_total_dollars": None if primary_rows.empty else float(primary_rows["shadow_minus_research_pnl_dollars"].sum()),
        "live_minus_shadow_total_dollars": None if primary_rows.empty else float(primary_rows["live_minus_shadow_pnl_dollars"].sum()),
        "shadow_minus_research_mean_per_trade_dollars": None if primary_rows.empty else float(primary_rows["shadow_minus_research_pnl_dollars"].mean()),
        "live_minus_shadow_mean_per_trade_dollars": None if primary_rows.empty else float(primary_rows["live_minus_shadow_pnl_dollars"].mean()),
        "rolling_row": rolling_payload,
    }


def build_parity_artifacts(
    *,
    live_root: Path,
    research_root: Path,
    as_of_date: date,
    report_timezone: str = DEFAULT_REPORT_TIMEZONE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    models: tuple[str, ...] = (),
    live_runs: tuple[Path, ...] = (),
    research_runs: tuple[Path, ...] = (),
    shadow_runs: tuple[Path, ...] = (),
    max_snapshot_lag_ms: int = DEFAULT_MAX_SNAPSHOT_LAG_MS,
    economic_floor_per_trade: float = DEFAULT_ECONOMIC_FLOOR_PER_TRADE,
    allow_replayed_in_verdict: bool = False,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> ParityArtifacts:
    timezone = ZoneInfo(report_timezone)
    analysis_days = max(lookback_days, rolling_window_days, 30)
    discovered_live_runs = _discover_live_runs(live_root, explicit_runs=live_runs)
    discovered_research_runs = _discover_research_runs(research_root, explicit_runs=research_runs)

    selected_models = {model.lower() for model in models}

    all_live_rows: list[dict[str, Any]] = []
    all_signal_rows: list[dict[str, Any]] = []
    all_live_model_outputs: list[pd.DataFrame] = []
    all_live_feature_rows: list[pd.DataFrame] = []
    replay_rows: list[pd.DataFrame] = []

    research_event_frames: list[pd.DataFrame] = []
    research_config_by_model: dict[str, tuple[KalshiResearchSamplerConfig, str]] = {}

    for run_dir in discovered_research_runs:
        config_map = _load_research_configs(run_dir, lookback_days=analysis_days + 3)
        for model_name, config_tuple in config_map.items():
            research_config_by_model[model_name] = config_tuple
        raw_events = _load_research_event_rows(run_dir, lookback_days=analysis_days + 3)
        model_outputs = _read_stage_rows(run_dir, "model_outputs", lookback_days=analysis_days + 3)
        feature_rows = _read_stage_rows(run_dir, "feature_rows", lookback_days=analysis_days + 3)
        merged_events = _merge_research_events_with_archive(raw_events, model_outputs, feature_rows)
        if selected_models and not merged_events.empty:
            merged_events = merged_events.loc[
                merged_events["model"].map(lambda value: (_clean_string(value) or "").lower()).isin(selected_models)
            ].copy()
        if not merged_events.empty:
            research_event_frames.append(merged_events)

    for run_dir in discovered_live_runs:
        loaded = _load_live_execution_run(run_dir, lookback_days=analysis_days + 3, include_archive=True)
        canonical = _to_dataframe(loaded.canonical_rows)
        if selected_models and not canonical.empty:
            canonical = canonical.loc[
                canonical["model"].map(lambda value: (_clean_string(value) or "").lower()).isin(selected_models)
            ].copy()
        signal_rows = _load_live_signal_events(run_dir, lookback_days=analysis_days + 3)
        if selected_models and not signal_rows.empty:
            signal_rows = signal_rows.loc[
                signal_rows["model"].map(lambda value: (_clean_string(value) or "").lower()).isin(selected_models)
            ].copy()
        model_outputs = _read_stage_rows(run_dir, "model_outputs", lookback_days=analysis_days + 3)
        feature_rows = _read_stage_rows(run_dir, "feature_rows", lookback_days=analysis_days + 3)
        if not model_outputs.empty and "model_label" in model_outputs.columns:
            model_outputs["model"] = model_outputs["model_label"].map(lambda value: _clean_string(value) or "unknown")
        if not feature_rows.empty and "feature_row_id" in feature_rows.columns:
            feature_rows["feature_row_id"] = feature_rows["feature_row_id"].map(lambda value: _clean_string(value) or "")

        all_live_rows.extend(canonical.to_dict(orient="records"))
        all_signal_rows.extend(signal_rows.to_dict(orient="records"))
        if not model_outputs.empty:
            model_outputs = model_outputs.copy()
            model_outputs["run_name"] = run_dir.name
            all_live_model_outputs.append(model_outputs)
        if not feature_rows.empty:
            feature_rows = feature_rows.copy()
            feature_rows["run_name"] = run_dir.name
            all_live_feature_rows.append(feature_rows)

        if not model_outputs.empty:
            replay_feature_rows = feature_rows.copy() if not feature_rows.empty else pd.DataFrame()
            for model_name, model_rows in model_outputs.groupby("model"):
                config, origin = research_config_by_model.get(model_name, (KalshiResearchSamplerConfig(), "default"))
                replay_frame = _build_replay_outcomes(
                    run_name=run_dir.name,
                    model=model_name,
                    model_outputs=model_rows,
                    feature_rows=replay_feature_rows,
                    config=config,
                    config_origin=origin,
                )
                if not replay_frame.empty:
                    replay_rows.append(replay_frame)

    live_rows = pd.DataFrame(all_live_rows)
    if live_rows.empty:
        manifest = {
            "schema_version": PARITY_SCHEMA_VERSION,
            "as_of_date": as_of_date.isoformat(),
            "report_timezone": report_timezone,
            "lookback_days": lookback_days,
            "rolling_window_days": rolling_window_days,
            "max_snapshot_lag_ms": max_snapshot_lag_ms,
            "economic_floor_per_trade": economic_floor_per_trade,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "live_run_count": len(discovered_live_runs),
            "research_run_count": len(discovered_research_runs),
            "shadow_run_count": len(shadow_runs),
        }
        summary = {
            "schema_version": PARITY_SCHEMA_VERSION,
            "job_status": "no_live_trades",
            "snapshot_match_coverage": 0.0,
            "primary_cohort_coverage": 0.0,
            "replay_share": 0.0,
            "parity_breach": False,
            "low_coverage_warning": True,
        }
        empty = pd.DataFrame()
        return ParityArtifacts(empty, empty, empty, empty, summary, manifest)

    for column in ("recorded_at", "settled_at"):
        if column in live_rows.columns:
            live_rows[column] = live_rows[column].map(_parse_timestamp)
    live_rows = live_rows.loc[live_rows["status"] == "settled"].copy()
    if live_rows.empty:
        manifest = {
            "schema_version": PARITY_SCHEMA_VERSION,
            "as_of_date": as_of_date.isoformat(),
            "report_timezone": report_timezone,
            "lookback_days": lookback_days,
            "rolling_window_days": rolling_window_days,
            "max_snapshot_lag_ms": max_snapshot_lag_ms,
            "economic_floor_per_trade": economic_floor_per_trade,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "live_run_count": len(discovered_live_runs),
            "research_run_count": len(discovered_research_runs),
            "shadow_run_count": len(shadow_runs),
        }
        summary = {
            "schema_version": PARITY_SCHEMA_VERSION,
            "job_status": "no_settled_live_trades",
            "snapshot_match_coverage": 0.0,
            "primary_cohort_coverage": 0.0,
            "replay_share": 0.0,
            "parity_breach": False,
            "low_coverage_warning": True,
        }
        empty = pd.DataFrame()
        return ParityArtifacts(empty, empty, empty, empty, summary, manifest)

    live_rows["report_day"] = live_rows["settled_at"].map(lambda value: _local_date(value, timezone))
    live_rows = live_rows.loc[live_rows["report_day"].notna()].copy()
    live_rows = live_rows.loc[live_rows["report_day"] <= as_of_date].copy()
    earliest_date = as_of_date - timedelta(days=lookback_days - 1)
    live_rows = live_rows.loc[live_rows["report_day"] >= earliest_date].copy()

    signal_df = pd.DataFrame(all_signal_rows)
    live_model_outputs = pd.concat(all_live_model_outputs, ignore_index=True) if all_live_model_outputs else pd.DataFrame()
    live_feature_rows = pd.concat(all_live_feature_rows, ignore_index=True) if all_live_feature_rows else pd.DataFrame()
    research_events = pd.concat(research_event_frames, ignore_index=True) if research_event_frames else pd.DataFrame()
    replay_df = pd.concat(replay_rows, ignore_index=True) if replay_rows else pd.DataFrame()

    if not live_model_outputs.empty and "event_time" in live_model_outputs.columns:
        live_model_outputs["event_time"] = live_model_outputs["event_time"].map(_parse_timestamp)
        live_model_outputs["received_at"] = live_model_outputs["received_at"].map(_parse_timestamp)
    if not live_feature_rows.empty and "feature_row_id" in live_feature_rows.columns:
        live_feature_rows["feature_row_id"] = live_feature_rows["feature_row_id"].map(lambda value: _clean_string(value) or "")
    if not research_events.empty:
        research_events["event_time"] = research_events["event_time"].map(_parse_timestamp)
        research_events["settled_at"] = research_events.get("settled_at", pd.Series(dtype="datetime64[ns]")).map(_parse_timestamp)
    if not replay_df.empty:
        replay_df["event_time"] = replay_df["event_time"].map(_parse_timestamp)

    signal_lookup = _build_live_snapshot_lookup(signal_df)
    research_index = _build_research_index(research_events)
    replay_index = {
        (
            _clean_string(row.get("run_name")) or "",
            _clean_string(row.get("model")) or "unknown",
            _clean_string(row.get("feature_row_id")) or "",
        ): row
        for row in replay_df.to_dict(orient="records")
        if _clean_string(row.get("feature_row_id"))
    }

    paired_rows: list[dict[str, Any]] = []
    for live_row in live_rows.to_dict(orient="records"):
        run_name = _clean_string(live_row.get("run_name")) or ""
        model = _clean_string(live_row.get("model")) or "unknown"
        decision_id = _clean_string(live_row.get("decision_id")) or ""
        signal_candidates = signal_lookup.get((model, decision_id), [])
        signal_row = _pick_candidate(signal_candidates, _parse_timestamp(live_row.get("recorded_at")))

        run_model_outputs = live_model_outputs.loc[
            live_model_outputs["run_name"].map(lambda value: _clean_string(value) or "") == run_name
        ].copy()
        run_feature_rows = live_feature_rows.loc[
            live_feature_rows["run_name"].map(lambda value: _clean_string(value) or "") == run_name
        ].copy()
        match_quality, live_snapshot = _match_model_output(
            live_row=live_row,
            signal_row=signal_row,
            model_outputs=run_model_outputs,
            feature_rows=run_feature_rows,
            max_snapshot_lag_ms=max_snapshot_lag_ms,
        )

        research_source, research_row, research_match_quality = _lookup_research_row(
            live_row=live_row,
            live_snapshot=live_snapshot,
            research_index=research_index,
            replay_index=replay_index,
        )
        if research_match_quality == "model_version_mismatch":
            match_quality = "model_version_mismatch"

        intended_contracts = _choose_intended_contracts(live_row, signal_row)
        filled_contracts = _safe_int(live_row.get("filled_contracts")) or 0
        partial_fill_live = filled_contracts > 0 and filled_contracts < intended_contracts
        settlement_result = _clean_string(live_row.get("settlement_result"))
        live_realized_pnl = _safe_float(live_row.get("realized_pnl_dollars")) or 0.0
        live_fill_fees = _infer_live_fees_dollars(live_row)
        live_entry_price_cents = (
            _safe_int(live_row.get("fill_price_cents"))
            or _safe_int(live_row.get("reference_price_cents"))
            or _safe_int((signal_row or {}).get("reference_price_cents"))
        )

        research_action = "declined"
        research_run_name = None
        research_model_file = None
        research_schema_version = None
        research_theoretical_pnl = 0.0
        research_theoretical_pnl_native = 0.0
        research_contracts = 0
        research_side = None
        research_reason = None
        research_status = None
        research_config_origin = None

        if research_row is not None:
            research_run_name = _clean_string(research_row.get("run_name"))
            research_model_file = _clean_string(research_row.get("model_file"))
            research_schema_version = _clean_string(research_row.get("schema_version"))
            research_reason = _clean_string(research_row.get("reason"))
            research_status = _clean_string(research_row.get("research_status")) or _clean_string(research_row.get("status"))
            research_side = _clean_string(research_row.get("side"))
            research_contracts = _safe_int(research_row.get("contracts")) or 0
            research_config_origin = _clean_string(research_row.get("config_origin"))
            if research_status == "skipped" or research_side is None:
                research_action = "declined"
                research_theoretical_pnl = 0.0
                research_theoretical_pnl_native = 0.0
            else:
                research_action = "entered_same"
                if research_side != (_clean_string(live_row.get("side")) or ""):
                    research_action = "entered_different_side"
                elif research_contracts != intended_contracts:
                    research_action = "entered_different_size"
                research_reference_price_cents = _safe_int(research_row.get("reference_price_cents")) or _safe_int(
                    research_row.get("buy_yes_price_cents") if research_side == "YES" else research_row.get("buy_no_price_cents")
                )
                predicted_yes_probability = _safe_float(research_row.get("predicted_yes_probability")) or _safe_float(
                    live_snapshot.get("predicted_yes_probability") if live_snapshot else None
                ) or _safe_float(live_row.get("predicted_yes_probability")) or 0.0
                config = research_config_by_model.get(model, (KalshiResearchSamplerConfig(), "default"))[0]
                recorded_cash_required = _safe_float(research_row.get("cash_required_dollars")) or _safe_float(
                    research_row.get("estimated_cash_required_dollars")
                )
                research_theoretical_pnl_native = _research_theoretical_pnl(
                    side=research_side,
                    contracts=max(1, research_contracts),
                    entry_price_cents=research_reference_price_cents or 0,
                    settlement_result=settlement_result,
                    predicted_yes_probability=predicted_yes_probability,
                    slippage=config.slippage,
                    use_recorded_cash=(
                        recorded_cash_required if research_source == "recorded" and research_contracts == max(1, research_contracts) else None
                    ),
                )
                research_theoretical_pnl = _research_theoretical_pnl(
                    side=research_side,
                    contracts=intended_contracts,
                    entry_price_cents=research_reference_price_cents or 0,
                    settlement_result=settlement_result,
                    predicted_yes_probability=predicted_yes_probability,
                    slippage=config.slippage,
                    use_recorded_cash=None,
                )

        shadow_entry_price_cents = (
            _safe_int((signal_row or {}).get("buy_yes_price_cents"))
            if (_clean_string(live_row.get("side")) or "") == "YES"
            else _safe_int((signal_row or {}).get("buy_no_price_cents"))
        )
        if shadow_entry_price_cents is None:
            shadow_entry_price_cents = _safe_int(live_row.get("buy_yes_price_cents")) if (_clean_string(live_row.get("side")) or "") == "YES" else _safe_int(live_row.get("buy_no_price_cents"))
        if shadow_entry_price_cents is None:
            shadow_entry_price_cents = _safe_int(live_row.get("reference_price_cents")) or _safe_int((signal_row or {}).get("reference_price_cents"))

        shadow_contracts = intended_contracts
        shadow_size_assumed = True
        feed_top_book_contracts = _safe_int(live_row.get("feed_top_book_contracts")) or _safe_int((signal_row or {}).get("feed_top_book_contracts"))
        if feed_top_book_contracts is not None:
            shadow_contracts = min(shadow_contracts, max(0, feed_top_book_contracts))
            shadow_size_assumed = False
        if shadow_entry_price_cents is not None and shadow_contracts > 0:
            shadow_pnl, shadow_fees, shadow_cash_required = _shadow_counterfactual_pnl(
                side=_clean_string(live_row.get("side")) or "YES",
                contracts=shadow_contracts,
                entry_price_cents=shadow_entry_price_cents,
                settlement_result=settlement_result,
            )
        else:
            shadow_pnl, shadow_fees, shadow_cash_required = (None, None, None)

        primary_cohort = (
            research_row is not None
            and research_action == "entered_same"
            and shadow_pnl is not None
            and match_quality not in {"unmatched", "model_version_mismatch"}
            and research_source in {"recorded"} | ({"replayed"} if allow_replayed_in_verdict else set())
        )

        live_decision_time = _parse_timestamp((signal_row or {}).get("logged_at")) or _parse_timestamp(live_row.get("recorded_at"))
        feature_row_id = None if live_snapshot is None else _clean_string(live_snapshot.get("feature_row_id"))
        market_event_id = None if live_snapshot is None else _clean_string(live_snapshot.get("market_event_id"))
        model_file = None if live_snapshot is None else _clean_string(live_snapshot.get("model_file"))
        schema_version = None if live_snapshot is None else _clean_string(live_snapshot.get("schema_version"))
        report_day = _local_date(live_row.get("settled_at"), timezone)

        row = {
            "schema_version": PARITY_SCHEMA_VERSION,
            "run_name": run_name,
            "model": model,
            "ticker": _clean_string(live_row.get("ticker")),
            "decision_id": decision_id,
            "feature_row_id": feature_row_id,
            "market_event_id": market_event_id,
            "model_file": model_file,
            "feature_schema_version": schema_version,
            "decision_time": live_decision_time,
            "settlement_time": _parse_timestamp(live_row.get("settled_at")),
            "report_day": None if report_day is None else report_day.isoformat(),
            "cluster_key": _cluster_key(
                {
                    "market_event_id": market_event_id,
                    "feature_row_id": feature_row_id,
                    "ticker": _clean_string(live_row.get("ticker")),
                    "decision_time": live_decision_time,
                },
                timezone,
            ),
            "match_quality": match_quality,
            "live_side": _clean_string(live_row.get("side")),
            "live_requested_contracts": intended_contracts,
            "live_filled_contracts": filled_contracts,
            "live_partial_fill": partial_fill_live,
            "live_reference_price_cents": _safe_int((signal_row or {}).get("reference_price_cents")) or _safe_int(live_row.get("reference_price_cents")),
            "live_fill_price_cents": live_entry_price_cents,
            "live_fees_dollars": live_fill_fees,
            "live_cash_required_dollars": _safe_float(live_row.get("cash_required_dollars")),
            "live_realized_pnl_dollars": live_realized_pnl,
            "live_settlement_result": settlement_result,
            "research_source": research_source,
            "research_run_name": research_run_name,
            "research_model_file": research_model_file,
            "research_feature_schema_version": research_schema_version,
            "research_status": research_status,
            "research_reason": research_reason,
            "research_action": research_action,
            "research_side": research_side,
            "research_contracts": research_contracts,
            "research_theoretical_pnl_native_dollars": research_theoretical_pnl_native,
            "research_theoretical_pnl_dollars": research_theoretical_pnl,
            "research_config_origin": research_config_origin,
            "shadow_source": "synthetic",
            "shadow_side": _clean_string(live_row.get("side")),
            "shadow_contracts": shadow_contracts,
            "shadow_size_assumed": shadow_size_assumed,
            "shadow_entry_price_cents": shadow_entry_price_cents,
            "shadow_fees_dollars": shadow_fees,
            "shadow_cash_required_dollars": shadow_cash_required,
            "shadow_simulated_pnl_dollars": shadow_pnl,
            "shadow_minus_research_pnl_dollars": None if shadow_pnl is None else shadow_pnl - research_theoretical_pnl,
            "live_minus_shadow_pnl_dollars": None if shadow_pnl is None else live_realized_pnl - shadow_pnl,
            "live_minus_research_pnl_dollars": live_realized_pnl - research_theoretical_pnl,
            "primary_cohort": bool(primary_cohort),
            "shadow_audit_found": False,
            "shadow_audit_realized_pnl_dollars": None,
            "shadow_audit_run_name": None,
        }
        paired_rows.append(row)

    paired = pd.DataFrame(paired_rows)
    if paired.empty:
        manifest = {
            "schema_version": PARITY_SCHEMA_VERSION,
            "as_of_date": as_of_date.isoformat(),
            "report_timezone": report_timezone,
            "lookback_days": lookback_days,
            "rolling_window_days": rolling_window_days,
            "max_snapshot_lag_ms": max_snapshot_lag_ms,
            "economic_floor_per_trade": economic_floor_per_trade,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "live_run_count": len(discovered_live_runs),
            "research_run_count": len(discovered_research_runs),
            "shadow_run_count": len(shadow_runs),
        }
        summary = {
            "schema_version": PARITY_SCHEMA_VERSION,
            "job_status": "no_paired_rows",
            "snapshot_match_coverage": 0.0,
            "primary_cohort_coverage": 0.0,
            "replay_share": 0.0,
            "parity_breach": False,
            "low_coverage_warning": True,
        }
        empty = pd.DataFrame()
        return ParityArtifacts(empty, empty, empty, empty, summary, manifest)

    paired["decision_time"] = paired["decision_time"].map(_parse_timestamp)
    paired["settlement_time"] = paired["settlement_time"].map(_parse_timestamp)

    daily_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for report_day, day_rows in paired.groupby("report_day"):
        base_row = {
            "schema_version": PARITY_SCHEMA_VERSION,
            "report_day": report_day,
            "settled_live_trade_count": int(day_rows.shape[0]),
            "matched_snapshot_count": int(day_rows.loc[day_rows["match_quality"] != "unmatched"].shape[0]),
            "primary_cohort_trade_count": int(day_rows.loc[day_rows["primary_cohort"]].shape[0]),
            "research_theoretical_pnl_dollars": float(day_rows["research_theoretical_pnl_dollars"].sum()),
            "shadow_simulated_pnl_dollars": float(day_rows["shadow_simulated_pnl_dollars"].fillna(0.0).sum()),
            "live_realized_pnl_dollars": float(day_rows["live_realized_pnl_dollars"].sum()),
        }
        primary_day_rows = day_rows.loc[day_rows["primary_cohort"]].copy()
        for gap_column in PRIMARY_GAP_COLUMNS:
            base_row.update(_nominal_daily_gap_stats(primary_day_rows, gap_column))
        daily_rows.append(base_row)

        for segment_name in PRIMARY_SEGMENTS:
            if segment_name == "partial_fill_live":
                segment_slice = day_rows.loc[day_rows["live_partial_fill"]].copy()
            else:
                segment_slice = day_rows.loc[day_rows["research_action"] == segment_name].copy()
            if segment_slice.empty:
                continue
            segment_rows.append(
                {
                    "schema_version": PARITY_SCHEMA_VERSION,
                    "report_day": report_day,
                    "segment": segment_name,
                    "trade_count": int(segment_slice.shape[0]),
                    "live_realized_pnl_dollars": float(segment_slice["live_realized_pnl_dollars"].sum()),
                    "shadow_simulated_pnl_dollars": float(segment_slice["shadow_simulated_pnl_dollars"].fillna(0.0).sum()),
                    "research_theoretical_pnl_dollars": float(segment_slice["research_theoretical_pnl_dollars"].sum()),
                    "shadow_minus_research_pnl_dollars": float(segment_slice["shadow_minus_research_pnl_dollars"].fillna(0.0).sum()),
                    "live_minus_shadow_pnl_dollars": float(segment_slice["live_minus_shadow_pnl_dollars"].fillna(0.0).sum()),
                    "live_minus_research_pnl_dollars": float(segment_slice["live_minus_research_pnl_dollars"].fillna(0.0).sum()),
                }
            )

    daily_rollup = pd.DataFrame(daily_rows).sort_values("report_day")
    daily_segment_rollup = pd.DataFrame(segment_rows).sort_values(["report_day", "segment"]) if segment_rows else pd.DataFrame()

    rolling_rows: list[dict[str, Any]] = []
    report_days = sorted(daily_rollup["report_day"].tolist())
    for report_day_str in report_days:
        report_day = date.fromisoformat(report_day_str)
        window_start = report_day - timedelta(days=rolling_window_days - 1)
        window_rows = paired.loc[(paired["report_day"] >= window_start.isoformat()) & (paired["report_day"] <= report_day_str)].copy()
        primary_window = window_rows.loc[window_rows["primary_cohort"]].copy()
        row: dict[str, Any] = {
            "schema_version": PARITY_SCHEMA_VERSION,
            "report_day": report_day_str,
            "window_start_day": window_start.isoformat(),
            "window_end_day": report_day_str,
            "window_trade_count": int(primary_window.shape[0]),
            "window_distinct_days": int(primary_window["report_day"].nunique()) if not primary_window.empty else 0,
            "window_live_realized_pnl_dollars": float(window_rows["live_realized_pnl_dollars"].sum()),
            "window_shadow_simulated_pnl_dollars": float(window_rows["shadow_simulated_pnl_dollars"].fillna(0.0).sum()),
            "window_research_theoretical_pnl_dollars": float(window_rows["research_theoretical_pnl_dollars"].sum()),
        }
        for gap_index, gap_column in enumerate(PRIMARY_GAP_COLUMNS):
            row.update(
                _bootstrap_gap_stats(
                    primary_window,
                    gap_column,
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed + gap_index,
                )
            )
            row = _determine_breach(row, gap_column, economic_floor_per_trade=economic_floor_per_trade)

        live_shadow_breach = bool(row.get("live_minus_shadow_pnl_dollars_breach"))
        shadow_research_breach = bool(row.get("shadow_minus_research_pnl_dollars_breach"))
        if live_shadow_breach and shadow_research_breach:
            leak_label = "both_mixed"
        elif live_shadow_breach:
            direction = row.get("live_minus_shadow_pnl_dollars_direction") or "neutral"
            leak_label = (
                "live_execution_adverse"
                if direction == "adverse"
                else "live_execution_favorable"
                if direction == "favorable"
                else "none"
            )
        elif shadow_research_breach:
            direction = row.get("shadow_minus_research_pnl_dollars_direction") or "neutral"
            leak_label = (
                "shadow_vs_research_adverse"
                if direction == "adverse"
                else "shadow_vs_research_favorable"
                if direction == "favorable"
                else "none"
            )
        else:
            leak_label = "none"
        row["parity_breach"] = bool(live_shadow_breach or shadow_research_breach)
        row["leak_label"] = leak_label
        rolling_rows.append(row)

    rolling_rollup = pd.DataFrame(rolling_rows).sort_values("report_day")

    snapshot_match_coverage = float((paired["match_quality"] != "unmatched").mean()) if not paired.empty else 0.0
    primary_cohort_coverage = float(paired["primary_cohort"].mean()) if not paired.empty else 0.0
    replay_share = (
        float((paired["research_source"] == "replayed").mean())
        if not paired.empty
        else 0.0
    )
    low_coverage_warning = bool(
        snapshot_match_coverage < DEFAULT_LOW_SNAPSHOT_MATCH_COVERAGE
        or primary_cohort_coverage < DEFAULT_LOW_PRIMARY_COHORT_COVERAGE
    )
    latest_14d = _build_latest_window_summary(paired, rolling_rollup, end_date=as_of_date, days=14)
    latest_30d = _build_latest_window_summary(paired, rolling_rollup, end_date=as_of_date, days=30)
    latest_rolling_row = rolling_rollup.loc[rolling_rollup["report_day"] == as_of_date.isoformat()]
    latest_rolling = {} if latest_rolling_row.empty else latest_rolling_row.iloc[-1].to_dict()

    summary = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of_date": as_of_date.isoformat(),
        "report_timezone": report_timezone,
        "settled_live_trade_count": int(paired.shape[0]),
        "matched_snapshot_count": int(paired.loc[paired["match_quality"] != "unmatched"].shape[0]),
        "primary_cohort_trade_count": int(paired.loc[paired["primary_cohort"]].shape[0]),
        "unmatched_trade_count": int(paired.loc[paired["match_quality"] == "unmatched"].shape[0]),
        "model_version_mismatch_count": int(paired.loc[paired["match_quality"] == "model_version_mismatch"].shape[0]),
        "replayed_trade_count": int(paired.loc[paired["research_source"] == "replayed"].shape[0]),
        "snapshot_match_coverage": snapshot_match_coverage,
        "primary_cohort_coverage": primary_cohort_coverage,
        "replay_share": replay_share,
        "parity_breach": bool(latest_rolling.get("parity_breach", False)),
        "low_coverage_warning": low_coverage_warning,
        "job_status": "success" if not low_coverage_warning else "warning",
        "latest_14d": latest_14d,
        "latest_30d": latest_30d,
        "latest_rolling": latest_rolling,
    }

    manifest = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of_date": as_of_date.isoformat(),
        "report_timezone": report_timezone,
        "lookback_days": lookback_days,
        "rolling_window_days": rolling_window_days,
        "max_snapshot_lag_ms": max_snapshot_lag_ms,
        "economic_floor_per_trade": economic_floor_per_trade,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "allow_replayed_in_verdict": allow_replayed_in_verdict,
        "coverage_thresholds": {
            "snapshot_match_coverage": DEFAULT_LOW_SNAPSHOT_MATCH_COVERAGE,
            "primary_cohort_coverage": DEFAULT_LOW_PRIMARY_COHORT_COVERAGE,
        },
        "live_run_count": len(discovered_live_runs),
        "research_run_count": len(discovered_research_runs),
        "shadow_run_count": len(shadow_runs),
        "live_runs": [str(path) for path in discovered_live_runs],
        "research_runs": [str(path) for path in discovered_research_runs],
        "shadow_runs": [str(path) for path in shadow_runs],
        "models": sorted(selected_models),
    }

    return ParityArtifacts(
        paired_trades=paired.sort_values(["settlement_time", "model", "ticker", "decision_id"]),
        daily_rollup=daily_rollup,
        daily_segment_rollup=daily_segment_rollup,
        rolling_rollup=rolling_rollup,
        summary=summary,
        manifest=manifest,
    )


def render_parity_report_markdown(artifacts: ParityArtifacts) -> str:
    summary = artifacts.summary
    latest_14d = summary.get("latest_14d", {})
    latest_30d = summary.get("latest_30d", {})
    latest_rolling = summary.get("latest_rolling", {})
    lines = [
        "# Kalshi Nightly Parity Report",
        "",
        "## Summary",
        f"- Schema version: `{PARITY_SCHEMA_VERSION}`",
        f"- Job status: `{summary.get('job_status', 'unknown')}`",
        f"- Snapshot match coverage: `{summary.get('snapshot_match_coverage', 0.0):.2%}`",
        f"- Primary cohort coverage: `{summary.get('primary_cohort_coverage', 0.0):.2%}`",
        f"- Replay share: `{summary.get('replay_share', 0.0):.2%}`",
        f"- Latest leak label: `{latest_rolling.get('leak_label', 'none')}`",
        "",
        "## 14-Day Window",
        f"- Research theoretical PnL: `{_format_currency(_safe_float(latest_14d.get('research_theoretical_pnl_dollars')))} `",
        f"- Shadow simulated PnL: `{_format_currency(_safe_float(latest_14d.get('shadow_simulated_pnl_dollars')))} `",
        f"- Live realized PnL: `{_format_currency(_safe_float(latest_14d.get('live_realized_pnl_dollars')))} `",
        f"- Shadow minus research total: `{_format_currency(_safe_float(latest_14d.get('shadow_minus_research_total_dollars')))} `",
        f"- Shadow minus research mean/trade: `{_format_currency(_safe_float(latest_14d.get('shadow_minus_research_mean_per_trade_dollars')))} `",
        f"- Live minus shadow total: `{_format_currency(_safe_float(latest_14d.get('live_minus_shadow_total_dollars')))} `",
        f"- Live minus shadow mean/trade: `{_format_currency(_safe_float(latest_14d.get('live_minus_shadow_mean_per_trade_dollars')))} `",
        f"- Nominal 95% CI shadow minus research: `{_format_interval(_safe_float(latest_rolling.get('shadow_minus_research_pnl_dollars_nominal_ci_low')), _safe_float(latest_rolling.get('shadow_minus_research_pnl_dollars_nominal_ci_high')))} `",
        f"- Nominal 95% CI live minus shadow: `{_format_interval(_safe_float(latest_rolling.get('live_minus_shadow_pnl_dollars_nominal_ci_low')), _safe_float(latest_rolling.get('live_minus_shadow_pnl_dollars_nominal_ci_high')))} `",
        "",
        "## 30-Day Window",
        f"- Research theoretical PnL: `{_format_currency(_safe_float(latest_30d.get('research_theoretical_pnl_dollars')))} `",
        f"- Shadow simulated PnL: `{_format_currency(_safe_float(latest_30d.get('shadow_simulated_pnl_dollars')))} `",
        f"- Live realized PnL: `{_format_currency(_safe_float(latest_30d.get('live_realized_pnl_dollars')))} `",
        "",
        "## Drift Segments",
        f"- Declined rows: `{int(artifacts.paired_trades.loc[artifacts.paired_trades['research_action'] == 'declined'].shape[0])}`",
        f"- Different-side rows: `{int(artifacts.paired_trades.loc[artifacts.paired_trades['research_action'] == 'entered_different_side'].shape[0])}`",
        f"- Different-size rows: `{int(artifacts.paired_trades.loc[artifacts.paired_trades['research_action'] == 'entered_different_size'].shape[0])}`",
        f"- Partial-fill live rows: `{int(artifacts.paired_trades.loc[artifacts.paired_trades['live_partial_fill']].shape[0])}`",
        "",
        "## Notes",
        "- Nominal 95% intervals are reported for visibility; the parity breach verdict uses Bonferroni-adjusted rolling intervals across the two primary gap families.",
        "- No additional multiple-comparison adjustment is applied across overlapping rolling windows; treat the verdict as descriptive across time.",
    ]
    return "\n".join(lines) + "\n"


def save_parity_plot(
    artifacts: ParityArtifacts,
    output_path: Path,
    *,
    economic_floor_per_trade: float,
) -> None:
    rolling = artifacts.rolling_rollup.copy()
    if rolling.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No rolling parity data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return
    rolling["report_day_dt"] = pd.to_datetime(rolling["report_day"])
    for column in (
        "shadow_minus_research_pnl_dollars_total",
        "live_minus_shadow_pnl_dollars_total",
        "shadow_minus_research_pnl_dollars_nominal_total_ci_low",
        "shadow_minus_research_pnl_dollars_nominal_total_ci_high",
        "live_minus_shadow_pnl_dollars_nominal_total_ci_low",
        "live_minus_shadow_pnl_dollars_nominal_total_ci_high",
        "shadow_minus_research_pnl_dollars_mean",
        "live_minus_shadow_pnl_dollars_mean",
        "cumulative_shadow_minus_research",
        "cumulative_live_minus_shadow",
    ):
        if column in rolling.columns:
            rolling[column] = pd.to_numeric(rolling[column], errors="coerce")
    rolling["cumulative_shadow_minus_research"] = rolling["shadow_minus_research_pnl_dollars_total"].fillna(0.0).cumsum()
    rolling["cumulative_live_minus_shadow"] = rolling["live_minus_shadow_pnl_dollars_total"].fillna(0.0).cumsum()

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ax0.plot(
        rolling["report_day_dt"],
        rolling["cumulative_shadow_minus_research"],
        label="Cumulative shadow - research",
        linewidth=2.0,
    )
    ax0.plot(
        rolling["report_day_dt"],
        rolling["cumulative_live_minus_shadow"],
        label="Cumulative live - shadow",
        linewidth=2.0,
    )
    if {"shadow_minus_research_pnl_dollars_nominal_total_ci_low", "shadow_minus_research_pnl_dollars_nominal_total_ci_high"} <= set(
        rolling.columns
    ):
        ax0.fill_between(
            rolling["report_day_dt"],
            rolling["shadow_minus_research_pnl_dollars_nominal_total_ci_low"].to_numpy(dtype=float),
            rolling["shadow_minus_research_pnl_dollars_nominal_total_ci_high"].to_numpy(dtype=float),
            alpha=0.15,
            label="Shadow - research 95% CI",
        )
    if {"live_minus_shadow_pnl_dollars_nominal_total_ci_low", "live_minus_shadow_pnl_dollars_nominal_total_ci_high"} <= set(rolling.columns):
        ax0.fill_between(
            rolling["report_day_dt"],
            rolling["live_minus_shadow_pnl_dollars_nominal_total_ci_low"].to_numpy(dtype=float),
            rolling["live_minus_shadow_pnl_dollars_nominal_total_ci_high"].to_numpy(dtype=float),
            alpha=0.15,
            label="Live - shadow 95% CI",
        )
    ax0.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax0.set_ylabel("Gap dollars")
    ax0.set_title("Rolling cumulative parity gaps")
    ax0.legend(loc="best")

    ax1.plot(
        rolling["report_day_dt"],
        rolling["shadow_minus_research_pnl_dollars_mean"],
        label="Shadow - research mean/trade",
        linewidth=2.0,
    )
    ax1.plot(
        rolling["report_day_dt"],
        rolling["live_minus_shadow_pnl_dollars_mean"],
        label="Live - shadow mean/trade",
        linewidth=2.0,
    )
    ax1.axhline(economic_floor_per_trade, color="red", linestyle="--", alpha=0.5)
    ax1.axhline(-economic_floor_per_trade, color="red", linestyle="--", alpha=0.5)
    breach_mask = rolling["parity_breach"].fillna(False)
    ax1.scatter(
        rolling.loc[breach_mask, "report_day_dt"],
        rolling.loc[breach_mask, "live_minus_shadow_pnl_dollars_mean"],
        color="red",
        marker="x",
        label="Adjusted breach",
    )
    ax1.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax1.set_ylabel("Mean gap / trade")
    ax1.set_title("Rolling mean parity gaps and breach markers")
    ax1.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_parity_artifacts(
    artifacts: ParityArtifacts,
    *,
    output_root: Path,
    as_of_date: date,
    markdown_text: str,
    economic_floor_per_trade: float,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / as_of_date.isoformat()
    temp_dir = output_root / f".tmp-{as_of_date.isoformat()}-{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    artifacts.paired_trades.to_parquet(temp_dir / "paired_trades.parquet", index=False)
    artifacts.paired_trades.to_csv(temp_dir / "paired_trades.csv", index=False)
    artifacts.daily_rollup.to_csv(temp_dir / "daily_rollup.csv", index=False)
    artifacts.daily_segment_rollup.to_csv(temp_dir / "daily_segment_rollup.csv", index=False)
    artifacts.rolling_rollup.to_csv(temp_dir / "rolling_rollup.csv", index=False)
    (temp_dir / "summary.json").write_text(json.dumps(artifacts.summary, indent=2, sort_keys=True), encoding="utf-8")
    (temp_dir / "manifest.json").write_text(json.dumps(artifacts.manifest, indent=2, sort_keys=True), encoding="utf-8")
    (temp_dir / "parity_report.md").write_text(markdown_text, encoding="utf-8")
    save_parity_plot(
        artifacts,
        temp_dir / "parity_gaps.png",
        economic_floor_per_trade=economic_floor_per_trade,
    )

    backup_dir = output_root / f".backup-{as_of_date.isoformat()}-{uuid.uuid4().hex[:8]}"
    if final_dir.exists():
        final_dir.replace(backup_dir)
    temp_dir.replace(final_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)

    latest_dir = output_root / "latest"
    latest_temp_dir = output_root / f".latest-tmp-{uuid.uuid4().hex[:8]}"
    shutil.copytree(final_dir, latest_temp_dir)
    latest_backup_dir = output_root / f".latest-backup-{uuid.uuid4().hex[:8]}"
    if latest_dir.exists():
        latest_dir.replace(latest_backup_dir)
    latest_temp_dir.replace(latest_dir)
    if latest_backup_dir.exists():
        shutil.rmtree(latest_backup_dir, ignore_errors=True)
    return final_dir
