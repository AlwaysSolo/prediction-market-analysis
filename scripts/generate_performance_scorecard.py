from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.kalshi.performance_analysis import _load_live_execution_run, _load_live_research_run


DEFAULT_LIVE_ROOT = REPO_ROOT / "output" / "live"
DEFAULT_LIVE_RESEARCH_ROOT = REPO_ROOT / "output" / "live_research"
DEFAULT_REPORT_ROOT = REPO_ROOT / "artifacts" / "kalshi" / "performance_reports"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "kalshi" / "performance_scorecard"
DEFAULT_WINDOWS = (7, 30, 90)
DEFAULT_TIMEZONE = "America/Los_Angeles"
MINUTES_PER_YEAR = 365.0 * 24.0 * 60.0

FILL_REALISM_FIX_COMMIT = "a171210"
FILL_REALISM_FIX_AT = datetime.fromisoformat("2026-04-10T11:48:58-07:00")

TIMESTAMP_COLUMNS = (
    "recorded_at",
    "settled_at",
    "submitted_at",
    "execution_terminal_at",
)
FLOAT_COLUMNS = (
    "realized_pnl_dollars",
    "cumulative_realized_pnl_dollars",
    "cash_required_dollars",
    "chosen_post_cost_edge_cents",
    "expected_value_dollars",
    "fill_price_cents",
    "quote_age_seconds",
    "tau_minutes",
)
INT_COLUMNS = (
    "filled_contracts",
    "requested_contracts",
    "remaining_contracts",
    "quote_spread_cents",
    "reference_price_cents",
)
BOOL_COLUMNS = (
    "is_win",
    "same_as_offline_rule",
    "one_sided_quote",
)
BUCKET_COLUMNS = (
    "tau_bucket",
    "price_bucket",
    "probability_bucket",
    "edge_bucket",
    "bucket_policy_bucket",
)

REPORT_LINE_PATTERNS = {
    "source_type": re.compile(r"^- Source type: `(?P<value>[^`]+)`$"),
    "analyzed_path": re.compile(r"^- Analyzed path: `(?P<value>[^`]+)`$"),
    "recorded_count": re.compile(r"^- Recorded rows: `(?P<value>[\d,]+)`$"),
    "settled_count": re.compile(r"^- Settled rows: `(?P<value>[\d,]+)`$"),
    "open_count": re.compile(r"^- Open rows: `(?P<value>[\d,]+)`$"),
    "aggregate_net_pnl_dollars": re.compile(r"^- Aggregate settled PnL: `\$(?P<value>[-\d,\.]+)`$"),
}


@dataclass
class RunCandidate:
    run_key: str
    run_name: str
    source_type: str
    canonical_rows: pd.DataFrame
    execution_rows: pd.DataFrame
    data_source: str
    raw_run_dir: Path | None = None
    report_dir: Path | None = None
    report_path: Path | None = None
    report_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    report_variants_count: int = 1


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


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
    return parsed


def _parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _iter_jsonl_rows(path: Path, *, max_lines: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_lines is not None and index >= max_lines:
                break
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _resolve_repo_path(path_text: str | None) -> Path | None:
    cleaned = _clean_string(path_text)
    if cleaned is None:
        return None
    candidate = Path(cleaned)
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def _normalize_run_key(raw_run_dir: Path | None, *, source_type: str, run_name: str) -> str:
    if raw_run_dir is not None:
        return str(raw_run_dir.resolve())
    return f"{source_type}:{run_name}"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _parse_report_markdown(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {}
    text = report_path.read_text(encoding="utf-8")
    parsed: dict[str, Any] = {"report_text": text}
    heading = next((line for line in text.splitlines() if line.startswith("# ")), None)
    if heading is not None:
        parsed["run_name"] = heading.replace("# Kalshi Performance Analysis:", "").strip()
    for line in text.splitlines():
        stripped = line.strip()
        for field_name, pattern in REPORT_LINE_PATTERNS.items():
            match = pattern.match(stripped)
            if match:
                parsed[field_name] = match.group("value")
    return parsed


def _coerce_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    coerced = frame.copy()
    for column in FLOAT_COLUMNS:
        if column in coerced.columns:
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    for column in INT_COLUMNS:
        if column in coerced.columns:
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    for column in TIMESTAMP_COLUMNS:
        if column in coerced.columns:
            coerced[column] = pd.to_datetime(coerced[column], utc=True, errors="coerce")
    for column in BOOL_COLUMNS:
        if column in coerced.columns:
            coerced[column] = coerced[column].map(_parse_bool)
    for column in ("status", "model", "run_name", "source_type", "regime_label"):
        if column in coerced.columns:
            coerced[column] = coerced[column].map(lambda value: _clean_string(value) or "")
    return coerced


def _choose_fill_value(existing: pd.Series, incoming: pd.Series) -> pd.Series:
    return existing.where(existing.notna() & (existing != ""), incoming)


def _enrich_canonical_rows(canonical_rows: pd.DataFrame, execution_rows: pd.DataFrame) -> pd.DataFrame:
    if canonical_rows.empty or execution_rows.empty:
        return canonical_rows.copy()
    if "decision_id" not in canonical_rows.columns or "decision_id" not in execution_rows.columns:
        return canonical_rows.copy()

    enriched = canonical_rows.copy()
    merge_columns = [
        column
        for column in execution_rows.columns
        if column not in {"run_name", "source_type", "model", "decision_id"}
    ]
    supplemental = execution_rows[["model", "decision_id", *merge_columns]].drop_duplicates(
        subset=["model", "decision_id"],
        keep="last",
    )
    merged = enriched.merge(
        supplemental,
        on=["model", "decision_id"],
        how="left",
        suffixes=("", "__execution"),
    )
    for column in merge_columns:
        execution_column = f"{column}__execution"
        if execution_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[execution_column]
        else:
            merged[column] = _choose_fill_value(merged[column], merged[execution_column])
        merged.drop(columns=[execution_column], inplace=True)
    return merged


def _load_standalone_settlements(run_dir: Path, *, source_type: str) -> pd.DataFrame:
    identifier_column = "sample_id" if source_type == "live_research" else "decision_id"
    rows: list[dict[str, Any]] = []
    for settlement_path in sorted(run_dir.rglob("settlements.jsonl")):
        for event in _iter_jsonl_rows(settlement_path):
            payload = event.get("payload")
            source = payload if isinstance(payload, dict) else event
            identifier = _clean_string(source.get(identifier_column))
            if identifier is None:
                continue
            rows.append(
                {
                    "model": _clean_string(source.get("model")) or "",
                    identifier_column: identifier,
                    "ticker": _clean_string(source.get("ticker")) or "",
                    "side": _clean_string(source.get("side")) or "",
                    "settled_at": _parse_timestamp(
                        source.get("settled_at") or event.get("logged_at") or source.get("logged_at")
                    ),
                    "status": "settled",
                    "settlement_result": _clean_string(source.get("settlement_result")) or "",
                    "is_win": _parse_bool(source.get("is_win")),
                    "cash_required_dollars": _safe_float(source.get("cash_required_dollars")),
                    "realized_pnl_dollars": _safe_float(source.get("realized_pnl_dollars")),
                    "cumulative_realized_pnl_dollars": _safe_float(
                        source.get("cumulative_realized_pnl_dollars")
                    ),
                }
            )
    return _coerce_frame(pd.DataFrame(rows))


def _merge_standalone_settlements(
    canonical_rows: pd.DataFrame,
    standalone_settlements: pd.DataFrame,
    *,
    source_type: str,
    run_name: str,
) -> pd.DataFrame:
    if standalone_settlements.empty:
        return canonical_rows.copy()
    identifier_column = "sample_id" if source_type == "live_research" else "decision_id"
    if canonical_rows.empty:
        merged = standalone_settlements.copy()
        merged["run_name"] = run_name
        merged["source_type"] = source_type
        return merged

    merged = canonical_rows.copy()
    join_columns = [column for column in ("model", identifier_column) if column in merged.columns]
    if identifier_column not in join_columns:
        return merged
    supplemental = standalone_settlements.drop_duplicates(subset=join_columns, keep="last")
    merged = merged.merge(
        supplemental,
        on=join_columns,
        how="left",
        suffixes=("", "__settlement"),
    )
    for column in (
        "ticker",
        "side",
        "settled_at",
        "status",
        "settlement_result",
        "is_win",
        "cash_required_dollars",
        "realized_pnl_dollars",
        "cumulative_realized_pnl_dollars",
    ):
        settlement_column = f"{column}__settlement"
        if settlement_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[settlement_column]
        else:
            merged[column] = _choose_fill_value(merged[column], merged[settlement_column])
        merged.drop(columns=[settlement_column], inplace=True)
    return merged


def _report_candidate_sort_key(candidate: RunCandidate) -> tuple[int, int, int, int, float]:
    if "status" in candidate.canonical_rows.columns:
        settled_count = int((candidate.canonical_rows["status"] == "settled").sum())
    else:
        settled_count = 0
    has_execution_rows = 1 if not candidate.execution_rows.empty else 0
    has_rows = 1 if not candidate.canonical_rows.empty else 0
    column_count = int(candidate.canonical_rows.shape[1])
    modified_at = (
        candidate.report_dir.stat().st_mtime
        if candidate.report_dir is not None and candidate.report_dir.exists()
        else 0.0
    )
    return (has_rows, has_execution_rows, settled_count, column_count, modified_at)


def _discover_report_candidates(report_root: Path) -> list[RunCandidate]:
    if not report_root.exists():
        return []

    grouped: dict[str, list[RunCandidate]] = {}
    for report_dir in sorted(path for path in report_root.iterdir() if path.is_dir()):
        metadata = _read_json(report_dir / "run_metadata.json")
        report_info = _parse_report_markdown(report_dir / "report.md")
        run_name = (
            _clean_string(metadata.get("run_name"))
            or _clean_string(report_info.get("run_name"))
            or report_dir.name
        )
        source_type = (
            _clean_string(metadata.get("detected_source_type"))
            or _clean_string(report_info.get("source_type"))
            or "unknown"
        )
        raw_run_dir = _resolve_repo_path(
            _clean_string(metadata.get("analyzed_path"))
            or _clean_string(report_info.get("analyzed_path"))
        )
        candidate = RunCandidate(
            run_key=_normalize_run_key(raw_run_dir, source_type=source_type, run_name=run_name),
            run_name=run_name,
            source_type=source_type,
            canonical_rows=_coerce_frame(_read_csv_if_exists(report_dir / "canonical_rows.csv")),
            execution_rows=_coerce_frame(_read_csv_if_exists(report_dir / "execution_rows.csv")),
            data_source="performance_report",
            raw_run_dir=raw_run_dir,
            report_dir=report_dir,
            report_path=report_dir / "report.md",
            report_text=report_info.get("report_text"),
            metadata={**report_info, **metadata},
        )
        candidate.canonical_rows = _enrich_canonical_rows(candidate.canonical_rows, candidate.execution_rows)
        grouped.setdefault(candidate.run_key, []).append(candidate)

    selected: list[RunCandidate] = []
    for group in grouped.values():
        chosen = max(group, key=_report_candidate_sort_key)
        chosen.report_variants_count = len(group)
        selected.append(chosen)
    return sorted(selected, key=lambda candidate: candidate.run_name)


def _discover_raw_run_dirs(*roots: Path) -> list[Path]:
    discovered: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for child in sorted(path for path in root.iterdir() if path.is_dir()):
            if (child / "execution").exists() or (child / "research").exists():
                discovered[str(child.resolve())] = child.resolve()
    return sorted(discovered.values())


def _load_raw_candidate(run_dir: Path) -> RunCandidate | None:
    source_type: str
    if (run_dir / "research").exists():
        source_type = "live_research"
        loaded = _load_live_research_run(run_dir)
    elif (run_dir / "execution").exists():
        source_type = "live_execution"
        loaded = _load_live_execution_run(run_dir)
    else:
        return None

    canonical_rows = _coerce_frame(pd.DataFrame(loaded.canonical_rows))
    execution_rows = _coerce_frame(pd.DataFrame(loaded.execution_rows))
    canonical_rows = _merge_standalone_settlements(
        canonical_rows,
        _load_standalone_settlements(run_dir, source_type=source_type),
        source_type=source_type,
        run_name=loaded.run_name,
    )
    canonical_rows = _enrich_canonical_rows(canonical_rows, execution_rows)
    return RunCandidate(
        run_key=_normalize_run_key(run_dir, source_type=source_type, run_name=loaded.run_name),
        run_name=loaded.run_name,
        source_type=source_type,
        canonical_rows=canonical_rows,
        execution_rows=execution_rows,
        data_source="raw_logs",
        raw_run_dir=run_dir,
        metadata={"notes": loaded.notes, "warnings": loaded.warnings},
    )


def _find_execution_started_payload(run_dir: Path) -> dict[str, Any]:
    execution_root = run_dir / "execution"
    if not execution_root.exists():
        return {}
    for events_path in sorted(execution_root.rglob("events.jsonl")):
        for event in _iter_jsonl_rows(events_path, max_lines=50):
            if event.get("event_type") == "execution_started":
                payload = event.get("payload")
                if isinstance(payload, dict):
                    return payload
    return {}


def _infer_run_mode(run_candidate: RunCandidate) -> dict[str, Any]:
    run_name_lower = run_candidate.run_name.lower()
    if run_candidate.source_type == "live_research":
        return {
            "mode": "research",
            "environment": None,
            "shadow_like": False,
            "simulate_immediate_fills": None,
            "shadow_fill_latency_seconds": None,
        }
    if run_candidate.source_type == "offline_artifacts":
        return {
            "mode": "offline",
            "environment": None,
            "shadow_like": False,
            "simulate_immediate_fills": None,
            "shadow_fill_latency_seconds": None,
        }

    payload = (
        _find_execution_started_payload(run_candidate.raw_run_dir)
        if run_candidate.raw_run_dir is not None
        else {}
    )
    payload_mode = _clean_string(payload.get("mode"))
    if payload_mode is not None:
        mode = payload_mode.lower()
    elif any(token in run_name_lower for token in ("paper", "shadow", "preview")):
        mode = "paper"
    elif "live" in run_name_lower:
        mode = "live"
    else:
        mode = "unknown"

    shadow_like = mode == "paper" or any(token in run_name_lower for token in ("shadow", "preview"))
    return {
        "mode": mode,
        "environment": _clean_string(payload.get("environment")),
        "shadow_like": shadow_like,
        "simulate_immediate_fills": payload.get("simulate_immediate_fills"),
        "shadow_fill_latency_seconds": _safe_float(payload.get("shadow_fill_latency_seconds")),
    }


def _time_of_day_bucket(hour_value: int | None) -> str:
    if hour_value is None:
        return "unknown"
    if 0 <= hour_value < 6:
        return "overnight"
    if 6 <= hour_value < 12:
        return "morning"
    if 12 <= hour_value < 18:
        return "afternoon"
    return "evening"


def _prepare_fact_rows(
    run_candidate: RunCandidate,
    *,
    timezone: ZoneInfo,
) -> pd.DataFrame:
    if run_candidate.canonical_rows.empty:
        return pd.DataFrame()

    rows = run_candidate.canonical_rows.copy()
    for column in TIMESTAMP_COLUMNS:
        if column not in rows.columns:
            rows[column] = pd.NaT
    for column in (
        "realized_pnl_dollars",
        "cash_required_dollars",
        "chosen_post_cost_edge_cents",
        "expected_value_dollars",
        "fill_price_cents",
        "filled_contracts",
        "requested_contracts",
        "is_win",
        "regime_label",
        "status",
        "model",
    ):
        if column not in rows.columns:
            rows[column] = math.nan if column not in {"regime_label", "status", "model"} else ""

    rows["run_key"] = run_candidate.run_key
    rows["run_name"] = run_candidate.run_name
    rows["source_type"] = run_candidate.source_type
    rows["model"] = rows["model"].map(lambda value: _clean_string(value) or "unknown")
    rows["regime_label"] = rows["regime_label"].map(lambda value: _clean_string(value) or "unknown")
    rows["status"] = rows["status"].map(lambda value: _clean_string(value) or "unknown")

    mode_info = _infer_run_mode(run_candidate)
    rows["mode"] = mode_info["mode"]
    rows["environment"] = mode_info["environment"]
    rows["shadow_like"] = bool(mode_info["shadow_like"])
    rows["data_source"] = run_candidate.data_source
    rows["report_dir"] = str(run_candidate.report_dir) if run_candidate.report_dir is not None else ""
    rows["raw_run_dir"] = str(run_candidate.raw_run_dir) if run_candidate.raw_run_dir is not None else ""
    rows["report_variants_count"] = run_candidate.report_variants_count
    rows["simulate_immediate_fills"] = mode_info["simulate_immediate_fills"]
    rows["shadow_fill_latency_seconds"] = mode_info["shadow_fill_latency_seconds"]

    contracts = rows["filled_contracts"].where(rows["filled_contracts"] > 0, rows["requested_contracts"])
    rows["contracts"] = contracts.fillna(1.0).astype(float)
    rows["contracts"] = rows["contracts"].where(rows["contracts"] > 0, 1.0)

    rows["expected_edge_cents"] = rows["chosen_post_cost_edge_cents"]
    expected_from_dollars = (rows["expected_value_dollars"] * 100.0) / rows["contracts"]
    rows["expected_edge_cents"] = rows["expected_edge_cents"].where(
        rows["expected_edge_cents"].notna(),
        expected_from_dollars,
    )

    rows["is_settled"] = rows["status"].eq("settled") & rows["realized_pnl_dollars"].notna()
    rows["is_win"] = rows["is_win"].where(rows["is_win"].notna(), rows["realized_pnl_dollars"] > 0.0)
    rows["realized_edge_cents"] = (rows["realized_pnl_dollars"] * 100.0) / rows["contracts"]

    entry_cost_dollars = (rows["fill_price_cents"] / 100.0) * rows["contracts"]
    inferred_fee_dollars = rows["cash_required_dollars"] - entry_cost_dollars
    inferred_fee_dollars = inferred_fee_dollars.where(inferred_fee_dollars >= -0.015)
    inferred_fee_dollars = inferred_fee_dollars.where(
        inferred_fee_dollars.isna() | (inferred_fee_dollars >= 0.0),
        0.0,
    )
    rows["fee_dollars"] = inferred_fee_dollars
    rows["fee_covered"] = rows["fee_dollars"].notna()
    rows["gross_pnl_dollars"] = rows["realized_pnl_dollars"] + rows["fee_dollars"]
    rows["gross_pnl_with_fallback_dollars"] = rows["gross_pnl_dollars"].where(
        rows["gross_pnl_dollars"].notna(),
        rows["realized_pnl_dollars"],
    )
    rows["expected_edge_covered"] = rows["expected_edge_cents"].notna()

    rows["settled_at_local"] = rows["settled_at"].dt.tz_convert(timezone)
    rows["settled_local_date"] = rows["settled_at_local"].dt.date
    rows["settled_hour_local"] = rows["settled_at_local"].dt.hour
    rows["time_of_day_bucket"] = rows["settled_hour_local"].map(
        lambda value: _time_of_day_bucket(_safe_int(value))
    )
    for bucket_column in BUCKET_COLUMNS:
        if bucket_column in rows.columns:
            rows[bucket_column] = rows[bucket_column].map(lambda value: _clean_string(value) or "unknown")
    return rows


def _first_valid_timestamp(frame: pd.DataFrame) -> datetime | None:
    timestamps: list[datetime] = []
    for column in TIMESTAMP_COLUMNS:
        if column not in frame.columns:
            continue
        series = frame[column].dropna()
        for value in series.tolist():
            if isinstance(value, pd.Timestamp):
                timestamps.append(value.to_pydatetime())
            elif isinstance(value, datetime):
                timestamps.append(value)
    return min(timestamps) if timestamps else None


def _last_valid_timestamp(frame: pd.DataFrame) -> datetime | None:
    timestamps: list[datetime] = []
    for column in TIMESTAMP_COLUMNS:
        if column not in frame.columns:
            continue
        series = frame[column].dropna()
        for value in series.tolist():
            if isinstance(value, pd.Timestamp):
                timestamps.append(value.to_pydatetime())
            elif isinstance(value, datetime):
                timestamps.append(value)
    return max(timestamps) if timestamps else None


def _per_minute_pnl_series(settled_rows: pd.DataFrame) -> pd.Series:
    if settled_rows.empty:
        return pd.Series(dtype="float64")
    minute_index = settled_rows["settled_at"].dt.floor("min")
    minute_pnl = (
        settled_rows.assign(settled_minute=minute_index)
        .groupby("settled_minute", dropna=True)["realized_pnl_dollars"]
        .sum()
        .sort_index()
    )
    if minute_pnl.empty:
        return pd.Series(dtype="float64")
    full_index = pd.date_range(
        start=minute_pnl.index.min(),
        end=minute_pnl.index.max(),
        freq="min",
        tz=UTC,
    )
    return minute_pnl.reindex(full_index, fill_value=0.0)


def _max_drawdown(pnl_series: pd.Series) -> float | None:
    if pnl_series.empty:
        return None
    cumulative = pnl_series.cumsum()
    running_peak = cumulative.cummax()
    drawdown = running_peak - cumulative
    return float(drawdown.max()) if not drawdown.empty else None


def _annualized_sharpe(pnl_series: pd.Series) -> float | None:
    if pnl_series.empty:
        return None
    std = float(pnl_series.std(ddof=0))
    if std <= 0.0:
        return None
    return float(pnl_series.mean() / std * math.sqrt(MINUTES_PER_YEAR))


def _annualized_sortino(pnl_series: pd.Series) -> float | None:
    if pnl_series.empty:
        return None
    downside = pnl_series[pnl_series < 0.0]
    if downside.empty:
        return None
    downside_std = float(downside.std(ddof=0))
    if downside_std <= 0.0:
        return None
    return float(pnl_series.mean() / downside_std * math.sqrt(MINUTES_PER_YEAR))


def _annualized_calmar(pnl_series: pd.Series) -> float | None:
    if pnl_series.empty:
        return None
    max_drawdown = _max_drawdown(pnl_series)
    if max_drawdown is None or max_drawdown <= 0.0:
        return None
    annualized_pnl = float(pnl_series.mean() * MINUTES_PER_YEAR)
    return annualized_pnl / max_drawdown


def _aggregate_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    settled = frame.loc[frame["is_settled"]].copy()
    start_timestamp = _first_valid_timestamp(frame)
    end_timestamp = _last_valid_timestamp(frame)

    if settled.empty:
        return {
            "start_timestamp": None if start_timestamp is None else start_timestamp.isoformat(),
            "end_timestamp": None if end_timestamp is None else end_timestamp.isoformat(),
            "settled_row_count": 0,
            "gross_pnl_dollars": None,
            "net_pnl_after_fees_dollars": None,
            "hit_rate": None,
            "average_winner_dollars": None,
            "average_loser_dollars": None,
            "win_loss_ratio": None,
            "max_drawdown_dollars": None,
            "sharpe": None,
            "sortino": None,
            "calmar": None,
            "avg_realized_edge_cents": None,
            "avg_expected_edge_cents": None,
            "realized_vs_expected_edge_cents": None,
            "fee_coverage_ratio": None,
            "expected_edge_coverage_ratio": None,
        }

    winners = settled.loc[settled["realized_pnl_dollars"] > 0.0, "realized_pnl_dollars"]
    losers = settled.loc[settled["realized_pnl_dollars"] < 0.0, "realized_pnl_dollars"]
    avg_winner = None if winners.empty else float(winners.mean())
    avg_loser = None if losers.empty else float(losers.mean())
    hit_rate = float((settled["realized_pnl_dollars"] > 0.0).mean())
    win_loss_ratio = (
        None
        if avg_winner is None or avg_loser is None or avg_loser == 0.0
        else avg_winner / abs(avg_loser)
    )

    per_minute_pnl = _per_minute_pnl_series(settled)
    expected_subset = settled.loc[
        settled["realized_edge_cents"].notna() & settled["expected_edge_cents"].notna()
    ]
    avg_realized_edge = (
        None if expected_subset.empty else float(expected_subset["realized_edge_cents"].mean())
    )
    avg_expected_edge = (
        None if expected_subset.empty else float(expected_subset["expected_edge_cents"].mean())
    )
    realized_vs_expected = (
        None
        if avg_realized_edge is None or avg_expected_edge is None
        else avg_realized_edge - avg_expected_edge
    )

    fee_coverage_ratio = float(settled["fee_covered"].mean()) if "fee_covered" in settled.columns else None
    expected_coverage_ratio = (
        float(settled["expected_edge_covered"].mean())
        if "expected_edge_covered" in settled.columns
        else None
    )

    return {
        "start_timestamp": None if start_timestamp is None else start_timestamp.isoformat(),
        "end_timestamp": None if end_timestamp is None else end_timestamp.isoformat(),
        "settled_row_count": int(len(settled)),
        "gross_pnl_dollars": float(settled["gross_pnl_with_fallback_dollars"].sum()),
        "net_pnl_after_fees_dollars": float(settled["realized_pnl_dollars"].sum()),
        "hit_rate": hit_rate,
        "average_winner_dollars": avg_winner,
        "average_loser_dollars": avg_loser,
        "win_loss_ratio": win_loss_ratio,
        "max_drawdown_dollars": _max_drawdown(per_minute_pnl),
        "sharpe": _annualized_sharpe(per_minute_pnl),
        "sortino": _annualized_sortino(per_minute_pnl),
        "calmar": _annualized_calmar(per_minute_pnl),
        "avg_realized_edge_cents": avg_realized_edge,
        "avg_expected_edge_cents": avg_expected_edge,
        "realized_vs_expected_edge_cents": realized_vs_expected,
        "fee_coverage_ratio": fee_coverage_ratio,
        "expected_edge_coverage_ratio": expected_coverage_ratio,
    }


def _contamination_metadata(
    *,
    mode: str,
    shadow_like: bool,
    start_timestamp: datetime | None,
) -> tuple[bool, str | None]:
    if not shadow_like or start_timestamp is None:
        return False, None
    if start_timestamp >= FILL_REALISM_FIX_AT:
        return False, None
    return (
        True,
        f"{mode} run began before fill-realism fix {FILL_REALISM_FIX_COMMIT} at {FILL_REALISM_FIX_AT.isoformat()}",
    )


def _build_run_inventory_rows(fact_rows: pd.DataFrame) -> list[dict[str, Any]]:
    inventory_rows: list[dict[str, Any]] = []
    if fact_rows.empty:
        return inventory_rows

    run_group_columns = [
        "run_key",
        "run_name",
        "source_type",
        "mode",
        "data_source",
        "report_dir",
        "raw_run_dir",
        "environment",
        "shadow_like",
        "report_variants_count",
        "simulate_immediate_fills",
        "shadow_fill_latency_seconds",
    ]
    grouped = fact_rows.groupby(run_group_columns, dropna=False, sort=True)
    for keys, group in grouped:
        row = dict(zip(run_group_columns, keys))
        start_timestamp = _first_valid_timestamp(group)
        contamination_flag, contamination_reason = _contamination_metadata(
            mode=str(row["mode"]),
            shadow_like=bool(row["shadow_like"]),
            start_timestamp=start_timestamp,
        )
        metrics = _aggregate_metrics(group)
        inventory_rows.append(
            {
                **row,
                **metrics,
                "contamination_flag": contamination_flag,
                "contamination_reason": contamination_reason,
            }
        )
    return inventory_rows


def _build_run_scorecard_rows(fact_rows: pd.DataFrame) -> list[dict[str, Any]]:
    run_scorecard_rows: list[dict[str, Any]] = []
    if fact_rows.empty:
        return run_scorecard_rows

    group_columns = [
        "run_key",
        "run_name",
        "source_type",
        "mode",
        "model",
        "data_source",
        "report_dir",
        "raw_run_dir",
        "environment",
        "shadow_like",
        "report_variants_count",
        "simulate_immediate_fills",
        "shadow_fill_latency_seconds",
    ]
    grouped = fact_rows.groupby(group_columns, dropna=False, sort=True)
    for keys, group in grouped:
        row = dict(zip(group_columns, keys))
        start_timestamp = _first_valid_timestamp(group)
        contamination_flag, contamination_reason = _contamination_metadata(
            mode=str(row["mode"]),
            shadow_like=bool(row["shadow_like"]),
            start_timestamp=start_timestamp,
        )
        metrics = _aggregate_metrics(group)
        fee_quality = "net_only"
        fee_coverage_ratio = metrics.get("fee_coverage_ratio")
        if fee_coverage_ratio is not None:
            if fee_coverage_ratio >= 0.999:
                fee_quality = "full"
            elif fee_coverage_ratio > 0.0:
                fee_quality = "partial"
        run_scorecard_rows.append(
            {
                **row,
                **metrics,
                "contamination_flag": contamination_flag,
                "contamination_reason": contamination_reason,
                "gross_pnl_fee_coverage_quality": fee_quality,
            }
        )
    return run_scorecard_rows


def _filter_calendar_window(
    settled_rows: pd.DataFrame,
    *,
    as_of_date: date,
    days: int,
) -> pd.DataFrame:
    if settled_rows.empty:
        return settled_rows.copy()
    window_start = as_of_date - timedelta(days=days - 1)
    return settled_rows.loc[
        settled_rows["settled_local_date"].between(window_start, as_of_date, inclusive="both")
    ].copy()


def _rollup_rows_for_window(
    settled_rows: pd.DataFrame,
    *,
    window_days: int,
    as_of_date: date,
    mode_scope: str,
    segment_type: str,
    segment_dimension: str,
    segment_value: str,
) -> dict[str, Any]:
    metrics = _aggregate_metrics(settled_rows)
    return {
        "window_days": window_days,
        "window_start_date": (as_of_date - timedelta(days=window_days - 1)).isoformat(),
        "window_end_date": as_of_date.isoformat(),
        "mode_scope": mode_scope,
        "segment_type": segment_type,
        "segment_dimension": segment_dimension,
        "segment_value": segment_value,
        **metrics,
    }


def _build_window_rollups(
    fact_rows: pd.DataFrame,
    *,
    windows: tuple[int, ...],
    as_of_date: date,
) -> list[dict[str, Any]]:
    settled_rows = fact_rows.loc[fact_rows["is_settled"]].copy()
    if settled_rows.empty:
        return []

    rollups: list[dict[str, Any]] = []
    available_modes = ["all", *sorted(mode for mode in settled_rows["mode"].dropna().unique() if mode)]
    for window_days in windows:
        window_rows = _filter_calendar_window(settled_rows, as_of_date=as_of_date, days=window_days)
        for mode_scope in available_modes:
            scoped_rows = window_rows if mode_scope == "all" else window_rows.loc[window_rows["mode"] == mode_scope]
            if scoped_rows.empty:
                continue
            rollups.append(
                _rollup_rows_for_window(
                    scoped_rows,
                    window_days=window_days,
                    as_of_date=as_of_date,
                    mode_scope=mode_scope,
                    segment_type="overall",
                    segment_dimension="overall",
                    segment_value="all",
                )
            )
            for model, group in scoped_rows.groupby("model", sort=True):
                rollups.append(
                    _rollup_rows_for_window(
                        group,
                        window_days=window_days,
                        as_of_date=as_of_date,
                        mode_scope=mode_scope,
                        segment_type="model",
                        segment_dimension="model",
                        segment_value=str(model),
                    )
                )
            for regime_label, group in scoped_rows.groupby("regime_label", sort=True):
                rollups.append(
                    _rollup_rows_for_window(
                        group,
                        window_days=window_days,
                        as_of_date=as_of_date,
                        mode_scope=mode_scope,
                        segment_type="regime",
                        segment_dimension="regime_label",
                        segment_value=str(regime_label),
                    )
                )
            for bucket_column in BUCKET_COLUMNS:
                if bucket_column not in scoped_rows.columns:
                    continue
                for bucket_value, group in scoped_rows.groupby(bucket_column, sort=True):
                    rollups.append(
                        _rollup_rows_for_window(
                            group,
                            window_days=window_days,
                            as_of_date=as_of_date,
                            mode_scope=mode_scope,
                            segment_type="bucket",
                            segment_dimension=bucket_column,
                            segment_value=str(bucket_value),
                        )
                    )
                if bucket_column == "bucket_policy_bucket":
                    continue
            for time_bucket, group in scoped_rows.groupby("time_of_day_bucket", sort=True):
                rollups.append(
                    _rollup_rows_for_window(
                        group,
                        window_days=window_days,
                        as_of_date=as_of_date,
                        mode_scope=mode_scope,
                        segment_type="time_of_day",
                        segment_dimension="time_of_day_bucket",
                        segment_value=str(time_bucket),
                    )
                )
    return rollups


def _build_query_summaries(
    fact_rows: pd.DataFrame,
    *,
    as_of_date: date,
) -> dict[str, Any]:
    settled_rows = fact_rows.loc[fact_rows["is_settled"]].copy()
    queries: dict[str, Any] = {}
    for mode in ("live", "paper", "research"):
        scoped = settled_rows.loc[settled_rows["mode"] == mode]
        if scoped.empty:
            continue
        metrics = _aggregate_metrics(_filter_calendar_window(scoped, as_of_date=as_of_date, days=30))
        queries[f"{mode}_last_30_calendar_days"] = {
            "window_days": 30,
            "window_start_date": (as_of_date - timedelta(days=29)).isoformat(),
            "window_end_date": as_of_date.isoformat(),
            "mode": mode,
            **metrics,
        }
    return queries


def _format_currency(value: Any) -> str:
    parsed = _safe_float(value)
    return "n/a" if parsed is None else f"${parsed:,.2f}"


def _format_number(value: Any, digits: int = 2) -> str:
    parsed = _safe_float(value)
    return "n/a" if parsed is None else f"{parsed:,.{digits}f}"


def _render_summary_markdown(
    *,
    as_of_date: date,
    timezone_name: str,
    queries: dict[str, Any],
    run_scorecard: pd.DataFrame,
) -> str:
    lines = [
        "# Unified Performance Scorecard",
        "",
        f"- As of date: `{as_of_date.isoformat()}`",
        f"- Local timezone: `{timezone_name}`",
        f"- Fill-realism contamination cutoff: `{FILL_REALISM_FIX_COMMIT}` at `{FILL_REALISM_FIX_AT.isoformat()}`",
    ]
    live_30 = queries.get("live_last_30_calendar_days")
    if live_30:
        lines.extend(
            [
                "",
                "## Acceptance Answer",
                "",
                f"- Live net PnL after fees over the last 30 calendar days: `{_format_currency(live_30.get('net_pnl_after_fees_dollars'))}`",
                f"- Live Sharpe over the last 30 calendar days: `{_format_number(live_30.get('sharpe'), 3)}`",
                f"- Live realized edge vs expected edge over the last 30 calendar days: `{_format_number(live_30.get('avg_realized_edge_cents'), 2)}c` vs `{_format_number(live_30.get('avg_expected_edge_cents'), 2)}c`",
                f"- Live realized minus expected edge delta: `{_format_number(live_30.get('realized_vs_expected_edge_cents'), 2)}c`",
            ]
        )
    if not run_scorecard.empty:
        recent = run_scorecard.sort_values("end_timestamp", ascending=False).head(10)
        lines.extend(
            [
                "",
                "## Recent Run Scorecard",
                "",
                "| Run | Mode | Model | Settled | Net PnL | Sharpe | Contaminated |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in recent.iterrows():
            lines.append(
                f"| {row['run_name']} | {row['mode']} | {row['model']} | "
                f"{int(row['settled_row_count']) if pd.notna(row['settled_row_count']) else 0} | "
                f"{_format_currency(row['net_pnl_after_fees_dollars'])} | "
                f"{_format_number(row['sharpe'], 3)} | "
                f"{'yes' if bool(row['contamination_flag']) else 'no'} |"
            )
    return "\n".join(lines).strip() + "\n"


def generate_performance_scorecard(
    *,
    report_root: Path = DEFAULT_REPORT_ROOT,
    live_roots: tuple[Path, ...] = (DEFAULT_LIVE_ROOT,),
    live_research_roots: tuple[Path, ...] = (),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    as_of_date: date | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict[str, Any]:
    timezone = ZoneInfo(timezone_name)
    resolved_as_of_date = as_of_date or datetime.now(tz=timezone).date()

    report_candidates = _discover_report_candidates(report_root)
    covered_run_keys = {candidate.run_key for candidate in report_candidates}

    raw_candidates: list[RunCandidate] = []
    for run_dir in _discover_raw_run_dirs(*live_roots, *live_research_roots):
        raw_key = _normalize_run_key(run_dir, source_type="raw", run_name=run_dir.name)
        if raw_key in covered_run_keys:
            continue
        candidate = _load_raw_candidate(run_dir)
        if candidate is not None:
            raw_candidates.append(candidate)

    all_candidates = [*report_candidates, *raw_candidates]
    fact_frames = []
    for candidate in all_candidates:
        if candidate.canonical_rows.empty:
            continue
        prepared = _prepare_fact_rows(candidate, timezone=timezone)
        if not prepared.empty:
            fact_frames.append(prepared)
    if fact_frames:
        fact_records: list[dict[str, Any]] = []
        for frame in fact_frames:
            fact_records.extend(frame.to_dict(orient="records"))
        fact_rows = pd.DataFrame.from_records(fact_records)
    else:
        fact_rows = pd.DataFrame()

    run_inventory_rows = _build_run_inventory_rows(fact_rows)
    run_scorecard_rows = _build_run_scorecard_rows(fact_rows)
    window_rollups = _build_window_rollups(
        fact_rows,
        windows=windows,
        as_of_date=resolved_as_of_date,
    )
    queries = _build_query_summaries(fact_rows, as_of_date=resolved_as_of_date)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_inventory_df = pd.DataFrame(run_inventory_rows)
    if not run_inventory_df.empty:
        run_inventory_df = run_inventory_df.sort_values(
            ["end_timestamp", "run_name"],
            ascending=[False, True],
            na_position="last",
        )
    run_scorecard_df = pd.DataFrame(run_scorecard_rows)
    if not run_scorecard_df.empty:
        run_scorecard_df = run_scorecard_df.sort_values(
            ["end_timestamp", "run_name", "model"],
            ascending=[False, True, True],
            na_position="last",
        )
    window_rollups_df = pd.DataFrame(window_rollups)
    if not window_rollups_df.empty:
        window_rollups_df = window_rollups_df.sort_values(
            ["window_days", "mode_scope", "segment_type", "segment_dimension", "segment_value"],
            ascending=True,
            na_position="last",
        )

    run_inventory_path = output_dir / "run_inventory.csv"
    run_scorecard_path = output_dir / "run_scorecard.csv"
    window_rollups_path = output_dir / "window_rollups.csv"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "performance_scorecard.md"

    run_inventory_df.to_csv(run_inventory_path, index=False)
    run_scorecard_df.to_csv(run_scorecard_path, index=False)
    window_rollups_df.to_csv(window_rollups_path, index=False)

    summary_payload = {
        "as_of_date": resolved_as_of_date.isoformat(),
        "timezone": timezone_name,
        "fill_realism_fix_commit": FILL_REALISM_FIX_COMMIT,
        "fill_realism_fix_at": FILL_REALISM_FIX_AT.isoformat(),
        "report_root": str(report_root),
        "live_roots": [str(path) for path in live_roots],
        "live_research_roots": [str(path) for path in live_research_roots],
        "run_inventory_csv": str(run_inventory_path),
        "run_scorecard_csv": str(run_scorecard_path),
        "window_rollups_csv": str(window_rollups_path),
        "queries": queries,
        "ingested_run_count": len(run_inventory_rows),
        "ingested_run_model_count": len(run_scorecard_rows),
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    summary_md_path.write_text(
        _render_summary_markdown(
            as_of_date=resolved_as_of_date,
            timezone_name=timezone_name,
            queries=queries,
            run_scorecard=run_scorecard_df,
        ),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "run_inventory_csv": str(run_inventory_path),
        "run_scorecard_csv": str(run_scorecard_path),
        "window_rollups_csv": str(window_rollups_path),
        "summary_json": str(summary_json_path),
        "summary_markdown": str(summary_md_path),
        "queries": queries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a unified performance scorecard across live runs and performance reports."
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
        help="Root directory containing artifacts/kalshi/performance_reports run folders.",
    )
    parser.add_argument(
        "--live-root",
        type=Path,
        action="append",
        default=[],
        help="Repeatable root directory containing output/live run folders.",
    )
    parser.add_argument(
        "--live-research-root",
        type=Path,
        action="append",
        default=[],
        help="Repeatable root directory containing output/live_research run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the scorecard CSV and summary outputs will be written.",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Optional calendar anchor date in YYYY-MM-DD format. Defaults to the current local date.",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help="IANA timezone used for calendar windows and time-of-day segmenting.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        nargs="+",
        default=list(DEFAULT_WINDOWS),
        help="Calendar-day windows to compute, for example 7 30 90.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    as_of_date = None if args.as_of_date is None else date.fromisoformat(args.as_of_date)
    live_roots = tuple(args.live_root) or (DEFAULT_LIVE_ROOT,)
    live_research_roots = tuple(args.live_research_root)
    outputs = generate_performance_scorecard(
        report_root=args.report_root,
        live_roots=live_roots,
        live_research_roots=live_research_roots,
        output_dir=args.output_dir,
        as_of_date=as_of_date,
        timezone_name=args.timezone,
        windows=tuple(args.window_days),
    )

    live_30 = outputs["queries"].get("live_last_30_calendar_days")
    if live_30 is not None:
        print(
            "Live last 30 calendar days: "
            f"net PnL after fees {_format_currency(live_30.get('net_pnl_after_fees_dollars'))}, "
            f"Sharpe {_format_number(live_30.get('sharpe'), 3)}, "
            f"realized edge {_format_number(live_30.get('avg_realized_edge_cents'), 2)}c vs "
            f"expected edge {_format_number(live_30.get('avg_expected_edge_cents'), 2)}c "
            f"(delta {_format_number(live_30.get('realized_vs_expected_edge_cents'), 2)}c)."
        )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
