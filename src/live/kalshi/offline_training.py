from __future__ import annotations

import heapq
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from shutil import copy2
from typing import Any
from zoneinfo import TZPATH, ZoneInfo

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from lightgbm import Booster, LGBMClassifier, early_stopping
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from src.live.kalshi.calibration import fit_platt_scaler, save_platt_calibration
from src.live.data.btc_spot_feed import BTCSpotUpdate
from src.live.kalshi.features import (
    DEFAULT_FEATURE_SCHEMA,
    FEATURE_ORDER,
    FEATURE_SCHEMA_CHOICES,
    HOURLY_CONTEXT_FEATURE_ORDER as LIVE_HOURLY_CONTEXT_FEATURE_ORDER,
    LINEAR_V1_DERIVED_FEATURE_ORDER,
    LINEAR_V1_DERIVED_FORMULAS,
    LINEAR_V1_FEATURE_ORDER,
    LINEAR_V1_FEATURE_SCHEMA,
    SPOT_V1_FEATURE_ORDER,
    SPOT_V1_FEATURE_SCHEMA,
    KalshiFeatureEngineConfig,
    KalshiTradeFeatureAccumulator,
    feature_order_hash,
)
from src.live.kalshi.regime import evaluate_kxbtc15m_regime
from src.live.kalshi.signal_risk import (
    KalshiSignalRiskConfig,
    calculate_kelly_sizing_metrics,
    calculate_cost_metrics,
    find_max_acceptable_entry_price_cents,
)
from src.live.kalshi.spot_features import SpotFeatureTracker
from src.live.kalshi.strike import strike_price_from_market
from src.live.kalshi.types import KalshiTickerUpdate

DEFAULT_SERIES = "KXBTC15M"
DEFAULT_OPTUNA_TRIALS = 40
DEFAULT_OPTUNA_SAMPLE_SIZE = 2_000_000
DEFAULT_STARTING_CASH_DOLLARS = 10_000.0
DEFAULT_LIGHTGBM_ARTIFACTS_ROOT = Path("artifacts") / "kalshi" / "kxbtc15m_lightgbm"
DEFAULT_LASSO_ARTIFACTS_ROOT = Path("artifacts") / "kalshi" / "kxbtc15m_lasso"
DEFAULT_BAGGED_LASSO_ARTIFACTS_ROOT = Path("artifacts") / "kalshi" / "kxbtc15m_bagged_lasso"
DEFAULT_ELASTIC_NET_ARTIFACTS_ROOT = Path("artifacts") / "kalshi" / "kxbtc15m_elastic_net"
DEFAULT_LINEAR_SVM_ARTIFACTS_ROOT = Path("artifacts") / "kalshi" / "kxbtc15m_linear_svm"
DEFAULT_ARTIFACTS_ROOT = DEFAULT_LIGHTGBM_ARTIFACTS_ROOT
DEFAULT_LASSO_SAMPLE_SIZE = 2_000_000
DEFAULT_LASSO_C_VALUES = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
DEFAULT_LASSO_MAX_ITER = 2000
DEFAULT_LASSO_TOL = 1e-4
DEFAULT_BAGGED_LASSO_SAMPLE_SIZE = 1_000_000
DEFAULT_BAGGED_LASSO_C_VALUES = DEFAULT_LASSO_C_VALUES
DEFAULT_BAGGED_LASSO_N_ESTIMATORS = 15
DEFAULT_BAGGED_LASSO_MAX_SAMPLES = 0.5
DEFAULT_BAGGED_LASSO_MAX_ITER = DEFAULT_LASSO_MAX_ITER
DEFAULT_BAGGED_LASSO_TOL = DEFAULT_LASSO_TOL
DEFAULT_BAGGED_LASSO_N_JOBS = -1
DEFAULT_ELASTIC_NET_SAMPLE_SIZE = 2_000_000
DEFAULT_ELASTIC_NET_C_VALUES = DEFAULT_LASSO_C_VALUES
DEFAULT_ELASTIC_NET_L1_RATIOS = (0.1, 0.25, 0.5, 0.75, 0.9)
DEFAULT_ELASTIC_NET_MAX_ITER = DEFAULT_LASSO_MAX_ITER
DEFAULT_ELASTIC_NET_TOL = DEFAULT_LASSO_TOL
DEFAULT_LINEAR_SVM_SAMPLE_SIZE = 2_000_000
DEFAULT_LINEAR_SVM_C_VALUES = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
DEFAULT_LINEAR_SVM_MAX_ITER = 5000
DEFAULT_LINEAR_SVM_TOL = 1e-4
ML_RANDOM_SEED = 42
MINIMUM_POLICY_TRADES = 500
MINIMUM_POLICY_TRADES_PER_DAY = 10.0
DEFAULT_PURGE_EMBARGO_SAFETY_MARGIN_SECONDS = 60
DEFAULT_TARGET_HORIZON_SECONDS = 15 * 60
_BINARY_CLASS_LABELS = [0, 1]
_BASE_FEATURE_ORDER = tuple(FEATURE_ORDER)
_MARKET_FEATURE_METADATA_COLUMNS = (
    "ticker",
    "trade_id",
    "created_time",
    "open_time",
    "close_time",
    "actual_outcome",
    "market_prob",
    "count",
    "taker_side",
)
HOURLY_CONTEXT_SERIES = "KXBTCD"
HOURLY_CONTEXT_TAU_MAX_MINUTES = 60.0
HOURLY_CONTEXT_MAX_STALENESS_SECONDS = 300.0
HOURLY_CONTEXT_MIN_RECENT_TRADE_COUNT = 1.0
HOURLY_CONTEXT_FEATURE_ORDER = tuple(LIVE_HOURLY_CONTEXT_FEATURE_ORDER)
_GENERIC_BTC_DIRECTION_TITLE = "btc price up in next 15 mins?"
LOGGER = logging.getLogger(__name__)
DIAGNOSTIC_TIMEZONE = "America/New_York"
DIAGNOSTIC_TIMEZONE_INFO = ZoneInfo(DIAGNOSTIC_TIMEZONE)
REALIZED_VOL_LOOKBACK_MINUTES = 240
REALIZED_VOL_MIN_PERIODS = 180
REALIZED_VOL_ANNUALIZATION_FACTOR = float(np.sqrt(60.0 * 24.0 * 365.0))
REALIZED_VOL_REGIME_LABELS = ("low", "medium", "high", "extreme")
REALIZED_VOL_REGIME_UNKNOWN = "unknown"
SPOT_REALIZED_VOL_REGIME_EDGES = (0.40, 0.60, 0.90)
SPOT_REALIZED_VOL_REGIME_VERSION = "spot_2026q1_backfill_v1"
SPOT_REALIZED_VOL_REGIME_DERIVED_FROM = "2026 Q1 external_spot_normalized/backfill prior-4h BTC spot vol"
MARKET_PROB_REALIZED_VOL_REGIME_EDGES = (100.0, 125.0, 140.0)
MARKET_PROB_REALIZED_VOL_REGIME_VERSION = "market_prob_2025q4_2026q1_proxy_v1"
MARKET_PROB_REALIZED_VOL_REGIME_DERIVED_FROM = (
    "2025-12-10 to 2026-03-17 KXBTC15M 1-minute last yes-price proxy prior-4h vol"
)
REALIZED_VOL_REGIME_REVISIT_NOTE = "Revisit fixed edges after broader historical coverage is available."
SESSION_BLOCK_LABELS = ("asia", "europe", "us_preopen", "us_regular", "us_postclose")


@dataclass(frozen=True)
class HourlyContextConfig:
    series_ticker: str = HOURLY_CONTEXT_SERIES
    tau_max_minutes: float = HOURLY_CONTEXT_TAU_MAX_MINUTES
    max_staleness_seconds: float = HOURLY_CONTEXT_MAX_STALENESS_SECONDS
    min_recent_trade_count_300s: float = HOURLY_CONTEXT_MIN_RECENT_TRADE_COUNT


def _default_external_spot_normalized_root() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "output" / "kalshi_kxbtc15m_data" / "curated" / "external_spot_normalized" / "backfill"
    return candidate if candidate.exists() else None


def _timezone_metadata() -> dict[str, object]:
    try:
        tzdata_version = importlib_metadata.version("tzdata")
        zoneinfo_source = "tzdata"
    except importlib_metadata.PackageNotFoundError:
        tzdata_version = None
        zoneinfo_source = "system_zoneinfo"
    return {
        "timezone": DIAGNOSTIC_TIMEZONE,
        "zoneinfo_source": zoneinfo_source,
        "tzdata_version": tzdata_version,
        "zoneinfo_paths": [str(path) for path in TZPATH],
    }


@lru_cache(maxsize=256)
def _load_external_spot_events_for_day(day_dir: str) -> pd.DataFrame:
    day_path = Path(day_dir)
    events_path = day_path / "events.parquet"
    if not events_path.exists():
        return pd.DataFrame(columns=["event_time", "btc_spot_price"])
    frame = pd.read_parquet(events_path, columns=["event_time", "btc_spot_price"])
    if frame.empty:
        return frame
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    return frame.dropna(subset=["btc_spot_price"]).sort_values("event_time").reset_index(drop=True)


def _load_external_spot_minute_series(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    *,
    external_spot_root: Path | None = None,
) -> pd.DataFrame | None:
    root = external_spot_root or _default_external_spot_normalized_root()
    if root is None or not root.exists():
        return None
    start_day = (pd.Timestamp(start_time) - pd.Timedelta(minutes=REALIZED_VOL_LOOKBACK_MINUTES)).floor("D")
    end_day = pd.Timestamp(end_time).floor("D")
    day_range = pd.date_range(start=start_day, end=end_day, freq="D", tz=UTC)
    frames: list[pd.DataFrame] = []
    for day in day_range:
        day_dir = root / day.strftime("%Y-%m-%d")
        if not day_dir.exists():
            continue
        day_frame = _load_external_spot_events_for_day(str(day_dir))
        if not day_frame.empty:
            frames.append(day_frame)
    if not frames:
        return None
    spot_events = pd.concat(frames, ignore_index=True).sort_values("event_time")
    minute_series = (
        spot_events.set_index("event_time")["btc_spot_price"].resample("1min").last().ffill().to_frame("btc_spot_price")
    )
    minute_series["spot_realized_vol_4h"] = (
        np.log(minute_series["btc_spot_price"]).diff().rolling(REALIZED_VOL_LOOKBACK_MINUTES, min_periods=REALIZED_VOL_MIN_PERIODS).std()
        * REALIZED_VOL_ANNUALIZATION_FACTOR
    )
    return minute_series.reset_index(names="event_minute")


def _align_spot_realized_vol_to_rows(
    created_times: pd.Series,
    *,
    external_spot_root: Path | None = None,
) -> pd.Series:
    if created_times.empty:
        return pd.Series(dtype=np.float64)
    minute_series = _load_external_spot_minute_series(
        pd.Timestamp(created_times.min()),
        pd.Timestamp(created_times.max()),
        external_spot_root=external_spot_root,
    )
    if minute_series is None or minute_series.empty:
        return pd.Series(np.nan, index=created_times.index, dtype=np.float64)
    lookup = pd.DataFrame({"created_time": pd.to_datetime(created_times, utc=True)}).sort_values("created_time")
    aligned = pd.merge_asof(
        lookup,
        minute_series[["event_minute", "spot_realized_vol_4h"]].sort_values("event_minute"),
        left_on="created_time",
        right_on="event_minute",
        direction="backward",
        tolerance=pd.Timedelta(minutes=1),
    )
    result = pd.Series(aligned["spot_realized_vol_4h"].to_numpy(dtype=np.float64), index=lookup.index)
    return result.reindex(created_times.index)


def _align_market_prob_realized_vol_to_rows(created_times: pd.Series, market_prob: pd.Series) -> pd.Series:
    if created_times.empty:
        return pd.Series(dtype=np.float64)
    minute_frame = (
        pd.DataFrame({"created_time": pd.to_datetime(created_times, utc=True), "market_prob": pd.to_numeric(market_prob, errors="coerce")})
        .dropna(subset=["market_prob"])
        .sort_values("created_time")
        .set_index("created_time")
    )
    if minute_frame.empty:
        return pd.Series(np.nan, index=created_times.index, dtype=np.float64)
    minute_series = minute_frame["market_prob"].resample("1min").last().ffill().to_frame("market_prob")
    minute_series["market_prob_realized_vol_4h"] = (
        minute_series["market_prob"].diff().rolling(REALIZED_VOL_LOOKBACK_MINUTES, min_periods=REALIZED_VOL_MIN_PERIODS).std()
        * REALIZED_VOL_ANNUALIZATION_FACTOR
    )
    lookup = pd.DataFrame({"created_time": pd.to_datetime(created_times, utc=True)}).sort_values("created_time")
    aligned = pd.merge_asof(
        lookup,
        minute_series.reset_index(names="event_minute")[["event_minute", "market_prob_realized_vol_4h"]].sort_values("event_minute"),
        left_on="created_time",
        right_on="event_minute",
        direction="backward",
        tolerance=pd.Timedelta(minutes=1),
    )
    result = pd.Series(aligned["market_prob_realized_vol_4h"].to_numpy(dtype=np.float64), index=lookup.index)
    return result.reindex(created_times.index)


def _assign_realized_vol_bucket(metric: float, edges: Sequence[float]) -> str:
    if pd.isna(metric):
        return REALIZED_VOL_REGIME_UNKNOWN
    if metric < edges[0]:
        return REALIZED_VOL_REGIME_LABELS[0]
    if metric < edges[1]:
        return REALIZED_VOL_REGIME_LABELS[1]
    if metric < edges[2]:
        return REALIZED_VOL_REGIME_LABELS[2]
    return REALIZED_VOL_REGIME_LABELS[3]


def _session_block_for_timestamp(timestamp: pd.Timestamp) -> str:
    hour = int(timestamp.hour)
    minute = int(timestamp.minute)
    total_minutes = hour * 60 + minute
    if 20 * 60 <= total_minutes or total_minutes < 60:
        return "asia"
    if total_minutes < 7 * 60:
        return "europe"
    if total_minutes < 9 * 60 + 30:
        return "us_preopen"
    # BTC trades 24/7, but we still use the NYSE regular-hours window as the US macro/liquidity anchor.
    if total_minutes < 16 * 60:
        return "us_regular"
    return "us_postclose"


def _annotate_diagnostic_context(
    split_df: pd.DataFrame,
    *,
    external_spot_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    annotated = split_df.copy()
    annotated["created_time"] = pd.to_datetime(annotated["created_time"], utc=True)
    spot_metric = _align_spot_realized_vol_to_rows(annotated["created_time"], external_spot_root=external_spot_root)
    fallback_metric = _align_market_prob_realized_vol_to_rows(annotated["created_time"], annotated["market_prob"])
    source = np.where(~spot_metric.isna(), "spot", np.where(~fallback_metric.isna(), "market_prob", "unknown"))
    metric = np.where(source == "spot", spot_metric.to_numpy(dtype=np.float64), fallback_metric.to_numpy(dtype=np.float64))
    annotated["realized_vol_regime_source"] = source
    annotated["realized_vol_regime_metric"] = metric
    annotated["realized_vol_regime_bucket_version"] = np.where(
        annotated["realized_vol_regime_source"] == "spot",
        SPOT_REALIZED_VOL_REGIME_VERSION,
        np.where(
            annotated["realized_vol_regime_source"] == "market_prob",
            MARKET_PROB_REALIZED_VOL_REGIME_VERSION,
            "unknown",
        ),
    )
    annotated["realized_vol_regime_bucket"] = [
        _assign_realized_vol_bucket(value, SPOT_REALIZED_VOL_REGIME_EDGES if src == "spot" else MARKET_PROB_REALIZED_VOL_REGIME_EDGES)
        if src in {"spot", "market_prob"}
        else REALIZED_VOL_REGIME_UNKNOWN
        for value, src in zip(annotated["realized_vol_regime_metric"], annotated["realized_vol_regime_source"], strict=False)
    ]
    created_time_et = annotated["created_time"].dt.tz_convert(DIAGNOSTIC_TIMEZONE_INFO)
    annotated["hour_of_day_et"] = created_time_et.dt.hour.astype(int)
    annotated["hour_of_day_et_label"] = created_time_et.dt.strftime("%H")
    annotated["session_block_et"] = created_time_et.apply(_session_block_for_timestamp)

    total_rows = int(len(annotated))
    source_counts = annotated["realized_vol_regime_source"].value_counts(dropna=False).to_dict()
    source_summary = []
    for source_name in ("spot", "market_prob", "unknown"):
        count = int(source_counts.get(source_name, 0))
        source_summary.append(
            {
                "realized_vol_regime_source": source_name,
                "rows": count,
                "row_pct": (count / total_rows * 100.0) if total_rows else 0.0,
            }
        )
    metadata = {
        "lookback_minutes": REALIZED_VOL_LOOKBACK_MINUTES,
        "annualization_factor": REALIZED_VOL_ANNUALIZATION_FACTOR,
        "row_source_summary": source_summary,
        "bucket_versions": {
            "spot": {
                "version": SPOT_REALIZED_VOL_REGIME_VERSION,
                "edges": list(SPOT_REALIZED_VOL_REGIME_EDGES),
                "derived_from": SPOT_REALIZED_VOL_REGIME_DERIVED_FROM,
            },
            "market_prob": {
                "version": MARKET_PROB_REALIZED_VOL_REGIME_VERSION,
                "edges": list(MARKET_PROB_REALIZED_VOL_REGIME_EDGES),
                "derived_from": MARKET_PROB_REALIZED_VOL_REGIME_DERIVED_FROM,
            },
        },
        "revisit_note": REALIZED_VOL_REGIME_REVISIT_NOTE,
        "timezone_metadata": _timezone_metadata(),
    }
    return annotated, metadata


def _candidate_input_pairs(series_ticker: str) -> list[tuple[Path, Path]]:
    repo_root = Path(__file__).resolve().parents[3]
    candidates: list[tuple[Path, Path]] = []
    seen: set[tuple[Path, Path]] = set()
    for base in (
        repo_root / "output" / "kalshi_series_backfill",
        repo_root / "output" / "kalshi_series",
    ):
        markets_path = base / f"{series_ticker}_markets.parquet"
        for trades_path in (
            base / series_ticker / "trades",
            base / f"{series_ticker}_trades.parquet",
        ):
            if markets_path.exists() and trades_path.exists():
                pair = (markets_path, trades_path)
                if pair not in seen:
                    candidates.append(pair)
                    seen.add(pair)
    return candidates


def _candidate_raw_archives() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    candidates: list[Path] = []
    for candidate in (
        repo_root / "data" / "data" / "data" / "kalshi",
        repo_root / "data" / "kalshi",
    ):
        if (candidate / "markets").exists() and (candidate / "trades").exists():
            candidates.append(candidate)
    return candidates


def _format_candidate_input_pairs(series_ticker: str) -> str:
    candidates = _candidate_input_pairs(series_ticker)
    if not candidates:
        message = (
            "No standard input candidates were found under:\n"
            '  - "output\\kalshi_series_backfill"\n'
            '  - "output\\kalshi_series"'
        )
        raw_archives = _candidate_raw_archives()
        if raw_archives:
            archive_lines = "\n".join(f'  - "{candidate}"' for candidate in raw_archives)
            message += (
                "\nDetected local master parquet archive(s):\n"
                f"{archive_lines}\n"
                "You can extract the missing series locally with:\n"
                f'  uv run python scripts\\extract_kalshi_series_from_archive.py {series_ticker}'
            )
        return message

    lines = ["Detected local input candidates:"]
    for markets_path, trades_path in candidates:
        lines.append(f'  - --markets-path "{markets_path}" --trades-path "{trades_path}"')
    lines.append("You can also omit both flags and let the script pick the first detected candidate.")
    return "\n".join(lines)


@dataclass(frozen=True)
class SplitManifest:
    series: str
    generated_at: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    train_tickers: tuple[str, ...]
    validation_tickers: tuple[str, ...]
    test_tickers: tuple[str, ...]
    purge_embargo_enabled: bool = False
    purge_embargo_gap_seconds: int = 0
    feature_lookback_seconds: int = 0
    target_horizon_seconds: int = 0
    safety_margin_seconds: int = 0


@dataclass(frozen=True)
class PreparedExternalSpotFrame:
    frame: pd.DataFrame
    available_time_ns: np.ndarray


@dataclass(frozen=True)
class FeatureCacheStatus:
    ready: bool
    manifest_exists: bool
    manifest_schema_name: str | None
    schema_matches: bool
    requested_tickers: int
    present_requested_tickers: int
    missing_requested_tickers: tuple[str, ...]


@dataclass(frozen=True)
class PolicyConfig:
    edge_threshold_cents: float
    min_tau_minutes: float
    max_tau_minutes: float
    price_band_min_cents: int
    price_band_max_cents: int
    reserve_cash_pct: float
    maintain_edge_cents: float | None = None
    apply_regime_hard_gate: bool = False
    starting_cash_dollars: float = DEFAULT_STARTING_CASH_DOLLARS
    contracts_per_order: int = 1
    capital_pct_per_order: float | None = None
    kelly_fraction_multiplier: float | None = None
    kelly_fraction_cap_pct: float | None = None
    slippage_pct: float = 1.0
    allow_stacking: bool = False
    max_entries_per_ticker: int | None = None
    require_price_improvement_for_stack: bool = False

    @property
    def signal_config(self) -> KalshiSignalRiskConfig:
        maintain_edge_cents = (
            float(self.maintain_edge_cents)
            if self.maintain_edge_cents is not None
            else min(1.0, float(self.edge_threshold_cents))
        )
        return KalshiSignalRiskConfig(
            edge_threshold_cents=self.edge_threshold_cents,
            maintain_edge_cents=maintain_edge_cents,
            min_tau_minutes=self.min_tau_minutes,
            max_tau_minutes=self.max_tau_minutes,
            apply_regime_hard_gate=self.apply_regime_hard_gate,
            allow_stacking=self.allow_stacking,
            starting_cash_dollars=self.starting_cash_dollars,
            contracts_per_order=self.contracts_per_order,
            reserve_cash_pct=self.reserve_cash_pct,
            slippage_pct=self.slippage_pct,
            price_band_min_cents=self.price_band_min_cents,
            price_band_max_cents=self.price_band_max_cents,
        )


@dataclass(frozen=True)
class PolicyResult:
    config: PolicyConfig
    objective: float
    trades: int
    net_pnl_dollars: float
    max_drawdown_dollars: float
    max_drawdown_pct: float
    return_pct: float
    log_loss: float
    skipped_due_open_ticker: int = 0
    skipped_due_price_band: int = 0
    skipped_due_regime: int = 0
    skipped_due_post_cost_edge: int = 0


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train_tickers: tuple[str, ...]
    validation_tickers: tuple[str, ...]
    test_tickers: tuple[str, ...]


def _max_feature_lookback_seconds(
    *,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
) -> int:
    normalized_schema = normalize_feature_schema(feature_schema)
    lookback_seconds = max(int(window) for window in KalshiFeatureEngineConfig().rolling_window_seconds)
    if normalized_schema == SPOT_V1_FEATURE_SCHEMA:
        lookback_seconds = max(lookback_seconds, 1800)
    return lookback_seconds


def _infer_target_horizon_seconds(markets_df: pd.DataFrame) -> int:
    if "open_time" not in markets_df.columns or "close_time" not in markets_df.columns or markets_df.empty:
        return DEFAULT_TARGET_HORIZON_SECONDS
    open_times = pd.to_datetime(markets_df["open_time"], utc=True, errors="coerce")
    close_times = pd.to_datetime(markets_df["close_time"], utc=True, errors="coerce")
    durations = (close_times - open_times).dropna()
    if durations.empty:
        return DEFAULT_TARGET_HORIZON_SECONDS
    max_seconds = int(max(duration.total_seconds() for duration in durations if duration.total_seconds() > 0.0))
    return max_seconds if max_seconds > 0 else DEFAULT_TARGET_HORIZON_SECONDS


def compute_purge_embargo_gap_seconds(
    markets_df: pd.DataFrame,
    *,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
    safety_margin_seconds: int = DEFAULT_PURGE_EMBARGO_SAFETY_MARGIN_SECONDS,
) -> int:
    feature_lookback_seconds = _max_feature_lookback_seconds(feature_schema=feature_schema)
    target_horizon_seconds = _infer_target_horizon_seconds(markets_df)
    return int(feature_lookback_seconds + target_horizon_seconds + max(0, int(safety_margin_seconds)))


def _searchsorted_timestamp(close_times: pd.Series, cutoff: pd.Timestamp) -> int:
    close_time_ns = pd.to_datetime(close_times, utc=True).astype("int64", copy=False).to_numpy(copy=False)
    return int(np.searchsorted(close_time_ns, cutoff.value, side="left"))


@dataclass(frozen=True)
class RegularizedLogisticArtifact:
    scaler: StandardScaler
    model: LogisticRegression


LassoArtifact = RegularizedLogisticArtifact
ElasticNetArtifact = RegularizedLogisticArtifact


@dataclass(frozen=True)
class BaggedLassoArtifact:
    scaler: StandardScaler
    model: BaggingClassifier


@dataclass(frozen=True)
class LinearSVMArtifact:
    scaler: StandardScaler
    model: LinearSVC


@dataclass
class _OpenPosition:
    ticker: str
    side: str
    reference_price_cents: int
    close_time: pd.Timestamp
    entry_cost: float
    payout: float
    fees: float
    hold_minutes: float


@dataclass(frozen=True)
class _PreparedStandardPolicyInputs:
    created_time_ns: np.ndarray
    close_time_ns: np.ndarray
    ticker_codes: np.ndarray
    tau_minutes: np.ndarray
    actual_outcomes: np.ndarray
    reference_price_cents: np.ndarray
    post_cost_edge: np.ndarray
    cash_required_dollars: np.ndarray
    entry_cost_dollars: np.ndarray
    fees_dollars: np.ndarray
    payout_dollars: np.ndarray
    hold_minutes: np.ndarray
    max_close_time_ns: int


@dataclass(frozen=True)
class _StandardOpenPosition:
    ticker_code: int
    entry_cost: float
    payout: float
    fees: float
    hold_minutes: float


def resolve_default_inputs(series_ticker: str) -> tuple[Path, Path]:
    candidates = _candidate_input_pairs(series_ticker)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not find markets/trades for {series_ticker}.\n{_format_candidate_input_pairs(series_ticker)}"
    )


def resolve_inputs(
    series_ticker: str,
    markets_path: str | None,
    trades_path: str | None,
) -> tuple[Path, Path]:
    if markets_path and trades_path:
        resolved_markets_path = Path(markets_path).expanduser()
        resolved_trades_path = Path(trades_path).expanduser()
        missing: list[str] = []
        if not resolved_markets_path.exists():
            missing.append(f'Provided --markets-path does not exist: "{resolved_markets_path}"')
        if not resolved_trades_path.exists():
            missing.append(f'Provided --trades-path does not exist: "{resolved_trades_path}"')
        if missing:
            raise FileNotFoundError("\n".join([*missing, _format_candidate_input_pairs(series_ticker)]))
        return resolved_markets_path, resolved_trades_path
    if markets_path or trades_path:
        raise ValueError(
            "Pass both --markets-path and --trades-path together.\n"
            f"{_format_candidate_input_pairs(series_ticker)}"
        )
    return resolve_default_inputs(series_ticker)


def resolve_artifacts_dir(
    artifacts_root: str | None,
    run_name: str | None,
    *,
    default_root: Path | None = None,
) -> Path:
    root = Path(artifacts_root) if artifacts_root else (default_root or DEFAULT_ARTIFACTS_ROOT)
    if run_name:
        return root / run_name
    generated_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root / generated_at


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def normalize_feature_schema(feature_schema: str | None = None) -> str:
    normalized = str(feature_schema or DEFAULT_FEATURE_SCHEMA).strip().lower()
    if normalized not in FEATURE_SCHEMA_CHOICES:
        choices = ", ".join(FEATURE_SCHEMA_CHOICES)
        raise ValueError(f"Unsupported feature schema '{feature_schema}'. Expected one of: {choices}.")
    return normalized


def feature_schema_metadata(feature_schema: str | None = None) -> dict[str, object]:
    normalized = normalize_feature_schema(feature_schema)
    metadata: dict[str, object] = {"schema_name": normalized}
    if normalized == LINEAR_V1_FEATURE_SCHEMA:
        metadata["derived_feature_formulas"] = dict(LINEAR_V1_DERIVED_FORMULAS)
    feature_order = build_offline_feature_order(
        hourly_context=(HourlyContextConfig() if normalized == LINEAR_V1_FEATURE_SCHEMA else None),
        feature_schema=normalized,
    )
    metadata["feature_order_hash"] = feature_order_hash(feature_order)
    return metadata


def build_offline_feature_order(
    *,
    hourly_context: HourlyContextConfig | None = None,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
) -> tuple[str, ...]:
    normalized_schema = normalize_feature_schema(feature_schema)
    raw_feature_order = _BASE_FEATURE_ORDER if hourly_context is None else _BASE_FEATURE_ORDER + HOURLY_CONTEXT_FEATURE_ORDER
    if normalized_schema == DEFAULT_FEATURE_SCHEMA:
        return raw_feature_order
    if normalized_schema == LINEAR_V1_FEATURE_SCHEMA:
        if hourly_context is None:
            raise ValueError("The linear_v1 feature schema requires hourly context features.")
        return LINEAR_V1_FEATURE_ORDER
    if normalized_schema == SPOT_V1_FEATURE_SCHEMA:
        return tuple([*raw_feature_order, *SPOT_V1_FEATURE_ORDER])
    raise ValueError(f"Unsupported feature schema '{feature_schema}'.")


def infer_feature_names(df: pd.DataFrame) -> tuple[str, ...]:
    return tuple(str(column) for column in df.columns if str(column) not in _MARKET_FEATURE_METADATA_COLUMNS)


def save_feature_manifest(
    path: Path,
    *,
    feature_order: Sequence[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "feature_order": list(feature_order or _BASE_FEATURE_ORDER),
        "schema_name": DEFAULT_FEATURE_SCHEMA,
    }
    if metadata:
        payload.update(metadata)
    write_json(path, payload)


def _feature_order_from_manifest_payload(payload: dict[str, object] | None) -> tuple[str, ...] | None:
    if not payload:
        return None
    feature_order_value = payload.get("feature_order")
    if not isinstance(feature_order_value, list):
        return None
    return tuple(str(value) for value in feature_order_value)


def _load_feature_manifest_payload(cache_dir: Path) -> dict[str, object] | None:
    path = cache_dir / "feature_manifest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def inspect_feature_cache(
    cache_dir: Path,
    tickers: Sequence[str],
    *,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
) -> FeatureCacheStatus:
    normalized_schema = normalize_feature_schema(feature_schema)
    if not cache_dir.exists():
        return FeatureCacheStatus(
            ready=False,
            manifest_exists=False,
            manifest_schema_name=None,
            schema_matches=False,
            requested_tickers=len(tuple(tickers)),
            present_requested_tickers=0,
            missing_requested_tickers=tuple(str(ticker) for ticker in tickers),
        )
    manifest_payload = _load_feature_manifest_payload(cache_dir)
    manifest_exists = manifest_payload is not None
    manifest_schema_name: str | None = None
    schema_matches = False
    if manifest_payload is not None:
        raw_schema_name = manifest_payload.get("schema_name", DEFAULT_FEATURE_SCHEMA)
        try:
            manifest_schema_name = normalize_feature_schema(str(raw_schema_name))
        except ValueError:
            manifest_schema_name = str(raw_schema_name)
        schema_matches = manifest_schema_name == normalized_schema
    missing_requested_tickers = tuple(
        str(ticker)
        for ticker in tickers
        if not (cache_dir / f"{ticker}.parquet").exists()
    )
    requested_tickers = len(tuple(tickers))
    present_requested_tickers = requested_tickers - len(missing_requested_tickers)
    return FeatureCacheStatus(
        ready=manifest_exists and schema_matches and not missing_requested_tickers,
        manifest_exists=manifest_exists,
        manifest_schema_name=manifest_schema_name,
        schema_matches=schema_matches,
        requested_tickers=requested_tickers,
        present_requested_tickers=present_requested_tickers,
        missing_requested_tickers=missing_requested_tickers,
    )


def project_feature_frame(
    df: pd.DataFrame,
    *,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
) -> pd.DataFrame:
    normalized_schema = normalize_feature_schema(feature_schema)
    if normalized_schema == DEFAULT_FEATURE_SCHEMA:
        projected = df.copy()
        projected.attrs["feature_schema"] = normalized_schema
        return projected

    if normalized_schema == SPOT_V1_FEATURE_SCHEMA:
        projected = df.copy()
        projected.attrs["feature_schema"] = normalized_schema
        return projected

    if normalized_schema != LINEAR_V1_FEATURE_SCHEMA:
        raise ValueError(f"Unsupported feature schema '{feature_schema}'.")

    required_columns = [feature_name for feature_name in LINEAR_V1_FEATURE_ORDER if feature_name not in LINEAR_V1_DERIVED_FEATURE_ORDER]
    missing_columns = [feature_name for feature_name in required_columns if feature_name not in df.columns]
    if missing_columns:
        raise ValueError(
            "The linear_v1 feature schema requires hourly-context feature columns that are missing from the raw frame: "
            + ", ".join(missing_columns)
        )

    if df.empty:
        projected = pd.DataFrame(columns=_market_feature_frame_columns(LINEAR_V1_FEATURE_ORDER))
        projected.attrs["feature_schema"] = normalized_schema
        return projected

    projected = df.loc[:, [*_MARKET_FEATURE_METADATA_COLUMNS, *required_columns]].copy()
    tau_decay = df["tau_minutes"].astype(np.float64, copy=False).clip(lower=0.0) + 1.0
    projected["k15_z_tau_decay"] = (
        df["z_implied"].astype(np.float64, copy=False) / tau_decay
    ).astype(np.float32, copy=False)
    projected["k15_return_300s_tau_decay"] = (
        df["price_return_300s"].astype(np.float64, copy=False) / tau_decay
    ).astype(np.float32, copy=False)
    projected["k15_k1h_z_product"] = (
        df["z_implied"].astype(np.float64, copy=False)
        * df["kxbtcd_atm_z_implied"].astype(np.float64, copy=False)
    ).astype(np.float32, copy=False)
    projected["k15_k1h_return_product_300s"] = (
        df["price_return_300s"].astype(np.float64, copy=False)
        * df["kxbtcd_atm_price_return_300s"].astype(np.float64, copy=False)
    ).astype(np.float32, copy=False)
    projected = projected.loc[:, _market_feature_frame_columns(LINEAR_V1_FEATURE_ORDER)]
    projected.attrs["feature_schema"] = normalized_schema
    return projected


def load_markets_frame(markets_path: Path, series_ticker: str) -> pd.DataFrame:
    if markets_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(markets_path)
    elif markets_path.suffix.lower() == ".csv":
        df = pd.read_csv(markets_path)
    else:
        raise ValueError(f"Unsupported markets file type: {markets_path}")
    df = df[df["ticker"].astype(str).str.startswith(f"{series_ticker}-")].copy()
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    valid_results = df["result"].astype(str).str.lower().isin({"yes", "no"})
    return df.loc[valid_results].sort_values(["close_time", "ticker"]).reset_index(drop=True)


def build_split_manifest(
    markets_df: pd.DataFrame,
    *,
    series_ticker: str = DEFAULT_SERIES,
    train_fraction: float = 0.65,
    validation_fraction: float = 0.15,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
    purge_embargo: bool = False,
    purge_embargo_gap_seconds: int | None = None,
    purge_embargo_safety_margin_seconds: int = DEFAULT_PURGE_EMBARGO_SAFETY_MARGIN_SECONDS,
) -> SplitManifest:
    if markets_df.empty:
        raise ValueError("No resolved markets found to split.")
    ordered_markets = markets_df.sort_values(["close_time", "ticker"]).reset_index(drop=True)
    ordered_tickers = tuple(ordered_markets["ticker"].astype(str))
    total = len(ordered_tickers)
    train_end = max(1, int(total * train_fraction))
    validation_end = max(train_end + 1, int(total * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, total - 1)
    feature_lookback_seconds = _max_feature_lookback_seconds(feature_schema=feature_schema)
    target_horizon_seconds = _infer_target_horizon_seconds(ordered_markets)
    resolved_gap_seconds = (
        int(purge_embargo_gap_seconds)
        if purge_embargo_gap_seconds is not None
        else compute_purge_embargo_gap_seconds(
            ordered_markets,
            feature_schema=feature_schema,
            safety_margin_seconds=purge_embargo_safety_margin_seconds,
        )
    )
    if not purge_embargo:
        resolved_gap_seconds = 0
    if purge_embargo:
        if "close_time" not in ordered_markets.columns:
            raise ValueError("Purged/embargoed splits require close_time in the markets frame.")
        close_times = pd.to_datetime(ordered_markets["close_time"], utc=True)
        gap = pd.Timedelta(seconds=resolved_gap_seconds)
        train_end_time = pd.Timestamp(close_times.iloc[train_end - 1])
        validation_start_index = _searchsorted_timestamp(close_times, train_end_time + gap)
        if validation_start_index >= total - 1:
            raise ValueError("Purged/embargoed split leaves no room for validation and test tickers.")
        desired_validation_end = min(total, validation_start_index + max(1, int(total * validation_fraction)))
        validation_end_index = desired_validation_end
        test_start_index = total
        while validation_end_index > validation_start_index:
            validation_end_time = pd.Timestamp(close_times.iloc[validation_end_index - 1])
            test_start_index = _searchsorted_timestamp(close_times, validation_end_time + gap)
            if test_start_index < total:
                break
            validation_end_index -= 1
        if validation_end_index <= validation_start_index or test_start_index >= total:
            raise ValueError("Purged/embargoed split leaves no room for the test segment.")
        train_tickers = ordered_tickers[:train_end]
        validation_tickers = ordered_tickers[validation_start_index:validation_end_index]
        test_tickers = ordered_tickers[test_start_index:]
    else:
        train_tickers = ordered_tickers[:train_end]
        validation_tickers = ordered_tickers[train_end:validation_end]
        test_tickers = ordered_tickers[validation_end:]
    return SplitManifest(
        series=series_ticker,
        generated_at=datetime.now(UTC).isoformat(),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=1.0 - train_fraction - validation_fraction,
        train_tickers=train_tickers,
        validation_tickers=validation_tickers,
        test_tickers=test_tickers,
        purge_embargo_enabled=bool(purge_embargo),
        purge_embargo_gap_seconds=int(resolved_gap_seconds),
        feature_lookback_seconds=int(feature_lookback_seconds),
        target_horizon_seconds=int(target_horizon_seconds),
        safety_margin_seconds=int(purge_embargo_safety_margin_seconds) if purge_embargo else 0,
    )


def save_split_manifest(path: Path, manifest: SplitManifest) -> None:
    write_json(path, asdict(manifest))


def load_trades_for_ticker(trades_path: Path, ticker: str) -> pd.DataFrame:
    if trades_path.is_dir():
        path = trades_path / f"{ticker}.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["trade_id", "ticker", "count", "yes_price", "no_price", "taker_side", "created_time"])
        df = pd.read_parquet(path)
    else:
        df = pd.read_parquet(trades_path, filters=[[("ticker", "=", ticker)]])
    if df.empty:
        return df
    df["created_time"] = pd.to_datetime(df["created_time"], utc=True)
    return df.sort_values(["created_time", "trade_id"]).reset_index(drop=True)


def _market_feature_frame_columns(
    feature_order: Sequence[str] | None = None,
) -> list[str]:
    return [*_MARKET_FEATURE_METADATA_COLUMNS, *(feature_order or _BASE_FEATURE_ORDER)]


def build_market_feature_frame(
    trades_df: pd.DataFrame,
    market_row: pd.Series,
    *,
    feature_config: KalshiFeatureEngineConfig | None = None,
) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=_market_feature_frame_columns())

    accumulator = KalshiTradeFeatureAccumulator(config=feature_config)
    ticker = str(market_row["ticker"])
    result_label = str(market_row["result"]).lower()
    actual_outcome = 1 if result_label == "yes" else 0
    close_time = pd.to_datetime(market_row.get("close_time"), utc=True)
    open_time = pd.to_datetime(market_row.get("open_time"), utc=True) if market_row.get("open_time") is not None else None
    rows: list[dict[str, object]] = []

    previous_yes_price_cents: int | None = None
    for row in trades_df.itertuples(index=False):
        yes_price_cents = int(row.yes_price)
        update = KalshiTickerUpdate(
            ticker=ticker,
            event_time=row.created_time,
            last_yes_price_cents=yes_price_cents,
            previous_yes_price_cents=previous_yes_price_cents,
            close_time=close_time,
            is_open=True,
            market_prob=None,
            previous_market_prob=None,
            price_momentum=None,
            tau_minutes=None,
            open_time=open_time,
            trade_id=str(row.trade_id),
            count=int(row.count),
            taker_side=str(row.taker_side),
            last_trade_time=row.created_time,
            last_price_cents=yes_price_cents,
            source="trade",
        )
        state = accumulator.build_feature_state_from_update(update)
        previous_yes_price_cents = yes_price_cents
        if not state.in_training_window or state.feature_values() is None or state.market_prob is None or state.tau_minutes is None:
            continue
        payload = {
            "ticker": ticker,
            "trade_id": str(row.trade_id),
            "created_time": row.created_time,
            "open_time": open_time,
            "close_time": close_time,
            "actual_outcome": actual_outcome,
            "market_prob": state.market_prob,
            "tau_minutes": state.tau_minutes,
            "count": int(row.count),
            "taker_side": str(row.taker_side),
        }
        payload.update(dict(zip(FEATURE_ORDER, state.feature_values())))
        rows.append(payload)
    return pd.DataFrame(rows, columns=_market_feature_frame_columns())


def load_external_spot_frame(external_spot_root: Path) -> pd.DataFrame:
    root = Path(external_spot_root)
    if not root.exists():
        raise FileNotFoundError(f"External spot root does not exist: {root}")
    if root.is_file():
        frames = [pd.read_parquet(root)]
    else:
        parquet_paths = sorted(root.rglob("*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No parquet files found under external spot root: {root}")
        frames = [pd.read_parquet(path) for path in parquet_paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if df.empty:
        return df
    if "event_time" not in df.columns:
        raise ValueError("External spot frame must include an event_time column.")
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    if "received_at" in df.columns:
        df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
        return df.sort_values(["received_at", "event_time"]).reset_index(drop=True)
    return df.sort_values("event_time").reset_index(drop=True)


def load_or_require_external_spot_frame(
    feature_schema: str | None,
    external_spot_root: str | Path | None,
) -> pd.DataFrame | None:
    normalized_schema = normalize_feature_schema(feature_schema)
    if normalized_schema != SPOT_V1_FEATURE_SCHEMA:
        return None
    if external_spot_root is None:
        raise ValueError("The spot_v1 feature schema requires --external-spot-root.")
    return load_external_spot_frame(Path(external_spot_root).expanduser())


def _prepare_external_spot_frame_for_cache_build(
    external_spot_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    *,
    external_spot_latency_ms: int = 0,
) -> PreparedExternalSpotFrame:
    if external_spot_df.empty:
        empty = external_spot_df.copy()
        empty["available_time"] = pd.Series(dtype="datetime64[ns, UTC]")
        return PreparedExternalSpotFrame(frame=empty, available_time_ns=np.array([], dtype=np.int64))
    spot_rows = external_spot_df.copy()
    spot_rows["event_time"] = pd.to_datetime(spot_rows["event_time"], utc=True)
    if "received_at" in spot_rows.columns:
        spot_rows["received_at"] = pd.to_datetime(spot_rows["received_at"], utc=True)
        availability_base = spot_rows["received_at"]
    else:
        availability_base = spot_rows["event_time"]
    spot_rows["available_time"] = availability_base + pd.to_timedelta(int(external_spot_latency_ms), unit="ms")
    spot_rows = spot_rows.sort_values(["available_time", "event_time"]).reset_index(drop=True)
    if not markets_df.empty and "close_time" in markets_df.columns:
        market_end = pd.to_datetime(markets_df["close_time"], utc=True).max()
        market_start_candidates: list[pd.Timestamp] = [market_end]
        if "open_time" in markets_df.columns:
            open_times = pd.to_datetime(markets_df["open_time"], utc=True)
            if not open_times.empty:
                market_start_candidates.append(open_times.min())
        available_time_ns = spot_rows["available_time"].astype("int64", copy=False).to_numpy()
        market_start_ns = min(timestamp.value for timestamp in market_start_candidates)
        market_end_ns = market_end.value
        start_idx = max(int(np.searchsorted(available_time_ns, market_start_ns, side="right")) - 1, 0)
        end_idx = int(np.searchsorted(available_time_ns, market_end_ns, side="right"))
        if end_idx <= start_idx:
            end_idx = min(len(spot_rows), start_idx + 1)
        spot_rows = spot_rows.iloc[start_idx:end_idx].reset_index(drop=True)
    return PreparedExternalSpotFrame(
        frame=spot_rows,
        available_time_ns=spot_rows["available_time"].astype("int64", copy=False).to_numpy(),
    )


def _slice_prepared_external_spot_frame_for_market_window(
    prepared_spot: PreparedExternalSpotFrame,
    *,
    market_row: pd.Series,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    if target_df.empty or prepared_spot.frame.empty:
        return prepared_spot.frame.iloc[0:0].copy()
    market_start_candidates = [pd.Timestamp(target_df["created_time"].min())]
    open_time = market_row.get("open_time")
    if open_time is not None and not pd.isna(open_time):
        market_start_candidates.append(pd.Timestamp(open_time))
    market_start_ns = min(timestamp.value for timestamp in market_start_candidates)
    market_end_ns = pd.Timestamp(target_df["created_time"].max()).value
    available_time_ns = prepared_spot.available_time_ns
    start_idx = max(int(np.searchsorted(available_time_ns, market_start_ns, side="right")) - 1, 0)
    end_idx = int(np.searchsorted(available_time_ns, market_end_ns, side="right"))
    if end_idx <= start_idx:
        end_idx = min(len(prepared_spot.frame), start_idx + 1)
    return prepared_spot.frame.iloc[start_idx:end_idx].reset_index(drop=True)


def enrich_market_feature_frame_with_external_spot(
    target_df: pd.DataFrame,
    *,
    market_row: pd.Series,
    external_spot_df: pd.DataFrame,
    external_spot_latency_ms: int = 0,
) -> pd.DataFrame:
    feature_order = build_offline_feature_order(feature_schema=SPOT_V1_FEATURE_SCHEMA)
    if target_df.empty:
        return pd.DataFrame(columns=_market_feature_frame_columns(feature_order))
    if external_spot_df.empty:
        raise ValueError("External spot frame is empty; cannot build spot_v1 features.")
    required_columns = {
        "event_time",
        "btc_spot_price",
        "btc_spot_twap_60s",
        "btc_spot_age_ms",
        "btc_spot_is_fresh",
        "btc_spot_venues_fresh",
        "btc_spot_venue_divergence_bps",
        "btc_vol_effective_sample_size",
        "btc_spot_source",
        "btc_spot_return_30s",
        "btc_spot_return_120s",
        "btc_spot_return_300s",
        "btc_spot_return_900s",
        "btc_spot_vol_120s",
        "btc_spot_vol_300s",
        "btc_spot_vol_900s",
        "btc_spot_vol_1800s",
        "btc_spot_vol_ewma_hl300",
    }
    missing = sorted(required_columns.difference(external_spot_df.columns))
    if missing:
        raise ValueError(f"External spot frame is missing required columns: {', '.join(missing)}")
    strike_price = strike_price_from_market(market_row.to_dict())
    if strike_price is None:
        strike_price = _fallback_spot_open_strike_price(market_row, external_spot_df)
    if strike_price is None:
        raise ValueError(f"Could not determine strike price for {market_row.get('ticker')}.")
    if "available_time" in external_spot_df.columns and int(external_spot_latency_ms) == 0:
        spot_rows = external_spot_df
    else:
        spot_rows = external_spot_df.copy()
        latency = pd.to_timedelta(int(external_spot_latency_ms), unit="ms")
        if "received_at" in spot_rows.columns:
            spot_rows["available_time"] = pd.to_datetime(spot_rows["received_at"], utc=True) + latency
        else:
            spot_rows["available_time"] = pd.to_datetime(spot_rows["event_time"], utc=True) + latency
    tracker = SpotFeatureTracker()
    spot_pointer = -1
    selected_row: pd.Series | None = None
    records: list[dict[str, object]] = []
    for target_row in target_df.itertuples(index=False):
        payload = target_row._asdict()
        target_time = pd.Timestamp(payload["created_time"])
        while spot_pointer + 1 < len(spot_rows) and pd.Timestamp(spot_rows.iloc[spot_pointer + 1]["available_time"]) <= target_time:
            spot_pointer += 1
            selected_row = spot_rows.iloc[spot_pointer]
        if selected_row is None:
            raise ValueError(f"No external spot snapshot available before {target_time} for {payload['ticker']}.")
        spot_update = BTCSpotUpdate(
            event_time=pd.Timestamp(selected_row["event_time"]).to_pydatetime(warn=False),
            received_at=(
                pd.Timestamp(selected_row["received_at"]).to_pydatetime(warn=False)
                if "received_at" in selected_row.index
                else pd.Timestamp(selected_row["event_time"]).to_pydatetime(warn=False)
            ),
            btc_spot_price=float(selected_row["btc_spot_price"]),
            btc_spot_twap_60s=float(selected_row["btc_spot_twap_60s"]),
            btc_spot_age_ms=float(selected_row["btc_spot_age_ms"]),
            btc_spot_is_fresh=bool(selected_row["btc_spot_is_fresh"]),
            btc_spot_venues_fresh=int(selected_row["btc_spot_venues_fresh"]),
            btc_spot_venue_divergence_bps=float(selected_row["btc_spot_venue_divergence_bps"]),
            btc_vol_effective_sample_size=float(selected_row["btc_vol_effective_sample_size"]),
            btc_spot_source=str(selected_row["btc_spot_source"]),
            btc_spot_return_30s=float(selected_row["btc_spot_return_30s"]),
            btc_spot_return_120s=float(selected_row["btc_spot_return_120s"]),
            btc_spot_return_300s=float(selected_row["btc_spot_return_300s"]),
            btc_spot_return_900s=float(selected_row["btc_spot_return_900s"]),
            btc_spot_vol_120s=float(selected_row["btc_spot_vol_120s"]),
            btc_spot_vol_300s=float(selected_row["btc_spot_vol_300s"]),
            btc_spot_vol_900s=float(selected_row["btc_spot_vol_900s"]),
            btc_spot_vol_1800s=float(selected_row["btc_spot_vol_1800s"]),
            btc_spot_vol_ewma_hl300=float(selected_row["btc_spot_vol_ewma_hl300"]),
        )
        payload.update(
            tracker.enrich(
                event_time=target_time.to_pydatetime(),
                quote_mid_prob=float(payload.get("market_prob")) if payload.get("market_prob") is not None else None,
                tau_minutes=float(payload.get("tau_minutes")) if payload.get("tau_minutes") is not None else None,
                strike_price=strike_price,
                spot_update=spot_update,
            )
        )
        records.append(payload)
    enriched = pd.DataFrame(records)
    return enriched.loc[:, _market_feature_frame_columns(feature_order)]


def _fallback_spot_open_strike_price(
    market_row: pd.Series,
    external_spot_df: pd.DataFrame,
) -> float | None:
    title = str(market_row.get("title") or "").strip().lower()
    if title != _GENERIC_BTC_DIRECTION_TITLE:
        return None
    open_time_value = market_row.get("open_time")
    if open_time_value is None or pd.isna(open_time_value):
        return None
    open_time = pd.Timestamp(open_time_value)
    if open_time.tzinfo is None:
        open_time = open_time.tz_localize("UTC")
    else:
        open_time = open_time.tz_convert("UTC")
    event_times = pd.to_datetime(external_spot_df["event_time"], utc=True)
    prior_or_open = external_spot_df.loc[event_times <= open_time]
    if prior_or_open.empty:
        candidate_frame = external_spot_df.loc[event_times >= open_time]
        if candidate_frame.empty:
            return None
        candidate_row = candidate_frame.iloc[0]
    else:
        candidate_row = prior_or_open.iloc[-1]
    strike_price = float(candidate_row["btc_spot_price"])
    LOGGER.warning(
        "Falling back to external spot at market open for %s because strike metadata was unavailable.",
        market_row.get("ticker"),
    )
    return strike_price


def _context_cache_dir(cache_dir: Path, series_ticker: str) -> Path:
    return cache_dir / f"__context_{series_ticker.lower()}"


def _context_feature_config(hourly_context: HourlyContextConfig) -> KalshiFeatureEngineConfig:
    return KalshiFeatureEngineConfig(tau_max_minutes=float(hourly_context.tau_max_minutes))


def _direction_agreement(left_value: float, right_value: float) -> float:
    return float(np.sign(left_value) * np.sign(right_value))


def _empty_hourly_context_payload() -> dict[str, float]:
    return {feature_name: 0.0 for feature_name in HOURLY_CONTEXT_FEATURE_ORDER}


def _hourly_context_payload(
    *,
    target_row: dict[str, object],
    context_row: pd.Series | None,
) -> dict[str, float]:
    if context_row is None:
        return _empty_hourly_context_payload()

    context_z_implied = float(context_row["z_implied"])
    context_price_return_300s = float(context_row["price_return_300s"])
    target_z_implied = float(target_row["z_implied"])
    target_price_return_300s = float(target_row["price_return_300s"])
    return {
        "kxbtcd_atm_z_implied": context_z_implied,
        "kxbtcd_atm_price_momentum": float(context_row["price_momentum"]),
        "kxbtcd_atm_abs_price_momentum": float(context_row["abs_price_momentum"]),
        "kxbtcd_atm_distance_from_mid": float(context_row["distance_from_mid"]),
        "kxbtcd_atm_trade_count_300s": float(context_row["trade_count_300s"]),
        "kxbtcd_atm_signed_contracts_sum_300s": float(context_row["signed_contracts_sum_300s"]),
        "kxbtcd_atm_price_return_300s": context_price_return_300s,
        "kxbtcd_atm_price_volatility_300s": float(context_row["price_volatility_300s"]),
        "kxbtcd_atm_yes_taker_share_300s": float(context_row["yes_taker_share_300s"]),
        "k15_minus_k1h_atm_z": target_z_implied - context_z_implied,
        "k15_minus_k1h_atm_price_return_300s": target_price_return_300s - context_price_return_300s,
        "k15_k1h_atm_direction_agreement": _direction_agreement(
            target_price_return_300s,
            context_price_return_300s,
        ),
    }


def _build_context_feature_cache(
    context_markets_df: pd.DataFrame,
    context_trades_path: Path,
    context_cache_dir: Path,
    hourly_context: HourlyContextConfig,
) -> None:
    context_cache_dir.mkdir(parents=True, exist_ok=True)
    feature_config = _context_feature_config(hourly_context)
    for market_row in context_markets_df.itertuples(index=False):
        path = context_cache_dir / f"{market_row.ticker}.parquet"
        if path.exists():
            continue
        feature_df = build_market_feature_frame(
            load_trades_for_ticker(context_trades_path, market_row.ticker),
            pd.Series(market_row._asdict()),
            feature_config=feature_config,
        )
        feature_df.to_parquet(path, index=False)


def _load_context_feature_frame(
    context_cache_dir: Path,
    ticker: str,
    *,
    frame_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    cached = frame_cache.get(ticker)
    if cached is not None:
        return cached
    path = context_cache_dir / f"{ticker}.parquet"
    if not path.exists():
        frame = pd.DataFrame(columns=_market_feature_frame_columns())
    else:
        frame = pd.read_parquet(path)
        if not frame.empty:
            frame["created_time"] = pd.to_datetime(frame["created_time"], utc=True)
            frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
            frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
            frame = frame.sort_values(["created_time", "trade_id"]).reset_index(drop=True)
    frame_cache[ticker] = frame
    return frame


def enrich_market_feature_frame_with_hourly_context(
    target_df: pd.DataFrame,
    *,
    context_markets_df: pd.DataFrame,
    context_cache_dir: Path,
    hourly_context: HourlyContextConfig,
    frame_cache: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    feature_order = build_offline_feature_order(hourly_context=hourly_context)
    if target_df.empty:
        return pd.DataFrame(columns=_market_feature_frame_columns(feature_order))

    frame_cache = frame_cache if frame_cache is not None else {}
    min_time = pd.Timestamp(target_df["created_time"].min())
    max_time = pd.Timestamp(target_df["created_time"].max())
    overlapping_context = context_markets_df.loc[
        (context_markets_df["open_time"] <= max_time) & (context_markets_df["close_time"] > min_time),
        ["ticker", "open_time", "close_time"],
    ].sort_values(["close_time", "ticker"])

    if overlapping_context.empty:
        enriched = target_df.copy()
        for feature_name, default_value in _empty_hourly_context_payload().items():
            enriched[feature_name] = default_value
        return enriched.loc[:, _market_feature_frame_columns(feature_order)]

    candidate_tickers = tuple(overlapping_context["ticker"].astype(str))
    context_frames = {
        ticker: _load_context_feature_frame(context_cache_dir, ticker, frame_cache=frame_cache)
        for ticker in candidate_tickers
    }
    context_pointers = {ticker: -1 for ticker in candidate_tickers}
    current_context_rows: dict[str, pd.Series | None] = {ticker: None for ticker in candidate_tickers}
    empty_payload = _empty_hourly_context_payload()
    records: list[dict[str, object]] = []

    for target_row in target_df.itertuples(index=False):
        target_payload = target_row._asdict()
        target_time = pd.Timestamp(target_payload["created_time"])
        for ticker, frame in context_frames.items():
            if frame.empty:
                continue
            pointer = context_pointers[ticker]
            while pointer + 1 < len(frame) and pd.Timestamp(frame.iloc[pointer + 1]["created_time"]) <= target_time:
                pointer += 1
                current_context_rows[ticker] = frame.iloc[pointer]
            context_pointers[ticker] = pointer

        active_context = overlapping_context.loc[
            (overlapping_context["open_time"] <= target_time) & (overlapping_context["close_time"] > target_time)
        ]
        selected_context_row: pd.Series | None = None
        if not active_context.empty:
            nearest_close_time = active_context["close_time"].min()
            expiry_candidates = active_context.loc[active_context["close_time"] == nearest_close_time]
            best_rank: tuple[float, float, float, float, str] | None = None
            for context_meta in expiry_candidates.itertuples(index=False):
                context_row = current_context_rows.get(str(context_meta.ticker))
                if context_row is None:
                    continue
                context_created_time = pd.Timestamp(context_row["created_time"])
                age_seconds = max(0.0, (target_time - context_created_time).total_seconds())
                if age_seconds > hourly_context.max_staleness_seconds:
                    continue
                trade_count_300s = float(context_row["trade_count_300s"])
                if trade_count_300s < hourly_context.min_recent_trade_count_300s:
                    continue
                context_market_prob = context_row.get("market_prob")
                if context_market_prob is None or pd.isna(context_market_prob):
                    continue
                rank = (
                    abs(float(context_market_prob) - 0.5),
                    -trade_count_300s,
                    -float(context_row.get("contracts_sum_300s", 0.0)),
                    age_seconds,
                    str(context_meta.ticker),
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    selected_context_row = context_row

        target_payload.update(_hourly_context_payload(target_row=target_payload, context_row=selected_context_row))
        if selected_context_row is None:
            for feature_name, default_value in empty_payload.items():
                target_payload.setdefault(feature_name, default_value)
        records.append(target_payload)

    return pd.DataFrame(records, columns=_market_feature_frame_columns(feature_order))


def build_feature_cache(
    markets_df: pd.DataFrame,
    trades_path: Path,
    cache_dir: Path,
    *,
    hourly_context: HourlyContextConfig | None = None,
    context_markets_df: pd.DataFrame | None = None,
    context_trades_path: Path | None = None,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
    external_spot_df: pd.DataFrame | None = None,
    external_spot_latency_ms: int = 0,
    progress_desc: str | None = None,
) -> None:
    normalized_schema = normalize_feature_schema(feature_schema)
    cache_dir.mkdir(parents=True, exist_ok=True)
    context_cache_dir: Path | None = None
    context_frame_cache: dict[str, pd.DataFrame] = {}
    prepared_external_spot: PreparedExternalSpotFrame | None = None
    if hourly_context is not None:
        if context_markets_df is None or context_trades_path is None:
            raise ValueError("Hourly context build requires both context_markets_df and context_trades_path.")
        context_cache_dir = _context_cache_dir(cache_dir, hourly_context.series_ticker)
        _build_context_feature_cache(context_markets_df, context_trades_path, context_cache_dir, hourly_context)
    if normalized_schema == SPOT_V1_FEATURE_SCHEMA:
        if external_spot_df is None:
            raise ValueError("spot_v1 cache builds require external_spot_df.")
        prepared_external_spot = _prepare_external_spot_frame_for_cache_build(
            external_spot_df,
            markets_df,
            external_spot_latency_ms=int(external_spot_latency_ms),
        )
    progress = tqdm(total=len(markets_df), desc=progress_desc, unit="ticker", leave=False) if progress_desc else None
    try:
        for market_row in markets_df.itertuples(index=False):
            path = cache_dir / f"{market_row.ticker}.parquet"
            if path.exists():
                if progress is not None:
                    progress.update(1)
                continue
            market_series = pd.Series(market_row._asdict())
            feature_df = build_market_feature_frame(
                load_trades_for_ticker(trades_path, market_row.ticker),
                market_series,
            )
            if hourly_context is not None and context_markets_df is not None and context_cache_dir is not None:
                feature_df = enrich_market_feature_frame_with_hourly_context(
                    feature_df,
                    context_markets_df=context_markets_df,
                    context_cache_dir=context_cache_dir,
                    hourly_context=hourly_context,
                    frame_cache=context_frame_cache,
                )
            if normalized_schema == SPOT_V1_FEATURE_SCHEMA:
                assert prepared_external_spot is not None
                market_spot_df = _slice_prepared_external_spot_frame_for_market_window(
                    prepared_external_spot,
                    market_row=market_series,
                    target_df=feature_df,
                )
                feature_df = enrich_market_feature_frame_with_external_spot(
                    feature_df,
                    market_row=market_series,
                    external_spot_df=market_spot_df,
                    external_spot_latency_ms=0,
                )
            feature_df.to_parquet(path, index=False)
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    save_feature_manifest(
        cache_dir / "feature_manifest.json",
        feature_order=build_offline_feature_order(hourly_context=hourly_context, feature_schema=normalized_schema),
        metadata={
            **feature_schema_metadata(normalized_schema),
            "hourly_context_series": hourly_context.series_ticker if hourly_context is not None else None,
            "external_spot_latency_ms": int(external_spot_latency_ms),
        },
    )


def load_feature_dataset(
    cache_dir: Path,
    tickers: tuple[str, ...],
    *,
    progress_desc: str | None = None,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
) -> pd.DataFrame:
    normalized_schema = normalize_feature_schema(feature_schema)
    manifest_payload = _load_feature_manifest_payload(cache_dir)
    frames: list[pd.DataFrame] = []
    found_any_cache_file = False
    progress = tqdm(total=len(tickers), desc=progress_desc, unit="ticker", leave=False) if progress_desc else None
    try:
        for ticker in tickers:
            path = cache_dir / f"{ticker}.parquet"
            if path.exists():
                found_any_cache_file = True
                frame = pd.read_parquet(path)
                if not frame.empty:
                    frames.append(frame)
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    if not frames:
        if not found_any_cache_file:
            raise ValueError("Requested split contains no cached feature files.")
        raw_feature_order = _feature_order_from_manifest_payload(manifest_payload) or _BASE_FEATURE_ORDER
        return project_feature_frame(
            pd.DataFrame(columns=_market_feature_frame_columns(raw_feature_order)),
            feature_schema=normalized_schema,
        )
    df = pd.concat(frames, ignore_index=True)
    df["created_time"] = pd.to_datetime(df["created_time"], utc=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    df = df.sort_values(["created_time", "ticker", "trade_id"]).reset_index(drop=True)
    return project_feature_frame(df, feature_schema=normalized_schema)


def _require_non_empty_split(name: str, df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError(f"{name} split contains no feature rows.")


def _binary_log_loss(actuals: np.ndarray, probabilities: np.ndarray) -> float:
    return float(log_loss(actuals, probabilities, labels=_BINARY_CLASS_LABELS))


def sample_train_subset(train_df: pd.DataFrame, sample_size: int, seed: int = ML_RANDOM_SEED) -> pd.DataFrame:
    if sample_size <= 0 or len(train_df) <= sample_size:
        return train_df
    labels = train_df["actual_outcome"].to_numpy(dtype=np.int8)
    idx = np.arange(len(train_df))
    sampled_idx, _rest = train_test_split(
        idx,
        train_size=sample_size,
        random_state=seed,
        stratify=labels,
    )
    return train_df.iloc[np.sort(sampled_idx)].reset_index(drop=True)


def feature_matrix(
    df: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    selected_feature_names = tuple(feature_names or infer_feature_names(df))
    return df.loc[:, list(selected_feature_names)].astype(np.float32, copy=False)


def labels(df: pd.DataFrame) -> np.ndarray:
    return df["actual_outcome"].to_numpy(dtype=np.int8, copy=True)


def _optuna_import() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("optuna is required to run the LightGBM hyperparameter search.") from exc
    return optuna


def run_optuna_search(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    n_trials: int = DEFAULT_OPTUNA_TRIALS,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    optuna = _optuna_import()
    _require_non_empty_split("train", train_df)
    _require_non_empty_split("validation", validation_df)
    X_train = feature_matrix(train_df)
    y_train = labels(train_df)
    X_valid = feature_matrix(validation_df)
    y_valid = labels(validation_df)
    history: list[dict[str, object]] = []

    def objective(trial: Any) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 8.0),
        }
        model = LGBMClassifier(
            objective="binary",
            random_state=ML_RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(100, verbose=False)],
        )
        valid_pred = np.clip(model.predict_proba(X_valid)[:, 1], 1e-6, 1.0 - 1e-6)
        valid_loss = _binary_log_loss(y_valid, valid_pred)
        history.append(
            {
                "trial": trial.number,
                "params": params,
                "validation_log_loss": valid_loss,
                "best_iteration": int(getattr(model, "best_iteration_", -1) or -1),
            }
        )
        return valid_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_trial.params), history


def train_lightgbm_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    params: dict[str, object],
) -> tuple[LGBMClassifier, dict[str, object]]:
    _require_non_empty_split("train", train_df)
    _require_non_empty_split("validation", validation_df)
    feature_names = infer_feature_names(train_df)
    X_train = feature_matrix(train_df, feature_names)
    y_train = labels(train_df)
    X_valid = feature_matrix(validation_df, feature_names)
    y_valid = labels(validation_df)
    model = LGBMClassifier(
        objective="binary",
        random_state=ML_RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(100, verbose=False)],
    )
    valid_pred = np.clip(model.predict_proba(X_valid)[:, 1], 1e-6, 1.0 - 1e-6)
    metrics = {
        "feature_names": list(feature_names),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "best_iteration": int(getattr(model, "best_iteration_", -1) or -1),
        "validation_log_loss": _binary_log_loss(y_valid, valid_pred),
        "params": params,
    }
    return model, metrics


def _lasso_feature_matrix(df: pd.DataFrame, feature_names: Sequence[str] | None = None) -> np.ndarray:
    return feature_matrix(df, feature_names).to_numpy(dtype=np.float64, copy=False)


def _feature_schema_for_frame(df: pd.DataFrame) -> str:
    return normalize_feature_schema(df.attrs.get("feature_schema", DEFAULT_FEATURE_SCHEMA))


def _sigmoid_from_scores(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=np.float64), -50.0, 50.0)
    return np.clip(1.0 / (1.0 + np.exp(-clipped)), 1e-6, 1.0 - 1e-6)


def _fit_regularized_logistic_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    c_value: float,
    l1_ratio: float,
    fit_intercept: bool = True,
    max_iter: int = DEFAULT_LASSO_MAX_ITER,
    tol: float = DEFAULT_LASSO_TOL,
) -> LogisticRegression:
    model = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=float(l1_ratio),
        C=float(c_value),
        solver="saga",
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        random_state=ML_RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def _fit_bagged_lasso_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    c_value: float,
    fit_intercept: bool = True,
    n_estimators: int = DEFAULT_BAGGED_LASSO_N_ESTIMATORS,
    max_samples: float = DEFAULT_BAGGED_LASSO_MAX_SAMPLES,
    max_iter: int = DEFAULT_BAGGED_LASSO_MAX_ITER,
    tol: float = DEFAULT_BAGGED_LASSO_TOL,
    n_jobs: int = DEFAULT_BAGGED_LASSO_N_JOBS,
) -> BaggingClassifier:
    base_estimator = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=1.0,
        C=float(c_value),
        solver="saga",
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        random_state=ML_RANDOM_SEED,
    )
    model = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=int(n_estimators),
        max_samples=float(max_samples),
        bootstrap=True,
        random_state=ML_RANDOM_SEED,
        n_jobs=int(n_jobs),
    )
    model.fit(X_train, y_train)
    return model


def _fit_linear_svm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    c_value: float,
    fit_intercept: bool = True,
    max_iter: int = DEFAULT_LINEAR_SVM_MAX_ITER,
    tol: float = DEFAULT_LINEAR_SVM_TOL,
) -> LinearSVC:
    model = LinearSVC(
        penalty="l1",
        loss="squared_hinge",
        dual=False,
        C=float(c_value),
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        random_state=ML_RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def _bagged_lasso_estimator_matrix(model: BaggingClassifier) -> np.ndarray:
    rows: list[np.ndarray] = []
    for estimator in model.estimators_:
        coef = getattr(estimator, "coef_", None)
        if coef is None:
            continue
        rows.append(np.asarray(coef[0], dtype=np.float64))
    if not rows:
        raise RuntimeError("Bagged LASSO did not expose any fitted estimator coefficients.")
    return np.vstack(rows)


def _bagged_lasso_intercepts(model: BaggingClassifier) -> np.ndarray:
    intercepts: list[float] = []
    for estimator in model.estimators_:
        intercept = getattr(estimator, "intercept_", None)
        if intercept is None:
            continue
        intercepts.append(float(intercept[0]))
    if not intercepts:
        raise RuntimeError("Bagged LASSO did not expose any fitted estimator intercepts.")
    return np.asarray(intercepts, dtype=np.float64)


def _bagged_lasso_iteration_counts(model: BaggingClassifier) -> list[int]:
    iterations: list[int] = []
    for estimator in model.estimators_:
        n_iter = getattr(estimator, "n_iter_", None)
        if n_iter is None:
            continue
        iterations.append(int(np.max(n_iter)))
    return iterations


def _bagged_lasso_statistics(model: BaggingClassifier) -> dict[str, object]:
    coefficient_matrix = _bagged_lasso_estimator_matrix(model)
    intercepts = _bagged_lasso_intercepts(model)
    selection_frequency = np.mean(np.abs(coefficient_matrix) > 0.0, axis=0)
    iteration_counts = _bagged_lasso_iteration_counts(model)
    return {
        "mean_coefficients": coefficient_matrix.mean(axis=0),
        "coefficient_std": coefficient_matrix.std(axis=0),
        "selection_frequency": selection_frequency,
        "mean_nonzero_coefficients": float(np.mean(np.count_nonzero(coefficient_matrix, axis=1))),
        "selected_feature_count_any_estimator": int(np.count_nonzero(selection_frequency > 0.0)),
        "selected_feature_count_majority_vote": int(np.count_nonzero(selection_frequency >= 0.5)),
        "mean_intercept": float(intercepts.mean()),
        "intercept_std": float(intercepts.std()),
        "mean_iterations_used": float(np.mean(iteration_counts)) if iteration_counts else None,
        "max_iterations_used": int(max(iteration_counts)) if iteration_counts else None,
    }


def _bagged_lasso_positive_class_probabilities(
    model: BaggingClassifier,
    X: np.ndarray,
) -> np.ndarray:
    estimator_count = len(model.estimators_)
    if estimator_count == 0:
        raise RuntimeError("Bagged LASSO model has no fitted estimators.")

    positive_sum = np.zeros(X.shape[0], dtype=np.float64)
    all_features = np.arange(X.shape[1], dtype=np.intp)
    estimator_features = getattr(model, "estimators_features_", None)
    feature_sets = estimator_features if estimator_features is not None else [all_features] * estimator_count

    for estimator, features in zip(model.estimators_, feature_sets, strict=True):
        feature_index = np.asarray(features, dtype=np.intp)
        if feature_index.shape == all_features.shape and np.array_equal(feature_index, all_features):
            estimator_X = X
        else:
            estimator_X = X[:, feature_index]

        estimator_proba = estimator.predict_proba(estimator_X)
        estimator_classes = np.asarray(getattr(estimator, "classes_", _BINARY_CLASS_LABELS))
        positive_column = np.flatnonzero(estimator_classes == 1)
        if positive_column.size == 1:
            positive_sum += estimator_proba[:, int(positive_column[0])]
            continue

        if estimator_proba.shape[1] == 1 and estimator_classes.size == 1:
            only_class = int(estimator_classes[0])
            if only_class == 1:
                positive_sum += 1.0
            elif only_class != 0:
                raise RuntimeError(f"Unexpected Bagged LASSO class label: {only_class}")
            continue

        raise RuntimeError(
            "Bagged LASSO estimator probabilities could not be aligned to the positive class."
        )

    return np.clip(positive_sum / float(estimator_count), 1e-6, 1.0 - 1e-6)


def _run_regularized_logistic_search(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    c_values: tuple[float, ...],
    l1_ratios: tuple[float, ...],
    max_iter: int = DEFAULT_LASSO_MAX_ITER,
    tol: float = DEFAULT_LASSO_TOL,
    progress_desc: str | None = None,
    model_label: str = "regularized logistic",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    _require_non_empty_split("train", train_df)
    _require_non_empty_split("validation", validation_df)
    feature_names = infer_feature_names(train_df)
    X_train = _lasso_feature_matrix(train_df, feature_names)
    y_train = labels(train_df)
    X_valid = _lasso_feature_matrix(validation_df, feature_names)
    y_valid = labels(validation_df)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    history: list[dict[str, object]] = []
    best_params: dict[str, object] | None = None
    best_loss: float | None = None
    candidates = [(float(c_value), float(l1_ratio)) for c_value in c_values for l1_ratio in l1_ratios]
    progress = tqdm(candidates, desc=progress_desc, unit="config") if progress_desc else None
    iterator = progress if progress is not None else candidates
    try:
        for c_value, l1_ratio in iterator:
            model = _fit_regularized_logistic_classifier(
                X_train_scaled,
                y_train,
                c_value=c_value,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=max_iter,
                tol=tol,
            )
            valid_pred = np.clip(model.predict_proba(X_valid_scaled)[:, 1], 1e-6, 1.0 - 1e-6)
            valid_loss = _binary_log_loss(y_valid, valid_pred)
            candidate = {
                "C": c_value,
                "penalty": "elasticnet",
                "l1_ratio": l1_ratio,
                "solver": "saga",
                "fit_intercept": True,
                "max_iter": int(max_iter),
                "tol": float(tol),
            }
            history.append(
                {
                    "params": candidate,
                    "validation_log_loss": valid_loss,
                    "nonzero_coefficients": int(np.count_nonzero(model.coef_[0])),
                    "iterations_used": int(np.max(model.n_iter_)),
                }
            )
            if best_loss is None or valid_loss < best_loss:
                best_loss = valid_loss
                best_params = candidate
                if progress is not None:
                    progress.set_postfix(
                        best_logloss=f"{valid_loss:.4f}",
                        l1_ratio=f"{l1_ratio:.2f}",
                        nonzero=int(np.count_nonzero(model.coef_[0])),
                    )
    finally:
        if progress is not None:
            progress.close()
    if best_params is None:
        raise RuntimeError(f"No {model_label} hyperparameter candidates were evaluated.")
    return best_params, history


def run_lasso_search(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    c_values: tuple[float, ...] = DEFAULT_LASSO_C_VALUES,
    max_iter: int = DEFAULT_LASSO_MAX_ITER,
    tol: float = DEFAULT_LASSO_TOL,
    progress_desc: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    return _run_regularized_logistic_search(
        train_df,
        validation_df,
        c_values=c_values,
        l1_ratios=(1.0,),
        max_iter=max_iter,
        tol=tol,
        progress_desc=progress_desc,
        model_label="LASSO",
    )


def run_bagged_lasso_search(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    c_values: tuple[float, ...] = DEFAULT_BAGGED_LASSO_C_VALUES,
    n_estimators: int = DEFAULT_BAGGED_LASSO_N_ESTIMATORS,
    max_samples: float = DEFAULT_BAGGED_LASSO_MAX_SAMPLES,
    max_iter: int = DEFAULT_BAGGED_LASSO_MAX_ITER,
    tol: float = DEFAULT_BAGGED_LASSO_TOL,
    n_jobs: int = DEFAULT_BAGGED_LASSO_N_JOBS,
    progress_desc: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    _require_non_empty_split("train", train_df)
    _require_non_empty_split("validation", validation_df)
    feature_names = infer_feature_names(train_df)
    X_train = _lasso_feature_matrix(train_df, feature_names)
    y_train = labels(train_df)
    X_valid = _lasso_feature_matrix(validation_df, feature_names)
    y_valid = labels(validation_df)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    history: list[dict[str, object]] = []
    best_params: dict[str, object] | None = None
    best_loss: float | None = None
    progress = tqdm(c_values, desc=progress_desc, unit="config") if progress_desc else None
    iterator = progress if progress is not None else c_values
    try:
        for c_value in iterator:
            model = _fit_bagged_lasso_classifier(
                X_train_scaled,
                y_train,
                c_value=float(c_value),
                fit_intercept=True,
                n_estimators=int(n_estimators),
                max_samples=float(max_samples),
                max_iter=int(max_iter),
                tol=float(tol),
                n_jobs=int(n_jobs),
            )
            valid_pred = _bagged_lasso_positive_class_probabilities(model, X_valid_scaled)
            valid_loss = _binary_log_loss(y_valid, valid_pred)
            stats = _bagged_lasso_statistics(model)
            candidate = {
                "C": float(c_value),
                "penalty": "elasticnet",
                "l1_ratio": 1.0,
                "solver": "saga",
                "fit_intercept": True,
                "n_estimators": int(n_estimators),
                "max_samples": float(max_samples),
                "bootstrap": True,
                "max_iter": int(max_iter),
                "tol": float(tol),
                "n_jobs": int(n_jobs),
            }
            history.append(
                {
                    "params": candidate,
                    "validation_log_loss": valid_loss,
                    "mean_nonzero_coefficients": stats["mean_nonzero_coefficients"],
                    "selected_feature_count_any_estimator": stats["selected_feature_count_any_estimator"],
                    "selected_feature_count_majority_vote": stats["selected_feature_count_majority_vote"],
                    "mean_iterations_used": stats["mean_iterations_used"],
                }
            )
            if best_loss is None or valid_loss < best_loss:
                best_loss = valid_loss
                best_params = candidate
                if progress is not None:
                    progress.set_postfix(
                        best_logloss=f"{valid_loss:.4f}",
                        mean_nonzero=f"{float(stats['mean_nonzero_coefficients']):.1f}",
                        majority=int(stats["selected_feature_count_majority_vote"]),
                    )
    finally:
        if progress is not None:
            progress.close()
    if best_params is None:
        raise RuntimeError("No Bagged LASSO hyperparameter candidates were evaluated.")
    return best_params, history


def run_elastic_net_search(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    c_values: tuple[float, ...] = DEFAULT_ELASTIC_NET_C_VALUES,
    l1_ratios: tuple[float, ...] = DEFAULT_ELASTIC_NET_L1_RATIOS,
    max_iter: int = DEFAULT_ELASTIC_NET_MAX_ITER,
    tol: float = DEFAULT_ELASTIC_NET_TOL,
    progress_desc: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    return _run_regularized_logistic_search(
        train_df,
        validation_df,
        c_values=c_values,
        l1_ratios=l1_ratios,
        max_iter=max_iter,
        tol=tol,
        progress_desc=progress_desc,
        model_label="Elastic Net",
    )


def run_linear_svm_search(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    c_values: tuple[float, ...] = DEFAULT_LINEAR_SVM_C_VALUES,
    max_iter: int = DEFAULT_LINEAR_SVM_MAX_ITER,
    tol: float = DEFAULT_LINEAR_SVM_TOL,
    progress_desc: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    _require_non_empty_split("train", train_df)
    _require_non_empty_split("validation", validation_df)
    feature_names = infer_feature_names(train_df)
    X_train = _lasso_feature_matrix(train_df, feature_names)
    y_train = labels(train_df)
    X_valid = _lasso_feature_matrix(validation_df, feature_names)
    y_valid = labels(validation_df)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    history: list[dict[str, object]] = []
    best_params: dict[str, object] | None = None
    best_loss: float | None = None
    progress = tqdm(c_values, desc=progress_desc, unit="config") if progress_desc else None
    iterator = progress if progress is not None else c_values
    try:
        for c_value in iterator:
            model = _fit_linear_svm_classifier(
                X_train_scaled,
                y_train,
                c_value=float(c_value),
                fit_intercept=True,
                max_iter=max_iter,
                tol=tol,
            )
            valid_scores = model.decision_function(X_valid_scaled)
            valid_pred = _sigmoid_from_scores(valid_scores)
            valid_loss = _binary_log_loss(y_valid, valid_pred)
            candidate = {
                "C": float(c_value),
                "penalty": "l1",
                "loss": "squared_hinge",
                "dual": False,
                "solver": "liblinear_style_linear_svc",
                "fit_intercept": True,
                "max_iter": int(max_iter),
                "tol": float(tol),
            }
            history.append(
                {
                    "params": candidate,
                    "validation_log_loss": valid_loss,
                    "nonzero_coefficients": int(np.count_nonzero(model.coef_[0])),
                    "iterations_used": int(np.max(model.n_iter_)),
                }
            )
            if best_loss is None or valid_loss < best_loss:
                best_loss = valid_loss
                best_params = candidate
                if progress is not None:
                    progress.set_postfix(best_logloss=f"{valid_loss:.4f}", nonzero=int(np.count_nonzero(model.coef_[0])))
    finally:
        if progress is not None:
            progress.close()
    if best_params is None:
        raise RuntimeError("No Linear SVM hyperparameter candidates were evaluated.")
    return best_params, history


def _train_regularized_logistic_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    params: dict[str, object],
) -> tuple[RegularizedLogisticArtifact, dict[str, object]]:
    _require_non_empty_split("train", train_df)
    _require_non_empty_split("validation", validation_df)
    feature_names = infer_feature_names(train_df)
    X_train = _lasso_feature_matrix(train_df, feature_names)
    y_train = labels(train_df)
    X_valid = _lasso_feature_matrix(validation_df, feature_names)
    y_valid = labels(validation_df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    model = _fit_regularized_logistic_classifier(
        X_train_scaled,
        y_train,
        c_value=float(params["C"]),
        l1_ratio=float(params.get("l1_ratio", 1.0)),
        fit_intercept=bool(params.get("fit_intercept", True)),
        max_iter=int(params.get("max_iter", DEFAULT_LASSO_MAX_ITER)),
        tol=float(params.get("tol", DEFAULT_LASSO_TOL)),
    )
    valid_pred = np.clip(model.predict_proba(X_valid_scaled)[:, 1], 1e-6, 1.0 - 1e-6)
    metrics = {
        "feature_names": list(feature_names),
        "feature_schema": _feature_schema_for_frame(train_df),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "best_iteration": None,
        "validation_log_loss": _binary_log_loss(y_valid, valid_pred),
        "params": params,
        "iterations_used": int(np.max(model.n_iter_)),
        "nonzero_coefficients": int(np.count_nonzero(model.coef_[0])),
        "intercept": float(model.intercept_[0]),
    }
    return RegularizedLogisticArtifact(scaler=scaler, model=model), metrics


def train_lasso_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    params: dict[str, object],
) -> tuple[LassoArtifact, dict[str, object]]:
    normalized_params = dict(params)
    normalized_params["l1_ratio"] = 1.0
    artifact, metrics = _train_regularized_logistic_model(train_df, validation_df, normalized_params)
    return artifact, metrics


def train_bagged_lasso_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    params: dict[str, object],
) -> tuple[BaggedLassoArtifact, dict[str, object]]:
    _require_non_empty_split("train", train_df)
    _require_non_empty_split("validation", validation_df)
    feature_names = infer_feature_names(train_df)
    X_train = _lasso_feature_matrix(train_df, feature_names)
    y_train = labels(train_df)
    X_valid = _lasso_feature_matrix(validation_df, feature_names)
    y_valid = labels(validation_df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    model = _fit_bagged_lasso_classifier(
        X_train_scaled,
        y_train,
        c_value=float(params["C"]),
        fit_intercept=bool(params.get("fit_intercept", True)),
        n_estimators=int(params.get("n_estimators", DEFAULT_BAGGED_LASSO_N_ESTIMATORS)),
        max_samples=float(params.get("max_samples", DEFAULT_BAGGED_LASSO_MAX_SAMPLES)),
        max_iter=int(params.get("max_iter", DEFAULT_BAGGED_LASSO_MAX_ITER)),
        tol=float(params.get("tol", DEFAULT_BAGGED_LASSO_TOL)),
        n_jobs=int(params.get("n_jobs", DEFAULT_BAGGED_LASSO_N_JOBS)),
    )
    valid_pred = _bagged_lasso_positive_class_probabilities(model, X_valid_scaled)
    stats = _bagged_lasso_statistics(model)
    metrics = {
        "feature_names": list(feature_names),
        "feature_schema": _feature_schema_for_frame(train_df),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "best_iteration": None,
        "validation_log_loss": _binary_log_loss(y_valid, valid_pred),
        "params": params,
        "estimators_trained": int(len(model.estimators_)),
        "mean_nonzero_coefficients": stats["mean_nonzero_coefficients"],
        "selected_feature_count_any_estimator": stats["selected_feature_count_any_estimator"],
        "selected_feature_count_majority_vote": stats["selected_feature_count_majority_vote"],
        "mean_intercept": stats["mean_intercept"],
        "intercept_std": stats["intercept_std"],
        "mean_iterations_used": stats["mean_iterations_used"],
        "max_iterations_used": stats["max_iterations_used"],
    }
    return BaggedLassoArtifact(scaler=scaler, model=model), metrics


def train_elastic_net_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    params: dict[str, object],
) -> tuple[ElasticNetArtifact, dict[str, object]]:
    artifact, metrics = _train_regularized_logistic_model(train_df, validation_df, params)
    return artifact, metrics


def train_linear_svm_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    params: dict[str, object],
) -> tuple[LinearSVMArtifact, dict[str, object]]:
    _require_non_empty_split("train", train_df)
    _require_non_empty_split("validation", validation_df)
    feature_names = infer_feature_names(train_df)
    X_train = _lasso_feature_matrix(train_df, feature_names)
    y_train = labels(train_df)
    X_valid = _lasso_feature_matrix(validation_df, feature_names)
    y_valid = labels(validation_df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    model = _fit_linear_svm_classifier(
        X_train_scaled,
        y_train,
        c_value=float(params["C"]),
        fit_intercept=bool(params.get("fit_intercept", True)),
        max_iter=int(params.get("max_iter", DEFAULT_LINEAR_SVM_MAX_ITER)),
        tol=float(params.get("tol", DEFAULT_LINEAR_SVM_TOL)),
    )
    valid_scores = model.decision_function(X_valid_scaled)
    valid_pred = _sigmoid_from_scores(valid_scores)
    metrics = {
        "feature_names": list(feature_names),
        "feature_schema": _feature_schema_for_frame(train_df),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "best_iteration": None,
        "validation_log_loss": _binary_log_loss(y_valid, valid_pred),
        "params": params,
        "iterations_used": int(np.max(model.n_iter_)),
        "nonzero_coefficients": int(np.count_nonzero(model.coef_[0])),
        "intercept": float(model.intercept_[0]),
    }
    return LinearSVMArtifact(scaler=scaler, model=model), metrics


def save_lightgbm_artifacts(
    artifacts_dir: Path,
    model: LGBMClassifier,
    metrics: dict[str, object],
) -> None:
    lightgbm_dir = artifacts_dir / "lightgbm"
    lightgbm_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(lightgbm_dir / "model.txt"))
    write_json(lightgbm_dir / "metrics.json", metrics)
    feature_names = tuple(str(name) for name in metrics.get("feature_names", _BASE_FEATURE_ORDER))
    feature_importance = {
        name: float(value) for name, value in zip(feature_names, model.feature_importances_)
    }
    write_json(lightgbm_dir / "feature_importance.json", feature_importance)


def _save_regularized_logistic_artifacts(
    artifacts_dir: Path,
    artifact: RegularizedLogisticArtifact,
    metrics: dict[str, object],
    *,
    model_subdir: str,
) -> None:
    output_dir = artifacts_dir / model_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = tuple(str(name) for name in metrics.get("feature_names", _BASE_FEATURE_ORDER))
    joblib_dump(
        {
            "model": artifact.model,
            "scaler": artifact.scaler,
            "feature_names": feature_names,
        },
        output_dir / "model.joblib",
    )
    write_json(output_dir / "metrics.json", metrics)
    coefficients = {
        name: float(value) for name, value in zip(feature_names, artifact.model.coef_[0])
    }
    write_json(
        output_dir / "coefficients.json",
        {
            "intercept": float(artifact.model.intercept_[0]),
            "coefficients": coefficients,
            "nonzero_coefficients": [name for name, value in coefficients.items() if abs(value) > 0.0],
        },
    )


def save_lasso_artifacts(
    artifacts_dir: Path,
    artifact: LassoArtifact,
    metrics: dict[str, object],
) -> None:
    _save_regularized_logistic_artifacts(artifacts_dir, artifact, metrics, model_subdir="lasso")


def save_bagged_lasso_artifacts(
    artifacts_dir: Path,
    artifact: BaggedLassoArtifact,
    metrics: dict[str, object],
) -> None:
    output_dir = artifacts_dir / "bagged_lasso"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = tuple(str(name) for name in metrics.get("feature_names", _BASE_FEATURE_ORDER))
    joblib_dump(
        {
            "model": artifact.model,
            "scaler": artifact.scaler,
            "feature_names": feature_names,
        },
        output_dir / "model.joblib",
    )
    write_json(output_dir / "metrics.json", metrics)
    stats = _bagged_lasso_statistics(artifact.model)
    mean_coefficients = {
        name: float(value) for name, value in zip(feature_names, np.asarray(stats["mean_coefficients"], dtype=np.float64))
    }
    coefficient_std = {
        name: float(value) for name, value in zip(feature_names, np.asarray(stats["coefficient_std"], dtype=np.float64))
    }
    selection_frequency = {
        name: float(value) for name, value in zip(feature_names, np.asarray(stats["selection_frequency"], dtype=np.float64))
    }
    write_json(
        output_dir / "coefficients.json",
        {
            "mean_intercept": stats["mean_intercept"],
            "intercept_std": stats["intercept_std"],
            "mean_coefficients": mean_coefficients,
            "coefficient_std": coefficient_std,
            "selection_frequency": selection_frequency,
            "selected_by_any_estimator": [name for name, value in selection_frequency.items() if value > 0.0],
            "selected_by_majority_vote": [name for name, value in selection_frequency.items() if value >= 0.5],
        },
    )


def save_elastic_net_artifacts(
    artifacts_dir: Path,
    artifact: ElasticNetArtifact,
    metrics: dict[str, object],
) -> None:
    _save_regularized_logistic_artifacts(artifacts_dir, artifact, metrics, model_subdir="elastic_net")


def save_linear_svm_artifacts(
    artifacts_dir: Path,
    artifact: LinearSVMArtifact,
    metrics: dict[str, object],
) -> None:
    svm_dir = artifacts_dir / "linear_svm"
    svm_dir.mkdir(parents=True, exist_ok=True)
    feature_names = tuple(str(name) for name in metrics.get("feature_names", _BASE_FEATURE_ORDER))
    joblib_dump(
        {
            "model": artifact.model,
            "scaler": artifact.scaler,
            "feature_names": feature_names,
        },
        svm_dir / "model.joblib",
    )
    write_json(svm_dir / "metrics.json", metrics)
    coefficients = {
        name: float(value) for name, value in zip(feature_names, artifact.model.coef_[0])
    }
    write_json(
        svm_dir / "coefficients.json",
        {
            "intercept": float(artifact.model.intercept_[0]),
            "coefficients": coefficients,
            "nonzero_coefficients": [name for name, value in coefficients.items() if abs(value) > 0.0],
        },
    )


def raw_predictions(model: LGBMClassifier, df: pd.DataFrame) -> np.ndarray:
    feature_names = tuple(str(name) for name in getattr(model, "feature_name_", []) if name)
    selected_feature_names = feature_names or infer_feature_names(df)
    return np.clip(model.predict_proba(feature_matrix(df, selected_feature_names))[:, 1], 1e-6, 1.0 - 1e-6)


def _load_regularized_logistic_artifact(model_path: Path) -> RegularizedLogisticArtifact:
    payload = joblib_load(model_path)
    return RegularizedLogisticArtifact(scaler=payload["scaler"], model=payload["model"])


def load_lasso_artifact(model_path: Path) -> LassoArtifact:
    return _load_regularized_logistic_artifact(model_path)


def load_bagged_lasso_artifact(model_path: Path) -> BaggedLassoArtifact:
    payload = joblib_load(model_path)
    return BaggedLassoArtifact(scaler=payload["scaler"], model=payload["model"])


def load_elastic_net_artifact(model_path: Path) -> ElasticNetArtifact:
    return _load_regularized_logistic_artifact(model_path)


def load_linear_svm_artifact(model_path: Path) -> LinearSVMArtifact:
    payload = joblib_load(model_path)
    return LinearSVMArtifact(scaler=payload["scaler"], model=payload["model"])


def _regularized_logistic_raw_predictions(artifact: RegularizedLogisticArtifact, df: pd.DataFrame) -> np.ndarray:
    payload_feature_names = getattr(artifact.model, "feature_names_in_", None)
    feature_names = tuple(str(name) for name in payload_feature_names) if payload_feature_names is not None else infer_feature_names(df)
    X = _lasso_feature_matrix(df, feature_names)
    X_scaled = artifact.scaler.transform(X)
    return np.clip(artifact.model.predict_proba(X_scaled)[:, 1], 1e-6, 1.0 - 1e-6)


def lasso_raw_predictions(artifact: LassoArtifact, df: pd.DataFrame) -> np.ndarray:
    return _regularized_logistic_raw_predictions(artifact, df)


def bagged_lasso_raw_predictions(artifact: BaggedLassoArtifact, df: pd.DataFrame) -> np.ndarray:
    payload_feature_names = getattr(artifact.model, "feature_names_in_", None)
    feature_names = tuple(str(name) for name in payload_feature_names) if payload_feature_names is not None else infer_feature_names(df)
    X = _lasso_feature_matrix(df, feature_names)
    X_scaled = artifact.scaler.transform(X)
    return _bagged_lasso_positive_class_probabilities(artifact.model, X_scaled)


def elastic_net_raw_predictions(artifact: ElasticNetArtifact, df: pd.DataFrame) -> np.ndarray:
    return _regularized_logistic_raw_predictions(artifact, df)


def linear_svm_raw_predictions(artifact: LinearSVMArtifact, df: pd.DataFrame) -> np.ndarray:
    payload_feature_names = getattr(artifact.model, "feature_names_in_", None)
    feature_names = tuple(str(name) for name in payload_feature_names) if payload_feature_names is not None else infer_feature_names(df)
    X = _lasso_feature_matrix(df, feature_names)
    X_scaled = artifact.scaler.transform(X)
    scores = artifact.model.decision_function(X_scaled)
    return _sigmoid_from_scores(scores)


def booster_raw_predictions(
    booster: Booster,
    df: pd.DataFrame,
    *,
    best_iteration: int | None = None,
) -> np.ndarray:
    kwargs: dict[str, object] = {}
    if best_iteration is not None and best_iteration > 0:
        kwargs["num_iteration"] = best_iteration
    feature_names = tuple(str(name) for name in booster.feature_name() if name)
    selected_feature_names = feature_names or infer_feature_names(df)
    return np.clip(booster.predict(feature_matrix(df, selected_feature_names), **kwargs), 1e-6, 1.0 - 1e-6)


def calibrate_validation_predictions(
    artifacts_dir: Path,
    validation_df: pd.DataFrame,
    validation_raw_pred: np.ndarray,
    *,
    model_family: str = "lightgbm",
) -> np.ndarray:
    _require_non_empty_split("validation", validation_df)
    calibration = fit_platt_scaler(validation_raw_pred, labels(validation_df))
    save_platt_calibration(artifacts_dir / model_family / "calibration.json", calibration)
    return calibration.apply(validation_raw_pred)  # type: ignore[return-value]


def apply_calibration_to_predictions(
    artifacts_dir: Path,
    probabilities: np.ndarray,
    *,
    model_family: str = "lightgbm",
) -> np.ndarray:
    from src.live.kalshi.calibration import load_platt_calibration

    calibration = load_platt_calibration(artifacts_dir / model_family / "calibration.json")
    return calibration.apply(probabilities)  # type: ignore[return-value]


def _reference_price_cents(side: str, market_prob: float) -> int:
    yes_price_cents = int(round(market_prob * 100.0))
    yes_price_cents = min(99, max(1, yes_price_cents))
    if side == "YES":
        return yes_price_cents
    return 100 - yes_price_cents


def _regime_payload_for_row(row: pd.Series | Any) -> dict[str, object]:
    regime = evaluate_kxbtc15m_regime(
        price_momentum=row["price_momentum"] if "price_momentum" in row else None,
        signed_contracts_sum_300s=row["signed_contracts_sum_300s"] if "signed_contracts_sum_300s" in row else None,
        yes_taker_share_300s=row["yes_taker_share_300s"] if "yes_taker_share_300s" in row else None,
    )
    return {
        "regime_label": regime.regime_label,
        "bearish_vote_count": regime.bearish_vote_count,
        "bullish_vote_count": regime.bullish_vote_count,
        "regime_price_momentum_bearish": regime.price_momentum_bearish,
        "regime_signed_flow_bearish": regime.signed_flow_bearish,
        "regime_yes_share_bearish": regime.yes_share_bearish,
        "regime_price_momentum_bullish": regime.price_momentum_bullish,
        "regime_signed_flow_bullish": regime.signed_flow_bullish,
        "regime_yes_share_bullish": regime.yes_share_bullish,
    }


def _is_standard_policy_fast_path_compatible(config: PolicyConfig) -> bool:
    return (
        config.contracts_per_order == 1
        and not config.apply_regime_hard_gate
        and config.capital_pct_per_order is None
        and config.kelly_fraction_multiplier is None
        and config.kelly_fraction_cap_pct is None
        and not config.allow_stacking
        and config.max_entries_per_ticker is None
        and not config.require_price_improvement_for_stack
    )


def _prepare_standard_policy_inputs(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    slippage: float,
) -> _PreparedStandardPolicyInputs:
    frame = df.loc[
        :,
        [
            "ticker",
            "trade_id",
            "created_time",
            "close_time",
            "market_prob",
            "tau_minutes",
            "actual_outcome",
        ],
    ].copy()
    frame["predicted_yes_probability"] = probabilities
    frame = frame.sort_values(["created_time", "ticker", "trade_id"]).reset_index(drop=True)

    market_prob = frame["market_prob"].to_numpy(dtype=np.float64, copy=True)
    predicted_yes_probability = frame["predicted_yes_probability"].to_numpy(dtype=np.float64, copy=True)
    side_is_yes = predicted_yes_probability > market_prob
    yes_price_cents = np.rint(market_prob * 100.0).astype(np.int16, copy=False)
    yes_price_cents = np.clip(yes_price_cents, 1, 99)
    reference_price_cents = np.where(side_is_yes, yes_price_cents, 100 - yes_price_cents).astype(np.int16, copy=False)

    displayed_entry_price = reference_price_cents.astype(np.float64, copy=False) / 100.0
    slipped_entry_price = np.minimum(0.999999, displayed_entry_price * (1.0 + slippage))
    fees_dollars = np.ceil(0.07 * slipped_entry_price * (1.0 - slipped_entry_price) * 100.0) / 100.0
    side_probability = np.where(side_is_yes, predicted_yes_probability, 1.0 - predicted_yes_probability)
    post_cost_edge = side_probability - (slipped_entry_price + fees_dollars)
    actual_outcomes = frame["actual_outcome"].to_numpy(dtype=np.int8, copy=True)
    payout_dollars = np.where(side_is_yes, actual_outcomes, 1 - actual_outcomes).astype(np.float64, copy=False)

    created_time_ns = pd.to_datetime(frame["created_time"], utc=True).astype("int64", copy=False).to_numpy(copy=True)
    close_time_ns = pd.to_datetime(frame["close_time"], utc=True).astype("int64", copy=False).to_numpy(copy=True)
    hold_minutes = np.maximum(0.0, (close_time_ns - created_time_ns) / 60_000_000_000.0)
    ticker_codes = pd.factorize(frame["ticker"], sort=False)[0].astype(np.int32, copy=False)

    return _PreparedStandardPolicyInputs(
        created_time_ns=created_time_ns,
        close_time_ns=close_time_ns,
        ticker_codes=ticker_codes,
        tau_minutes=frame["tau_minutes"].to_numpy(dtype=np.float64, copy=True),
        actual_outcomes=actual_outcomes,
        reference_price_cents=reference_price_cents.astype(np.int16, copy=False),
        post_cost_edge=post_cost_edge.astype(np.float64, copy=False),
        cash_required_dollars=(slipped_entry_price + fees_dollars).astype(np.float64, copy=False),
        entry_cost_dollars=slipped_entry_price.astype(np.float64, copy=False),
        fees_dollars=fees_dollars.astype(np.float64, copy=False),
        payout_dollars=payout_dollars,
        hold_minutes=hold_minutes.astype(np.float64, copy=False),
        max_close_time_ns=int(close_time_ns.max()),
    )


def _simulate_standard_policy_prepared(
    prepared: _PreparedStandardPolicyInputs,
    config: PolicyConfig,
    *,
    overall_log_loss: float,
) -> PolicyResult:
    signal_config = config.signal_config
    available_cash = signal_config.starting_cash_dollars
    open_cost_basis = 0.0
    peak_equity = signal_config.starting_cash_dollars
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    total_pnl = 0.0
    trades = 0
    skipped_due_open_ticker = 0
    skipped_due_price_band = 0
    skipped_due_regime = 0
    skipped_due_post_cost_edge = 0
    sequence = 0
    reserve_cash_ratio = signal_config.reserve_cash_pct / 100.0
    open_positions: list[tuple[int, int, _StandardOpenPosition]] = []
    active_tickers: set[int] = set()

    def update_drawdown() -> None:
        nonlocal peak_equity, max_drawdown, max_drawdown_pct
        equity = available_cash + open_cost_basis
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        drawdown_pct = 0.0 if peak_equity <= 0 else drawdown / peak_equity
        max_drawdown = max(max_drawdown, drawdown)
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct * 100.0)

    def settle_positions(cutoff_time_ns: int) -> None:
        nonlocal available_cash, open_cost_basis, total_pnl
        while open_positions and open_positions[0][0] <= cutoff_time_ns:
            _close_time_ns, _seq, position = heapq.heappop(open_positions)
            open_cost_basis -= position.entry_cost
            available_cash += position.payout
            total_pnl += (position.payout - position.entry_cost) - position.fees
            active_tickers.discard(position.ticker_code)
            update_drawdown()

    for row_index in range(len(prepared.created_time_ns)):
        settle_positions(int(prepared.created_time_ns[row_index]))
        tau_minutes = float(prepared.tau_minutes[row_index])
        if tau_minutes < signal_config.min_tau_minutes or tau_minutes > signal_config.max_tau_minutes:
            continue

        reference_price_cents = int(prepared.reference_price_cents[row_index])
        if reference_price_cents < signal_config.price_band_min_cents or reference_price_cents > signal_config.price_band_max_cents:
            skipped_due_price_band += 1
            continue

        ticker_code = int(prepared.ticker_codes[row_index])
        if ticker_code in active_tickers:
            skipped_due_open_ticker += 1
            continue

        post_cost_edge = float(prepared.post_cost_edge[row_index])
        if post_cost_edge + 1e-12 < signal_config.edge_threshold:
            skipped_due_post_cost_edge += 1
            continue

        cash_required = float(prepared.cash_required_dollars[row_index])
        reserve_cash = reserve_cash_ratio * (available_cash + open_cost_basis)
        if available_cash - cash_required < reserve_cash - 1e-12:
            continue

        entry_cost = float(prepared.entry_cost_dollars[row_index])
        payout_dollars = float(prepared.payout_dollars[row_index])
        fees_dollars = float(prepared.fees_dollars[row_index])
        hold_minutes = float(prepared.hold_minutes[row_index])

        available_cash -= cash_required
        open_cost_basis += entry_cost
        position = _StandardOpenPosition(
            ticker_code=ticker_code,
            entry_cost=entry_cost,
            payout=payout_dollars,
            fees=fees_dollars,
            hold_minutes=hold_minutes,
        )
        heapq.heappush(open_positions, (int(prepared.close_time_ns[row_index]), sequence, position))
        active_tickers.add(ticker_code)
        sequence += 1
        trades += 1
        update_drawdown()

    settle_positions(prepared.max_close_time_ns)
    ending_equity = available_cash + open_cost_basis
    return_pct = 0.0 if signal_config.starting_cash_dollars <= 0 else (
        (ending_equity / signal_config.starting_cash_dollars) - 1.0
    ) * 100.0
    objective = total_pnl / max(1.0, max_drawdown)
    return PolicyResult(
        config=config,
        objective=objective,
        trades=trades,
        net_pnl_dollars=total_pnl,
        max_drawdown_dollars=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        return_pct=return_pct,
        log_loss=overall_log_loss,
        skipped_due_open_ticker=skipped_due_open_ticker,
        skipped_due_price_band=skipped_due_price_band,
        skipped_due_regime=0,
        skipped_due_post_cost_edge=skipped_due_post_cost_edge,
    )


def _trade_record_columns() -> list[str]:
    return [
        "ticker",
        "trade_id",
        "created_time",
        "close_time",
        "side",
        "sizing_method",
        "contracts",
        "actual_outcome",
        "is_win",
        "market_prob",
        "predicted_yes_probability",
        "tau_minutes",
        "realized_vol_regime_source",
        "realized_vol_regime_bucket",
        "realized_vol_regime_bucket_version",
        "realized_vol_regime_metric",
        "hour_of_day_et",
        "hour_of_day_et_label",
        "session_block_et",
        "regime_label",
        "bearish_vote_count",
        "bullish_vote_count",
        "regime_price_momentum_bearish",
        "regime_signed_flow_bearish",
        "regime_yes_share_bearish",
        "regime_price_momentum_bullish",
        "regime_signed_flow_bullish",
        "regime_yes_share_bullish",
        "target_fraction_of_equity",
        "cash_budget_dollars",
        "kelly_raw_fraction_of_equity",
        "kelly_fraction_of_equity",
        "reference_price_cents",
        "max_acceptable_entry_price_cents",
        "post_cost_edge",
        "entry_cost_dollars",
        "fees_dollars",
        "cash_required_dollars",
        "payout_dollars",
        "gross_pnl_dollars",
        "net_pnl_dollars",
        "hold_minutes",
    ]


def prediction_export_columns() -> list[str]:
    return [
        "ticker",
        "trade_id",
        "created_time",
        "close_time",
        "market_prob",
        "tau_minutes",
        "actual_outcome",
        "raw_probability",
        "calibrated_probability",
        "realized_vol_regime_source",
        "realized_vol_regime_bucket",
        "realized_vol_regime_bucket_version",
        "realized_vol_regime_metric",
        "hour_of_day_et",
        "hour_of_day_et_label",
        "session_block_et",
    ]


def build_prediction_export_frame(
    split_df: pd.DataFrame,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
) -> pd.DataFrame:
    annotated, _metadata = _annotate_diagnostic_context(split_df)
    columns = [
        "ticker",
        "trade_id",
        "created_time",
        "close_time",
        "market_prob",
        "tau_minutes",
        "actual_outcome",
        "realized_vol_regime_source",
        "realized_vol_regime_bucket",
        "realized_vol_regime_bucket_version",
        "realized_vol_regime_metric",
        "hour_of_day_et",
        "hour_of_day_et_label",
        "session_block_et",
    ]
    predictions = annotated[columns].copy()
    predictions["raw_probability"] = np.asarray(raw_probabilities, dtype=np.float64)
    predictions["calibrated_probability"] = np.asarray(calibrated_probabilities, dtype=np.float64)
    return predictions.loc[:, prediction_export_columns()]


def _empty_trade_records_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_trade_record_columns())


def _simulate_policy_with_trade_records(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    config: PolicyConfig,
    *,
    overall_log_loss: float,
    progress_desc: str | None = None,
) -> tuple[PolicyResult, pd.DataFrame]:
    signal_config = config.signal_config
    frame = df.copy()
    frame["predicted_yes_probability"] = probabilities
    frame = frame.sort_values(["created_time", "ticker", "trade_id"]).reset_index(drop=True)

    available_cash = signal_config.starting_cash_dollars
    open_cost_basis = 0.0
    peak_equity = signal_config.starting_cash_dollars
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    total_pnl = 0.0
    total_fees = 0.0
    total_turnover = 0.0
    total_hold_minutes = 0.0
    wins = 0
    trades = 0
    skipped_due_open_ticker = 0
    skipped_due_price_band = 0
    skipped_due_regime = 0
    skipped_due_post_cost_edge = 0
    sequence = 0
    open_positions: list[tuple[pd.Timestamp, int, _OpenPosition]] = []
    open_positions_by_ticker: dict[str, list[_OpenPosition]] = {}
    trade_records: list[dict[str, object]] = []

    def resolve_contracts(
        *,
        side: str,
        predicted_yes_probability: float,
        reference_price_cents: int,
    ) -> tuple[int, dict[str, object]]:
        if config.kelly_fraction_multiplier is not None:
            equity = available_cash + open_cost_basis
            reserve_cash = (signal_config.reserve_cash_pct / 100.0) * equity
            deployable_cash = max(0.0, available_cash - reserve_cash)
            kelly = calculate_kelly_sizing_metrics(
                side=side,
                predicted_yes_probability=predicted_yes_probability,
                displayed_entry_price_cents=reference_price_cents,
                slippage=signal_config.slippage,
                fraction_multiplier=config.kelly_fraction_multiplier,
                fraction_cap=None if config.kelly_fraction_cap_pct is None else (config.kelly_fraction_cap_pct / 100.0),
            )
            target_fraction = kelly.capped_fraction_of_equity
            cash_budget = min(target_fraction * equity, deployable_cash)
            metadata = {
                "sizing_method": "kelly",
                "target_fraction_of_equity": float(target_fraction),
                "cash_budget_dollars": float(cash_budget),
                "kelly_raw_fraction_of_equity": float(kelly.raw_fraction_of_equity),
                "kelly_fraction_of_equity": float(kelly.capped_fraction_of_equity),
            }
            if cash_budget <= 0 or kelly.per_contract_cash_required_dollars > cash_budget + 1e-12:
                return 0, metadata

            upper_bound = max(1, int(cash_budget / max(reference_price_cents / 100.0, 1e-9)) + 1)
            low = 1
            high = upper_bound
            best = 1
            while low <= high:
                mid = (low + high) // 2
                _edge_mid, _entry_mid, _fees_mid, cash_required_mid = calculate_cost_metrics(
                    side=side,
                    predicted_yes_probability=predicted_yes_probability,
                    displayed_entry_price_cents=reference_price_cents,
                    contracts=mid,
                    slippage=signal_config.slippage,
                )
                if cash_required_mid <= cash_budget + 1e-12:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            return best, metadata

        if config.capital_pct_per_order is None:
            return signal_config.contracts_per_order, {
                "sizing_method": "fixed_contracts",
                "target_fraction_of_equity": None,
                "cash_budget_dollars": None,
                "kelly_raw_fraction_of_equity": None,
                "kelly_fraction_of_equity": None,
            }

        equity = available_cash + open_cost_basis
        reserve_cash = (signal_config.reserve_cash_pct / 100.0) * equity
        deployable_cash = max(0.0, available_cash - reserve_cash)
        target_cash = equity * (config.capital_pct_per_order / 100.0)
        cash_budget = min(target_cash, deployable_cash)
        metadata = {
            "sizing_method": "capital_pct",
            "target_fraction_of_equity": float(config.capital_pct_per_order / 100.0),
            "cash_budget_dollars": float(cash_budget),
            "kelly_raw_fraction_of_equity": None,
            "kelly_fraction_of_equity": None,
        }
        if cash_budget <= 0:
            return 0, metadata

        _edge_1, _entry_1, _fees_1, cash_required_1 = calculate_cost_metrics(
            side=side,
            predicted_yes_probability=predicted_yes_probability,
            displayed_entry_price_cents=reference_price_cents,
            contracts=1,
            slippage=signal_config.slippage,
        )
        if cash_required_1 > cash_budget + 1e-12:
            return 0, metadata

        slipped_entry_price = (reference_price_cents / 100.0) * (1.0 + signal_config.slippage)
        upper_bound = max(1, int(cash_budget / max(slipped_entry_price, 1e-9)) + 1)
        low = 1
        high = upper_bound
        best = 1
        while low <= high:
            mid = (low + high) // 2
            _edge_mid, _entry_mid, _fees_mid, cash_required_mid = calculate_cost_metrics(
                side=side,
                predicted_yes_probability=predicted_yes_probability,
                displayed_entry_price_cents=reference_price_cents,
                contracts=mid,
                slippage=signal_config.slippage,
            )
            if cash_required_mid <= cash_budget + 1e-12:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best, metadata

    def update_drawdown() -> None:
        nonlocal peak_equity, max_drawdown, max_drawdown_pct
        equity = available_cash + open_cost_basis
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        drawdown_pct = 0.0 if peak_equity <= 0 else drawdown / peak_equity
        max_drawdown = max(max_drawdown, drawdown)
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct * 100.0)

    def settle_positions(cutoff_time: pd.Timestamp) -> None:
        nonlocal available_cash, open_cost_basis, total_pnl, total_hold_minutes, wins
        while open_positions and open_positions[0][0] <= cutoff_time:
            _close_time, _seq, position = heapq.heappop(open_positions)
            open_cost_basis -= position.entry_cost
            available_cash += position.payout
            gross_pnl = position.payout - position.entry_cost
            net_pnl = gross_pnl - position.fees
            total_pnl += net_pnl
            total_hold_minutes += position.hold_minutes
            if net_pnl > 0:
                wins += 1
            ticker_positions = open_positions_by_ticker.get(position.ticker)
            if ticker_positions is not None:
                for index, candidate in enumerate(ticker_positions):
                    if candidate is position:
                        ticker_positions.pop(index)
                        break
                if not ticker_positions:
                    open_positions_by_ticker.pop(position.ticker, None)
            update_drawdown()

    progress = tqdm(total=len(frame), desc=progress_desc, unit="row", leave=False) if progress_desc else None
    try:
        for row_index, row in enumerate(frame.itertuples(index=False), start=1):
            if progress is not None and row_index % 10_000 == 0:
                progress.update(10_000)
            settle_positions(row.created_time)
            if row.tau_minutes < signal_config.min_tau_minutes or row.tau_minutes > signal_config.max_tau_minutes:
                continue
            side = "YES" if row.predicted_yes_probability > row.market_prob else "NO"
            regime_payload = _regime_payload_for_row(row._asdict()) if hasattr(row, "_asdict") else _regime_payload_for_row(row)
            if signal_config.apply_regime_hard_gate and side == "YES" and regime_payload["regime_label"] == "downtrend":
                skipped_due_regime += 1
                continue
            reference_price_cents = _reference_price_cents(side, float(row.market_prob))
            if reference_price_cents < signal_config.price_band_min_cents or reference_price_cents > signal_config.price_band_max_cents:
                skipped_due_price_band += 1
                continue
            existing_positions = open_positions_by_ticker.get(row.ticker, [])
            if existing_positions:
                if not signal_config.allow_stacking:
                    skipped_due_open_ticker += 1
                    continue
                if config.max_entries_per_ticker is not None and len(existing_positions) >= config.max_entries_per_ticker:
                    skipped_due_open_ticker += 1
                    continue
                if any(position.side != side for position in existing_positions):
                    skipped_due_open_ticker += 1
                    continue
                if config.require_price_improvement_for_stack:
                    last_position = existing_positions[-1]
                    if reference_price_cents >= last_position.reference_price_cents:
                        skipped_due_open_ticker += 1
                        continue
            contracts, sizing_metadata = resolve_contracts(
                side=side,
                predicted_yes_probability=float(row.predicted_yes_probability),
                reference_price_cents=reference_price_cents,
            )
            if contracts <= 0:
                continue
            max_acceptable = find_max_acceptable_entry_price_cents(
                side=side,
                predicted_yes_probability=float(row.predicted_yes_probability),
                config=signal_config,
                contracts=contracts,
            )
            if max_acceptable is None or reference_price_cents > max_acceptable:
                skipped_due_post_cost_edge += 1
                continue
            post_cost_edge, entry_cost, fees, cash_required = calculate_cost_metrics(
                side=side,
                predicted_yes_probability=float(row.predicted_yes_probability),
                displayed_entry_price_cents=reference_price_cents,
                contracts=contracts,
                slippage=signal_config.slippage,
            )
            if post_cost_edge + 1e-12 < signal_config.edge_threshold:
                skipped_due_post_cost_edge += 1
                continue
            reserve_cash = (signal_config.reserve_cash_pct / 100.0) * (available_cash + open_cost_basis)
            if available_cash - cash_required < reserve_cash - 1e-12:
                continue

            payout_probability = float(row.actual_outcome if side == "YES" else 1 - row.actual_outcome)
            payout_dollars = payout_probability * contracts
            gross_pnl_dollars = payout_dollars - entry_cost
            net_pnl_dollars = gross_pnl_dollars - fees
            hold_minutes = max(0.0, (row.close_time - row.created_time).total_seconds() / 60.0)

            available_cash -= cash_required
            open_cost_basis += entry_cost
            total_fees += fees
            total_turnover += entry_cost
            position = _OpenPosition(
                ticker=row.ticker,
                side=side,
                reference_price_cents=int(reference_price_cents),
                close_time=row.close_time,
                entry_cost=entry_cost,
                payout=payout_dollars,
                fees=fees,
                hold_minutes=hold_minutes,
            )
            heapq.heappush(
                open_positions,
                (
                    row.close_time,
                    sequence,
                    position,
                ),
            )
            open_positions_by_ticker.setdefault(row.ticker, []).append(position)
            trade_records.append(
                {
                    "ticker": row.ticker,
                    "trade_id": row.trade_id,
                    "created_time": row.created_time,
                    "close_time": row.close_time,
                    "side": side,
                    "sizing_method": sizing_metadata["sizing_method"],
                    "contracts": int(contracts),
                    "actual_outcome": int(row.actual_outcome),
                    "is_win": bool(payout_probability > 0.0),
                    "market_prob": float(row.market_prob),
                    "predicted_yes_probability": float(row.predicted_yes_probability),
                    "tau_minutes": float(row.tau_minutes),
                    "realized_vol_regime_source": getattr(row, "realized_vol_regime_source", "unknown"),
                    "realized_vol_regime_bucket": getattr(
                        row,
                        "realized_vol_regime_bucket",
                        REALIZED_VOL_REGIME_UNKNOWN,
                    ),
                    "realized_vol_regime_bucket_version": getattr(
                        row,
                        "realized_vol_regime_bucket_version",
                        "unknown",
                    ),
                    "realized_vol_regime_metric": (
                        float(row.realized_vol_regime_metric)
                        if pd.notna(getattr(row, "realized_vol_regime_metric", np.nan))
                        else None
                    ),
                    "hour_of_day_et": (
                        int(row.hour_of_day_et) if pd.notna(getattr(row, "hour_of_day_et", np.nan)) else None
                    ),
                    "hour_of_day_et_label": getattr(row, "hour_of_day_et_label", "unknown"),
                    "session_block_et": getattr(row, "session_block_et", "unknown"),
                    "regime_label": regime_payload["regime_label"],
                    "bearish_vote_count": regime_payload["bearish_vote_count"],
                    "bullish_vote_count": regime_payload["bullish_vote_count"],
                    "regime_price_momentum_bearish": regime_payload["regime_price_momentum_bearish"],
                    "regime_signed_flow_bearish": regime_payload["regime_signed_flow_bearish"],
                    "regime_yes_share_bearish": regime_payload["regime_yes_share_bearish"],
                    "regime_price_momentum_bullish": regime_payload["regime_price_momentum_bullish"],
                    "regime_signed_flow_bullish": regime_payload["regime_signed_flow_bullish"],
                    "regime_yes_share_bullish": regime_payload["regime_yes_share_bullish"],
                    "target_fraction_of_equity": sizing_metadata["target_fraction_of_equity"],
                    "cash_budget_dollars": sizing_metadata["cash_budget_dollars"],
                    "kelly_raw_fraction_of_equity": sizing_metadata["kelly_raw_fraction_of_equity"],
                    "kelly_fraction_of_equity": sizing_metadata["kelly_fraction_of_equity"],
                    "reference_price_cents": int(reference_price_cents),
                    "max_acceptable_entry_price_cents": int(max_acceptable),
                    "post_cost_edge": float(post_cost_edge),
                    "entry_cost_dollars": float(entry_cost),
                    "fees_dollars": float(fees),
                    "cash_required_dollars": float(cash_required),
                    "payout_dollars": float(payout_dollars),
                    "gross_pnl_dollars": float(gross_pnl_dollars),
                    "net_pnl_dollars": float(net_pnl_dollars),
                    "hold_minutes": float(hold_minutes),
                }
            )
            sequence += 1
            trades += 1
            update_drawdown()
        if progress is not None:
            progress.update(len(frame) % 10_000)
    finally:
        if progress is not None:
            progress.close()

    if not frame.empty:
        settle_positions(frame["close_time"].max())

    ending_equity = available_cash + open_cost_basis
    return_pct = 0.0 if signal_config.starting_cash_dollars <= 0 else (
        (ending_equity / signal_config.starting_cash_dollars) - 1.0
    ) * 100.0
    objective = total_pnl / max(1.0, max_drawdown)
    result = PolicyResult(
        config=config,
        objective=objective,
        trades=trades,
        net_pnl_dollars=total_pnl,
        max_drawdown_dollars=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        return_pct=return_pct,
        log_loss=overall_log_loss,
        skipped_due_open_ticker=skipped_due_open_ticker,
        skipped_due_price_band=skipped_due_price_band,
        skipped_due_regime=skipped_due_regime,
        skipped_due_post_cost_edge=skipped_due_post_cost_edge,
    )
    trade_frame = pd.DataFrame(trade_records, columns=_trade_record_columns()) if trade_records else _empty_trade_records_frame()
    return result, trade_frame


def _simulate_policy(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    config: PolicyConfig,
    *,
    overall_log_loss: float,
    progress_desc: str | None = None,
) -> PolicyResult:
    result, _trade_records = _simulate_policy_with_trade_records(
        df,
        probabilities,
        config,
        overall_log_loss=overall_log_loss,
        progress_desc=progress_desc,
    )
    return result


def _policy_sort_key(result: PolicyResult) -> tuple[float, float, float]:
    return (result.objective, result.net_pnl_dollars, -result.log_loss)


def _policy_grid_configs() -> list[PolicyConfig]:
    configs: list[PolicyConfig] = []
    for edge_threshold_cents in np.arange(0.5, 6.5, 0.5):
        for min_tau in (0.0, 1.0, 2.0, 3.0):
            for max_tau in (12.0, 14.0, 15.0):
                if min_tau > max_tau:
                    continue
                for price_band_min_cents, price_band_max_cents in ((10, 90), (20, 80), (30, 70)):
                    configs.append(
                        PolicyConfig(
                            edge_threshold_cents=float(edge_threshold_cents),
                            min_tau_minutes=min_tau,
                            max_tau_minutes=max_tau,
                            price_band_min_cents=price_band_min_cents,
                            price_band_max_cents=price_band_max_cents,
                            reserve_cash_pct=30.0,
                        )
                    )
    return configs


def serialize_policy_result(
    result: PolicyResult,
    *,
    selection_mode: str | None = None,
    met_validation_requirements: bool | None = None,
    minimum_trades_required: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "config": asdict(result.config),
        "objective": result.objective,
        "trades": result.trades,
        "net_pnl_dollars": result.net_pnl_dollars,
        "max_drawdown_dollars": result.max_drawdown_dollars,
        "max_drawdown_pct": result.max_drawdown_pct,
        "return_pct": result.return_pct,
        "log_loss": result.log_loss,
        "skipped_due_open_ticker": result.skipped_due_open_ticker,
        "skipped_due_price_band": result.skipped_due_price_band,
        "skipped_due_regime": result.skipped_due_regime,
        "skipped_due_post_cost_edge": result.skipped_due_post_cost_edge,
    }
    if selection_mode is not None:
        payload["selection_mode"] = selection_mode
    if met_validation_requirements is not None:
        payload["met_validation_requirements"] = met_validation_requirements
    if minimum_trades_required is not None:
        payload["minimum_trades_required"] = int(minimum_trades_required)
    return payload


def minimum_policy_trades_for_frame(
    df: pd.DataFrame,
    *,
    trades_per_day: float = MINIMUM_POLICY_TRADES_PER_DAY,
) -> int:
    _require_non_empty_split("validation", df)
    if trades_per_day <= 0:
        raise ValueError("trades_per_day must be positive.")
    created_times = pd.to_datetime(df["created_time"], utc=True, errors="coerce")
    start = created_times.min()
    end = created_times.max()
    if pd.isna(start) or pd.isna(end):
        return max(1, int(np.ceil(float(trades_per_day))))
    span_days = max((end - start).total_seconds() / 86_400.0, 1e-9)
    return max(1, int(np.ceil(span_days * float(trades_per_day))))


def sweep_policy_grid(
    validation_df: pd.DataFrame,
    probabilities: np.ndarray,
    overall_log_loss: float,
    *,
    progress_desc: str | None = None,
) -> list[PolicyResult]:
    _require_non_empty_split("validation", validation_df)
    results: list[PolicyResult] = []
    best_result: PolicyResult | None = None
    configs = _policy_grid_configs()
    if not configs:
        return results
    prepared_inputs = _prepare_standard_policy_inputs(
        validation_df,
        probabilities,
        slippage=configs[0].signal_config.slippage,
    )
    progress = tqdm(configs, desc=progress_desc, unit="policy") if progress_desc else None
    iterator = progress if progress is not None else configs
    try:
        for config in iterator:
            result = _simulate_standard_policy_prepared(
                prepared_inputs,
                config,
                overall_log_loss=overall_log_loss,
            )
            results.append(result)
            if best_result is None or _policy_sort_key(result) > _policy_sort_key(best_result):
                best_result = result
                if progress is not None:
                    progress.set_postfix(
                        best_obj=f"{best_result.objective:.3f}",
                        best_pnl=f"{best_result.net_pnl_dollars:.2f}",
                        best_trades=best_result.trades,
                    )
    finally:
        if progress is not None:
            progress.close()
    return results


def select_policy_candidate(
    policy_results: list[PolicyResult],
    *,
    require_positive_pnl: bool = True,
    minimum_trades: int = MINIMUM_POLICY_TRADES,
    fallback_to_best_overall: bool = False,
) -> tuple[PolicyResult, bool]:
    if not policy_results:
        raise RuntimeError("No validation policy results were generated.")

    strict_candidates = [
        result
        for result in policy_results
        if result.trades >= minimum_trades and (result.net_pnl_dollars > 0 if require_positive_pnl else True)
    ]
    if strict_candidates:
        strict_candidates.sort(key=_policy_sort_key, reverse=True)
        return strict_candidates[0], True

    if not fallback_to_best_overall:
        raise RuntimeError("No validation policy candidate met the profitability and minimum-trade thresholds.")

    fallback_candidates = [result for result in policy_results if result.trades >= minimum_trades]
    if not fallback_candidates:
        fallback_candidates = list(policy_results)
    fallback_candidates.sort(key=_policy_sort_key, reverse=True)
    return fallback_candidates[0], False


def tune_policy(
    validation_df: pd.DataFrame,
    probabilities: np.ndarray,
    overall_log_loss: float,
    *,
    minimum_trades_per_day: float = MINIMUM_POLICY_TRADES_PER_DAY,
) -> tuple[PolicyResult, list[dict[str, object]]]:
    policy_results = sweep_policy_grid(validation_df, probabilities, overall_log_loss)
    minimum_trades = minimum_policy_trades_for_frame(validation_df, trades_per_day=minimum_trades_per_day)
    best_policy, _met_validation_requirements = select_policy_candidate(policy_results, minimum_trades=minimum_trades)
    return best_policy, [serialize_policy_result(result) for result in policy_results]


def evaluate_policy(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    config: PolicyConfig,
    overall_log_loss: float,
    *,
    progress_desc: str | None = None,
) -> dict[str, object]:
    _require_non_empty_split("evaluation", df)
    if progress_desc is None and _is_standard_policy_fast_path_compatible(config):
        prepared_inputs = _prepare_standard_policy_inputs(df, probabilities, slippage=config.signal_config.slippage)
        result = _simulate_standard_policy_prepared(prepared_inputs, config, overall_log_loss=overall_log_loss)
    else:
        result = _simulate_policy(df, probabilities, config, overall_log_loss=overall_log_loss, progress_desc=progress_desc)
    return serialize_policy_result(result)


def _summarize_trade_groups(
    trade_records: pd.DataFrame,
    group_column: str,
    *,
    include_time_range: bool = False,
) -> list[dict[str, object]]:
    if trade_records.empty:
        return []
    summaries: list[dict[str, object]] = []
    total_trades = int(len(trade_records))
    grouped = trade_records.groupby(group_column, dropna=False, sort=False, observed=False)
    for group_value, group in grouped:
        payload: dict[str, object] = {
            group_column: "unknown" if pd.isna(group_value) else group_value,
            "trades": int(len(group)),
            "trade_pct": (float(len(group)) / total_trades * 100.0) if total_trades else 0.0,
            "contracts": int(group["contracts"].sum()),
            "wins": int(group["is_win"].sum()),
            "win_rate": float(group["is_win"].mean()),
            "net_pnl_dollars": float(group["net_pnl_dollars"].sum()),
            "gross_pnl_dollars": float(group["gross_pnl_dollars"].sum()),
            "fees_dollars": float(group["fees_dollars"].sum()),
            "avg_contracts": float(group["contracts"].mean()),
            "avg_net_pnl_dollars": float(group["net_pnl_dollars"].mean()),
            "avg_post_cost_edge": float(group["post_cost_edge"].mean()),
            "avg_reference_price_cents": float(group["reference_price_cents"].mean()),
            "avg_tau_minutes": float(group["tau_minutes"].mean()),
            "avg_hold_minutes": float(group["hold_minutes"].mean()),
        }
        if "target_fraction_of_equity" in group:
            target_fraction = group["target_fraction_of_equity"].dropna()
            payload["avg_target_fraction_of_equity"] = (
                float(target_fraction.mean()) if not target_fraction.empty else None
            )
        if "kelly_fraction_of_equity" in group:
            kelly_fraction = group["kelly_fraction_of_equity"].dropna()
            payload["avg_kelly_fraction_of_equity"] = (
                float(kelly_fraction.mean()) if not kelly_fraction.empty else None
            )
        if include_time_range:
            payload["start_time"] = group["created_time"].min()
            payload["end_time"] = group["created_time"].max()
        summaries.append(payload)
    return summaries


def _summarize_regime_row_counts(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty:
        return []
    summaries: list[dict[str, object]] = []
    total_rows = int(len(df))
    grouped = df.groupby("regime_label", dropna=False, sort=False, observed=False)
    for regime_label, group in grouped:
        side_counts = group["offline_rule_side"].value_counts(dropna=False)
        summaries.append(
            {
                "regime_label": "unknown" if pd.isna(regime_label) else regime_label,
                "rows": int(len(group)),
                "row_pct": (float(len(group)) / total_rows * 100.0) if total_rows else 0.0,
                "offline_rule_yes_rows": int(side_counts.get("YES", 0)),
                "offline_rule_no_rows": int(side_counts.get("NO", 0)),
            }
        )
    return summaries


def _summarize_regime_side_breakdown(trade_records: pd.DataFrame) -> list[dict[str, object]]:
    if trade_records.empty:
        return []
    summaries: list[dict[str, object]] = []
    grouped = trade_records.groupby(["regime_label", "side"], dropna=False, sort=False, observed=False)
    for (regime_label, side), group in grouped:
        summaries.append(
            {
                "regime_label": "unknown" if pd.isna(regime_label) else regime_label,
                "side": "unknown" if pd.isna(side) else side,
                "trades": int(len(group)),
                "contracts": int(group["contracts"].sum()),
                "wins": int(group["is_win"].sum()),
                "win_rate": float(group["is_win"].mean()),
                "net_pnl_dollars": float(group["net_pnl_dollars"].sum()),
                "avg_net_pnl_dollars": float(group["net_pnl_dollars"].mean()),
                "avg_post_cost_edge": float(group["post_cost_edge"].mean()),
            }
        )
    return summaries


def _calibration_bucket_summary(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    bucket_count: int = 10,
) -> list[dict[str, object]]:
    if df.empty:
        return []
    bucket_edges = np.linspace(0.0, 1.0, bucket_count + 1)
    bucket_labels = [f"{bucket_edges[i]:.1f}-{bucket_edges[i + 1]:.1f}" for i in range(bucket_count)]
    actuals = labels(df)
    summary_df = pd.DataFrame(
        {
            "probability": np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6),
            "actual_outcome": actuals,
            "market_prob": df["market_prob"].to_numpy(dtype=np.float64, copy=False),
        }
    )
    summary_df["bucket"] = pd.cut(
        summary_df["probability"],
        bins=bucket_edges,
        labels=bucket_labels,
        include_lowest=True,
        right=True,
    )
    total_rows = int(len(summary_df))
    rows: list[dict[str, object]] = []
    for bucket_label, group in summary_df.groupby("bucket", observed=False, sort=False):
        if group.empty:
            continue
        mean_probability = float(group["probability"].mean())
        empirical_yes_rate = float(group["actual_outcome"].mean())
        rows.append(
            {
                "bucket": bucket_label,
                "count": int(len(group)),
                "count_pct": (float(len(group)) / total_rows * 100.0) if total_rows else 0.0,
                "mean_predicted_yes_probability": mean_probability,
                "empirical_yes_rate": empirical_yes_rate,
                "mean_market_probability": float(group["market_prob"].mean()),
                "calibration_error": mean_probability - empirical_yes_rate,
                "absolute_calibration_error": abs(mean_probability - empirical_yes_rate),
            }
        )
    return rows


def _log_loss_for_group(actual_outcome: pd.Series, probabilities: pd.Series) -> float | None:
    if actual_outcome.empty:
        return None
    return float(log_loss(actual_outcome, probabilities, labels=[0, 1]))


def _summarize_row_trade_breakdown(
    row_frame: pd.DataFrame,
    trade_records: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    ordered_keys: Sequence[tuple[object, ...]] | None = None,
) -> list[dict[str, object]]:
    if row_frame.empty and trade_records.empty:
        return []
    row_total = int(len(row_frame))
    trade_total = int(len(trade_records))
    row_groups = {
        tuple(key if isinstance(key, tuple) else (key,)): group
        for key, group in row_frame.groupby(list(group_columns), dropna=False, sort=False, observed=False)
    }
    trade_groups = {
        tuple(key if isinstance(key, tuple) else (key,)): group
        for key, group in trade_records.groupby(list(group_columns), dropna=False, sort=False, observed=False)
    }
    keys = list(ordered_keys) if ordered_keys is not None else list(dict.fromkeys([*row_groups.keys(), *trade_groups.keys()]))
    summaries: list[dict[str, object]] = []
    for key in keys:
        row_group = row_groups.get(tuple(key), pd.DataFrame())
        trade_group = trade_groups.get(tuple(key), pd.DataFrame())
        payload: dict[str, object] = {}
        for column, value in zip(group_columns, tuple(key), strict=False):
            payload[column] = "unknown" if pd.isna(value) else value
        row_count = int(len(row_group))
        trade_count = int(len(trade_group))
        payload["rows"] = row_count
        payload["row_pct"] = (row_count / row_total * 100.0) if row_total else 0.0
        payload["trades"] = trade_count
        payload["trade_pct"] = (trade_count / trade_total * 100.0) if trade_total else 0.0
        payload["calibrated_log_loss"] = (
            _log_loss_for_group(row_group["actual_outcome"], row_group["calibrated_probability"])
            if row_count
            else None
        )
        payload["raw_log_loss"] = (
            _log_loss_for_group(row_group["actual_outcome"], row_group["raw_probability"])
            if row_count
            else None
        )
        payload["empirical_yes_rate"] = float(row_group["actual_outcome"].mean()) if row_count else None
        payload["mean_market_probability"] = float(row_group["market_prob"].mean()) if row_count else None
        payload["mean_raw_probability"] = float(row_group["raw_probability"].mean()) if row_count else None
        payload["mean_calibrated_probability"] = (
            float(row_group["calibrated_probability"].mean()) if row_count else None
        )
        payload["avg_reference_price_cents"] = (
            float(trade_group["reference_price_cents"].mean()) if trade_count else None
        )
        payload["contracts"] = int(trade_group["contracts"].sum()) if trade_count else 0
        payload["wins"] = int(trade_group["is_win"].sum()) if trade_count else 0
        payload["win_rate"] = float(trade_group["is_win"].mean()) if trade_count else None
        payload["net_pnl_dollars"] = float(trade_group["net_pnl_dollars"].sum()) if trade_count else 0.0
        payload["gross_pnl_dollars"] = float(trade_group["gross_pnl_dollars"].sum()) if trade_count else 0.0
        payload["fees_dollars"] = float(trade_group["fees_dollars"].sum()) if trade_count else 0.0
        payload["avg_net_pnl_dollars"] = float(trade_group["net_pnl_dollars"].mean()) if trade_count else None
        payload["avg_post_cost_edge"] = float(trade_group["post_cost_edge"].mean()) if trade_count else None
        payload["avg_tau_minutes"] = float(trade_group["tau_minutes"].mean()) if trade_count else None
        summaries.append(payload)
    return summaries


def build_policy_diagnostics(
    df: pd.DataFrame,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    config: PolicyConfig,
    *,
    overall_log_loss: float,
    progress_desc: str | None = None,
    time_blocks: int = 10,
) -> tuple[dict[str, object], pd.DataFrame]:
    diagnostics_inputs, realized_vol_metadata = _annotate_diagnostic_context(df)
    diagnostics_inputs["raw_probability"] = np.asarray(raw_probabilities, dtype=np.float64)
    diagnostics_inputs["calibrated_probability"] = np.asarray(calibrated_probabilities, dtype=np.float64)
    regime_inputs = diagnostics_inputs.copy()
    regime_payloads = regime_inputs.apply(_regime_payload_for_row, axis=1, result_type="expand")
    regime_inputs = pd.concat([regime_inputs.reset_index(drop=True), regime_payloads.reset_index(drop=True)], axis=1)
    regime_inputs["offline_rule_side"] = np.where(
        np.asarray(calibrated_probabilities, dtype=np.float64) > regime_inputs["market_prob"].to_numpy(dtype=np.float64, copy=False),
        "YES",
        "NO",
    )
    result, trade_records = _simulate_policy_with_trade_records(
        regime_inputs,
        calibrated_probabilities,
        config,
        overall_log_loss=overall_log_loss,
        progress_desc=progress_desc,
    )
    if not trade_records.empty:
        tau_edges = np.arange(0.0, 16.0 + 2.0, 2.0)
        tau_labels = [f"{int(tau_edges[i])}-{int(tau_edges[i + 1])}" for i in range(len(tau_edges) - 1)]
        trade_records = trade_records.copy()
        trade_records["tau_bucket"] = pd.cut(
            trade_records["tau_minutes"],
            bins=tau_edges,
            labels=tau_labels,
            include_lowest=True,
            right=False,
        )
        price_edges = np.arange(0, 100 + 10, 10)
        price_labels = [f"{price_edges[i]}-{price_edges[i + 1]}" for i in range(len(price_edges) - 1)]
        trade_records["price_bucket"] = pd.cut(
            trade_records["reference_price_cents"],
            bins=price_edges,
            labels=price_labels,
            include_lowest=True,
            right=False,
        )
        block_count = max(1, min(time_blocks, len(trade_records)))
        if block_count == 1:
            trade_records["time_block"] = "block_01"
        else:
            labels_for_blocks = [f"block_{index:02d}" for index in range(1, block_count + 1)]
            trade_records["time_block"] = pd.qcut(
                np.arange(len(trade_records)),
                q=block_count,
                labels=labels_for_blocks,
                duplicates="drop",
            )
    vol_order = [
        (source, bucket) for source in ("spot", "market_prob") for bucket in REALIZED_VOL_REGIME_LABELS
    ] + [("unknown", REALIZED_VOL_REGIME_UNKNOWN)]
    hour_order = [(hour, f"{hour:02d}") for hour in range(24)]
    session_order = [(label,) for label in SESSION_BLOCK_LABELS]

    counterfactual_policy = None
    if not config.apply_regime_hard_gate:
        counterfactual_config = PolicyConfig(
            edge_threshold_cents=config.edge_threshold_cents,
            min_tau_minutes=config.min_tau_minutes,
            max_tau_minutes=config.max_tau_minutes,
            price_band_min_cents=config.price_band_min_cents,
            price_band_max_cents=config.price_band_max_cents,
            reserve_cash_pct=config.reserve_cash_pct,
            apply_regime_hard_gate=True,
            starting_cash_dollars=config.starting_cash_dollars,
            contracts_per_order=config.contracts_per_order,
            capital_pct_per_order=config.capital_pct_per_order,
            kelly_fraction_multiplier=config.kelly_fraction_multiplier,
            kelly_fraction_cap_pct=config.kelly_fraction_cap_pct,
            slippage_pct=config.slippage_pct,
            allow_stacking=config.allow_stacking,
            max_entries_per_ticker=config.max_entries_per_ticker,
            require_price_improvement_for_stack=config.require_price_improvement_for_stack,
        )
        counterfactual_policy = serialize_policy_result(
            _simulate_policy(
                df,
                calibrated_probabilities,
                counterfactual_config,
                overall_log_loss=overall_log_loss,
                progress_desc=progress_desc,
            )
        )

    diagnostics = {
        "policy_metrics": serialize_policy_result(result),
        "trade_count": int(len(trade_records)),
        "side_breakdown": _summarize_trade_groups(trade_records, "side"),
        "regime_breakdown": _summarize_trade_groups(trade_records, "regime_label"),
        "regime_side_breakdown": _summarize_regime_side_breakdown(trade_records),
        "regime_row_counts": _summarize_regime_row_counts(regime_inputs),
        "realized_vol_regime_metadata": realized_vol_metadata,
        "realized_vol_regime_breakdown": _summarize_row_trade_breakdown(
            regime_inputs,
            trade_records,
            ["realized_vol_regime_source", "realized_vol_regime_bucket"],
            ordered_keys=vol_order,
        ),
        "hour_of_day_breakdown": _summarize_row_trade_breakdown(
            regime_inputs,
            trade_records,
            ["hour_of_day_et", "hour_of_day_et_label"],
            ordered_keys=hour_order,
        ),
        "session_block_breakdown": _summarize_row_trade_breakdown(
            regime_inputs,
            trade_records,
            ["session_block_et"],
            ordered_keys=session_order,
        ),
        "tau_bucket_breakdown": _summarize_trade_groups(trade_records, "tau_bucket"),
        "price_bucket_breakdown": _summarize_trade_groups(trade_records, "price_bucket"),
        "time_block_breakdown": _summarize_trade_groups(trade_records, "time_block", include_time_range=True),
        "calibration": {
            "raw_probability_buckets": _calibration_bucket_summary(df, raw_probabilities),
            "calibrated_probability_buckets": _calibration_bucket_summary(df, calibrated_probabilities),
        },
        "regime_counterfactuals": (
            {"downtrend_blocks_yes": counterfactual_policy}
            if counterfactual_policy is not None
            else {}
        ),
    }
    return diagnostics, trade_records


def format_realized_vol_regime_source_summary(split_name: str, diagnostics: dict[str, object]) -> str:
    metadata = diagnostics.get("realized_vol_regime_metadata", {})
    source_rows = metadata.get("row_source_summary", [])
    if not source_rows:
        return f"{split_name} realized vol regimes: unavailable"
    parts = [
        f"{row['realized_vol_regime_source']} {int(row['rows'])} rows ({float(row['row_pct']):.1f}%)"
        for row in source_rows
    ]
    return f"{split_name} realized vol regimes: " + " / ".join(parts)


def build_walk_forward_folds(
    markets_df: pd.DataFrame,
    num_blocks: int = 6,
    *,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
    purge_embargo: bool = False,
    purge_embargo_gap_seconds: int | None = None,
    purge_embargo_safety_margin_seconds: int = DEFAULT_PURGE_EMBARGO_SAFETY_MARGIN_SECONDS,
) -> list[WalkForwardFold]:
    ordered_markets = markets_df.copy()
    if "close_time" in ordered_markets.columns:
        ordered_markets["close_time"] = pd.to_datetime(ordered_markets["close_time"], utc=True)
        ordered_markets = ordered_markets.sort_values(["close_time", "ticker"]).reset_index(drop=True)
    else:
        ordered_markets = ordered_markets.reset_index(drop=True)
    tickers = list(ordered_markets["ticker"].astype(str))
    blocks = np.array_split(np.arange(len(tickers)), num_blocks)
    folds: list[WalkForwardFold] = []
    resolved_gap_seconds = (
        int(purge_embargo_gap_seconds)
        if purge_embargo_gap_seconds is not None
        else compute_purge_embargo_gap_seconds(
            ordered_markets,
            feature_schema=feature_schema,
            safety_margin_seconds=purge_embargo_safety_margin_seconds,
        )
    )
    gap = pd.Timedelta(seconds=resolved_gap_seconds)
    for fold_index in range(max(0, len(blocks) - 2)):
        validation_indices = blocks[fold_index + 1]
        test_indices = blocks[fold_index + 2]
        if not purge_embargo:
            train_tickers = tuple(tickers[index] for block in blocks[: fold_index + 1] for index in block.tolist())
            validation_tickers = tuple(tickers[index] for index in validation_indices.tolist())
            test_tickers = tuple(tickers[index] for index in test_indices.tolist())
        else:
            if "close_time" not in ordered_markets.columns:
                raise ValueError("Purged/embargoed walk-forward folds require close_time in the markets frame.")
            close_times = pd.to_datetime(ordered_markets["close_time"], utc=True)
            validation_start_time = pd.Timestamp(close_times.iloc[int(validation_indices[0])])
            validation_end_time = pd.Timestamp(close_times.iloc[int(validation_indices[-1])])
            core_test_start_time = pd.Timestamp(close_times.iloc[int(test_indices[0])])
            core_test_end_time = pd.Timestamp(close_times.iloc[int(test_indices[-1])])
            train_cutoff = validation_start_time - gap
            test_cutoff = max(core_test_start_time, validation_end_time + gap)
            train_mask = close_times < train_cutoff
            test_mask = (close_times >= test_cutoff) & (close_times <= core_test_end_time)
            train_tickers = tuple(ordered_markets.loc[train_mask, "ticker"].astype(str))
            validation_tickers = tuple(ordered_markets.iloc[validation_indices]["ticker"].astype(str))
            test_tickers = tuple(ordered_markets.loc[test_mask, "ticker"].astype(str))
            if not train_tickers or not validation_tickers or not test_tickers:
                raise ValueError(
                    f"Purged/embargoed walk-forward fold {fold_index + 1} is empty. "
                    "Reduce the gap or increase the time span."
                )
        folds.append(
            WalkForwardFold(
                fold_index=fold_index + 1,
                train_tickers=train_tickers,
                validation_tickers=validation_tickers,
                test_tickers=test_tickers,
            )
        )
    return folds


def publish_latest_artifacts(
    run_dir: Path,
    latest_dir: Path | None = None,
    *,
    model_family: str = "lightgbm",
) -> Path:
    latest_dir = latest_dir or (run_dir.parent / "latest")
    if run_dir == latest_dir:
        return latest_dir

    family_files: dict[str, tuple[tuple[str, str], ...]] = {
        "lightgbm": (
            ("lightgbm/model.txt", "lightgbm/model.txt"),
            ("lightgbm/metrics.json", "lightgbm/metrics.json"),
            ("lightgbm/feature_importance.json", "lightgbm/feature_importance.json"),
            ("lightgbm/calibration.json", "lightgbm/calibration.json"),
        ),
        "lasso": (
            ("lasso/model.joblib", "lasso/model.joblib"),
            ("lasso/metrics.json", "lasso/metrics.json"),
            ("lasso/coefficients.json", "lasso/coefficients.json"),
            ("lasso/calibration.json", "lasso/calibration.json"),
        ),
        "bagged_lasso": (
            ("bagged_lasso/model.joblib", "bagged_lasso/model.joblib"),
            ("bagged_lasso/metrics.json", "bagged_lasso/metrics.json"),
            ("bagged_lasso/coefficients.json", "bagged_lasso/coefficients.json"),
            ("bagged_lasso/calibration.json", "bagged_lasso/calibration.json"),
        ),
        "elastic_net": (
            ("elastic_net/model.joblib", "elastic_net/model.joblib"),
            ("elastic_net/metrics.json", "elastic_net/metrics.json"),
            ("elastic_net/coefficients.json", "elastic_net/coefficients.json"),
            ("elastic_net/calibration.json", "elastic_net/calibration.json"),
        ),
        "linear_svm": (
            ("linear_svm/model.joblib", "linear_svm/model.joblib"),
            ("linear_svm/metrics.json", "linear_svm/metrics.json"),
            ("linear_svm/coefficients.json", "linear_svm/coefficients.json"),
            ("linear_svm/calibration.json", "linear_svm/calibration.json"),
        ),
    }
    files_to_copy = (
        *family_files.get(model_family, ()),
        ("feature_manifest.json", "feature_manifest.json"),
        ("split_manifest.json", "split_manifest.json"),
        ("policy.json", "policy.json"),
        ("summary.json", "summary.json"),
    )
    for source_rel, destination_rel in files_to_copy:
        source = run_dir / source_rel
        if not source.exists():
            continue
        destination = latest_dir / destination_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, destination)

    write_json(
        latest_dir / "run.json",
        {
            "source_run_dir": run_dir,
            "published_at": datetime.now(UTC).isoformat(),
        },
    )
    return latest_dir


def evaluate_walk_forward(
    cache_dir: Path,
    markets_df: pd.DataFrame,
    params: dict[str, object],
    *,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
    purge_embargo: bool = False,
    purge_embargo_gap_seconds: int | None = None,
    purge_embargo_safety_margin_seconds: int = DEFAULT_PURGE_EMBARGO_SAFETY_MARGIN_SECONDS,
    fallback_to_best_overall_policy: bool = False,
    progress_desc: str | None = None,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    folds = build_walk_forward_folds(
        markets_df,
        feature_schema=feature_schema,
        purge_embargo=purge_embargo,
        purge_embargo_gap_seconds=purge_embargo_gap_seconds,
        purge_embargo_safety_margin_seconds=purge_embargo_safety_margin_seconds,
    )
    progress = tqdm(folds, desc=progress_desc, unit="fold") if progress_desc else None
    iterator = progress if progress is not None else folds
    try:
        for fold in iterator:
            train_df = load_feature_dataset(
                cache_dir,
                fold.train_tickers,
                progress_desc=f"Fold {fold.fold_index} train dataset" if progress is not None else None,
                feature_schema=feature_schema,
            )
            validation_df = load_feature_dataset(
                cache_dir,
                fold.validation_tickers,
                progress_desc=f"Fold {fold.fold_index} validation dataset" if progress is not None else None,
                feature_schema=feature_schema,
            )
            test_df = load_feature_dataset(
                cache_dir,
                fold.test_tickers,
                progress_desc=f"Fold {fold.fold_index} test dataset" if progress is not None else None,
                feature_schema=feature_schema,
            )
            model, metrics = train_lightgbm_model(train_df, validation_df, params)
            validation_raw = raw_predictions(model, validation_df)
            calibration = fit_platt_scaler(validation_raw, labels(validation_df))
            validation_calibrated = calibration.apply(validation_raw)  # type: ignore[return-value]
            validation_log_loss = _binary_log_loss(labels(validation_df), validation_calibrated)
            policy_results = sweep_policy_grid(
                validation_df,
                validation_calibrated,
                validation_log_loss,
                progress_desc=f"Fold {fold.fold_index} policy sweep" if progress is not None else None,
            )
            minimum_trades = minimum_policy_trades_for_frame(validation_df)
            best_policy, met_validation_requirements = select_policy_candidate(
                policy_results,
                minimum_trades=minimum_trades,
                fallback_to_best_overall=fallback_to_best_overall_policy,
            )
            test_raw = raw_predictions(model, test_df)
            test_calibrated = calibration.apply(test_raw)  # type: ignore[return-value]
            test_log_loss = _binary_log_loss(labels(test_df), test_calibrated)
            test_metrics = evaluate_policy(
                test_df,
                test_calibrated,
                best_policy.config,
                test_log_loss,
                progress_desc=f"Fold {fold.fold_index} test policy" if progress is not None else None,
            )
            summaries.append(
                {
                    "fold_index": fold.fold_index,
                    "train_tickers": len(fold.train_tickers),
                    "validation_tickers": len(fold.validation_tickers),
                    "test_tickers": len(fold.test_tickers),
                    "best_iteration": metrics["best_iteration"],
                    "validation_log_loss": validation_log_loss,
                    "minimum_validation_trades_required": minimum_trades,
                    "validation_policy_selection_mode": "strict" if met_validation_requirements else "fallback_best_overall",
                    "validation_policy_met_requirements": met_validation_requirements,
                    "test": test_metrics,
                }
            )
    finally:
        if progress is not None:
            progress.close()
    return summaries


def _evaluate_walk_forward_regularized_logistic(
    cache_dir: Path,
    markets_df: pd.DataFrame,
    params: dict[str, object],
    *,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
    purge_embargo: bool = False,
    purge_embargo_gap_seconds: int | None = None,
    purge_embargo_safety_margin_seconds: int = DEFAULT_PURGE_EMBARGO_SAFETY_MARGIN_SECONDS,
    train_fn: Any,
    predict_fn: Any,
    fallback_to_best_overall_policy: bool = False,
    progress_desc: str | None = None,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    folds = build_walk_forward_folds(
        markets_df,
        feature_schema=feature_schema,
        purge_embargo=purge_embargo,
        purge_embargo_gap_seconds=purge_embargo_gap_seconds,
        purge_embargo_safety_margin_seconds=purge_embargo_safety_margin_seconds,
    )
    progress = tqdm(folds, desc=progress_desc, unit="fold") if progress_desc else None
    iterator = progress if progress is not None else folds
    try:
        for fold in iterator:
            train_df = load_feature_dataset(
                cache_dir,
                fold.train_tickers,
                progress_desc=f"Fold {fold.fold_index} train dataset" if progress is not None else None,
                feature_schema=feature_schema,
            )
            validation_df = load_feature_dataset(
                cache_dir,
                fold.validation_tickers,
                progress_desc=f"Fold {fold.fold_index} validation dataset" if progress is not None else None,
                feature_schema=feature_schema,
            )
            test_df = load_feature_dataset(
                cache_dir,
                fold.test_tickers,
                progress_desc=f"Fold {fold.fold_index} test dataset" if progress is not None else None,
                feature_schema=feature_schema,
            )
            artifact, metrics = train_fn(train_df, validation_df, params)
            validation_raw = predict_fn(artifact, validation_df)
            calibration = fit_platt_scaler(validation_raw, labels(validation_df))
            validation_calibrated = calibration.apply(validation_raw)  # type: ignore[return-value]
            validation_log_loss = _binary_log_loss(labels(validation_df), validation_calibrated)
            policy_results = sweep_policy_grid(
                validation_df,
                validation_calibrated,
                validation_log_loss,
                progress_desc=f"Fold {fold.fold_index} policy sweep" if progress is not None else None,
            )
            minimum_trades = minimum_policy_trades_for_frame(validation_df)
            best_policy, met_validation_requirements = select_policy_candidate(
                policy_results,
                minimum_trades=minimum_trades,
                fallback_to_best_overall=fallback_to_best_overall_policy,
            )
            test_raw = predict_fn(artifact, test_df)
            test_calibrated = calibration.apply(test_raw)  # type: ignore[return-value]
            test_log_loss = _binary_log_loss(labels(test_df), test_calibrated)
            test_metrics = evaluate_policy(
                test_df,
                test_calibrated,
                best_policy.config,
                test_log_loss,
                progress_desc=f"Fold {fold.fold_index} test policy" if progress is not None else None,
            )
            summaries.append(
                {
                    "fold_index": fold.fold_index,
                    "train_tickers": len(fold.train_tickers),
                    "validation_tickers": len(fold.validation_tickers),
                    "test_tickers": len(fold.test_tickers),
                    "best_iteration": None,
                    "validation_log_loss": validation_log_loss,
                    "minimum_validation_trades_required": minimum_trades,
                    "validation_policy_selection_mode": "strict" if met_validation_requirements else "fallback_best_overall",
                    "validation_policy_met_requirements": met_validation_requirements,
                    "test": test_metrics,
                }
            )
    finally:
        if progress is not None:
            progress.close()
    return summaries


def evaluate_walk_forward_lasso(
    cache_dir: Path,
    markets_df: pd.DataFrame,
    params: dict[str, object],
    *,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
    purge_embargo: bool = False,
    purge_embargo_gap_seconds: int | None = None,
    purge_embargo_safety_margin_seconds: int = DEFAULT_PURGE_EMBARGO_SAFETY_MARGIN_SECONDS,
    fallback_to_best_overall_policy: bool = False,
    progress_desc: str | None = None,
) -> list[dict[str, object]]:
    return _evaluate_walk_forward_regularized_logistic(
        cache_dir,
        markets_df,
        params,
        feature_schema=feature_schema,
        purge_embargo=purge_embargo,
        purge_embargo_gap_seconds=purge_embargo_gap_seconds,
        purge_embargo_safety_margin_seconds=purge_embargo_safety_margin_seconds,
        train_fn=train_lasso_model,
        predict_fn=lasso_raw_predictions,
        fallback_to_best_overall_policy=fallback_to_best_overall_policy,
        progress_desc=progress_desc,
    )


def evaluate_walk_forward_bagged_lasso(
    cache_dir: Path,
    markets_df: pd.DataFrame,
    params: dict[str, object],
    *,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
    purge_embargo: bool = False,
    purge_embargo_gap_seconds: int | None = None,
    purge_embargo_safety_margin_seconds: int = DEFAULT_PURGE_EMBARGO_SAFETY_MARGIN_SECONDS,
    fallback_to_best_overall_policy: bool = False,
    progress_desc: str | None = None,
) -> list[dict[str, object]]:
    return _evaluate_walk_forward_regularized_logistic(
        cache_dir,
        markets_df,
        params,
        feature_schema=feature_schema,
        purge_embargo=purge_embargo,
        purge_embargo_gap_seconds=purge_embargo_gap_seconds,
        purge_embargo_safety_margin_seconds=purge_embargo_safety_margin_seconds,
        train_fn=train_bagged_lasso_model,
        predict_fn=bagged_lasso_raw_predictions,
        fallback_to_best_overall_policy=fallback_to_best_overall_policy,
        progress_desc=progress_desc,
    )


def evaluate_walk_forward_elastic_net(
    cache_dir: Path,
    markets_df: pd.DataFrame,
    params: dict[str, object],
    *,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
    purge_embargo: bool = False,
    purge_embargo_gap_seconds: int | None = None,
    purge_embargo_safety_margin_seconds: int = DEFAULT_PURGE_EMBARGO_SAFETY_MARGIN_SECONDS,
    fallback_to_best_overall_policy: bool = False,
    progress_desc: str | None = None,
) -> list[dict[str, object]]:
    return _evaluate_walk_forward_regularized_logistic(
        cache_dir,
        markets_df,
        params,
        feature_schema=feature_schema,
        purge_embargo=purge_embargo,
        purge_embargo_gap_seconds=purge_embargo_gap_seconds,
        purge_embargo_safety_margin_seconds=purge_embargo_safety_margin_seconds,
        train_fn=train_elastic_net_model,
        predict_fn=elastic_net_raw_predictions,
        fallback_to_best_overall_policy=fallback_to_best_overall_policy,
        progress_desc=progress_desc,
    )


def evaluate_walk_forward_linear_svm(
    cache_dir: Path,
    markets_df: pd.DataFrame,
    params: dict[str, object],
    *,
    feature_schema: str = DEFAULT_FEATURE_SCHEMA,
    purge_embargo: bool = False,
    purge_embargo_gap_seconds: int | None = None,
    purge_embargo_safety_margin_seconds: int = DEFAULT_PURGE_EMBARGO_SAFETY_MARGIN_SECONDS,
    fallback_to_best_overall_policy: bool = False,
    progress_desc: str | None = None,
) -> list[dict[str, object]]:
    return _evaluate_walk_forward_regularized_logistic(
        cache_dir,
        markets_df,
        params,
        feature_schema=feature_schema,
        purge_embargo=purge_embargo,
        purge_embargo_gap_seconds=purge_embargo_gap_seconds,
        purge_embargo_safety_margin_seconds=purge_embargo_safety_margin_seconds,
        train_fn=train_linear_svm_model,
        predict_fn=linear_svm_raw_predictions,
        fallback_to_best_overall_policy=fallback_to_best_overall_policy,
        progress_desc=progress_desc,
    )
