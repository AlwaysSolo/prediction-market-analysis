from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from joblib import load as joblib_load
from lightgbm import Booster

from src.live.kalshi.calibration import PlattCalibration, load_platt_calibration
from src.live.kalshi.feature_engine import KalshiFeatureStateEngine
from src.live.kalshi.features import (
    ALL_FEATURE_ORDER,
    FEATURE_ORDER,
    HOURLY_CONTEXT_REQUIRED_FEATURE_NAMES,
    KalshiFeatureState,
    KalshiFeatureUpdate,
)

_DEFAULT_LIGHTGBM_MODEL_FILE = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "kalshi"
    / "kxbtc15m_lightgbm"
    / "latest"
    / "lightgbm"
    / "model.txt"
)
_LEGACY_DEFAULT_LIGHTGBM_MODEL_FILE = (
    Path(__file__).resolve().parents[3]
    / "Gemini scripts"
    / "model_tournament_output"
    / "KXBTC15M_all_rows"
    / "lightgbm"
    / "model.txt"
)
DEFAULT_LIGHTGBM_MODEL_FILE = (
    _DEFAULT_LIGHTGBM_MODEL_FILE if _DEFAULT_LIGHTGBM_MODEL_FILE.exists() else _LEGACY_DEFAULT_LIGHTGBM_MODEL_FILE
)
DEFAULT_LASSO_MODEL_FILE = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "kalshi"
    / "kxbtc15m_lasso"
    / "latest"
    / "lasso"
    / "model.joblib"
)
DEFAULT_ELASTIC_NET_MODEL_FILE = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "kalshi"
    / "kxbtc15m_elastic_net"
    / "latest"
    / "elastic_net"
    / "model.joblib"
)
DEFAULT_BAGGED_LASSO_MODEL_FILE = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "kalshi"
    / "kxbtc15m_bagged_lasso"
    / "latest"
    / "bagged_lasso"
    / "model.joblib"
)
DEFAULT_LINEAR_SVM_MODEL_FILE = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "kalshi"
    / "kxbtc15m_linear_svm"
    / "latest"
    / "linear_svm"
    / "model.joblib"
)

Callback = Callable[["KalshiLightGBMScoreUpdate"], Awaitable[None] | None]
_FEATURE_NAME_TO_INDEX = {feature_name: index for index, feature_name in enumerate(ALL_FEATURE_ORDER)}
_HOURLY_CONTEXT_FEATURE_NAMES = frozenset(HOURLY_CONTEXT_REQUIRED_FEATURE_NAMES)


@dataclass(frozen=True)
class KalshiLightGBMScorerConfig:
    model_file: Path = field(default_factory=lambda: DEFAULT_LIGHTGBM_MODEL_FILE)
    metrics_file: Path | None = None
    calibration_file: Path | None = None
    prediction_clip_min: float = 1e-6
    prediction_clip_max: float = 1.0 - 1e-6
    use_best_iteration: bool = True
    apply_calibration: bool = True


@dataclass(frozen=True)
class LoadedLightGBMModel:
    booster: Booster
    model_file: Path
    best_iteration: int | None
    calibration: PlattCalibration | None = None
    feature_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class KalshiRegularizedLogisticScorerConfig:
    model_file: Path = field(default_factory=lambda: DEFAULT_LASSO_MODEL_FILE)
    metrics_file: Path | None = None
    calibration_file: Path | None = None
    prediction_clip_min: float = 1e-6
    prediction_clip_max: float = 1.0 - 1e-6
    apply_calibration: bool = True


@dataclass(frozen=True)
class KalshiLinearSVMScorerConfig:
    model_file: Path = field(default_factory=lambda: DEFAULT_LINEAR_SVM_MODEL_FILE)
    metrics_file: Path | None = None
    calibration_file: Path | None = None
    prediction_clip_min: float = 1e-6
    prediction_clip_max: float = 1.0 - 1e-6
    apply_calibration: bool = True


@dataclass(frozen=True)
class LoadedRegularizedLogisticModel:
    scaler: Any
    model: Any
    model_file: Path
    calibration: PlattCalibration | None = None
    feature_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class KalshiLightGBMScoreState:
    ticker: str
    event_time: datetime
    market_prob: float
    tau_minutes: float
    predicted_yes_probability: float
    model_edge: float
    model_file: Path
    last_yes_price_cents: int | None = None
    last_price_cents: int | None = None
    yes_bid_cents: int | None = None
    yes_ask_cents: int | None = None
    no_bid_cents: int | None = None
    no_ask_cents: int | None = None
    ticker_update_time: datetime | None = None
    trade_yes_prob: float | None = None
    quote_mid_prob: float | None = None
    quote_spread_cents: int | None = None
    buy_yes_price_cents: int | None = None
    buy_no_price_cents: int | None = None
    quote_age_seconds: float | None = None
    last_to_mid_gap: float | None = None
    price_momentum: float | None = None
    signed_contracts_sum_300s: float | None = None
    yes_taker_share_300s: float | None = None
    received_at: datetime | None = None
    event_id: str = ""
    raw_event_id: str | None = None
    source: str = "snapshot"


@dataclass(frozen=True)
class KalshiLightGBMScoreUpdate:
    ticker: str
    event_time: datetime
    market_prob: float
    tau_minutes: float
    predicted_yes_probability: float
    model_edge: float
    model_file: Path
    last_yes_price_cents: int | None = None
    last_price_cents: int | None = None
    yes_bid_cents: int | None = None
    yes_ask_cents: int | None = None
    no_bid_cents: int | None = None
    no_ask_cents: int | None = None
    ticker_update_time: datetime | None = None
    trade_yes_prob: float | None = None
    quote_mid_prob: float | None = None
    quote_spread_cents: int | None = None
    buy_yes_price_cents: int | None = None
    buy_no_price_cents: int | None = None
    quote_age_seconds: float | None = None
    last_to_mid_gap: float | None = None
    price_momentum: float | None = None
    signed_contracts_sum_300s: float | None = None
    yes_taker_share_300s: float | None = None
    received_at: datetime | None = None
    event_id: str = ""
    raw_event_id: str | None = None
    source: str = "snapshot"


def resolve_metrics_file(config: KalshiLightGBMScorerConfig) -> Path | None:
    if config.metrics_file is not None:
        return Path(config.metrics_file)
    return Path(config.model_file).with_name("metrics.json")


def resolve_calibration_file(config: KalshiLightGBMScorerConfig) -> Path | None:
    if config.calibration_file is not None:
        return Path(config.calibration_file)
    return Path(config.model_file).with_name("calibration.json")


def resolve_regularized_metrics_file(config: KalshiRegularizedLogisticScorerConfig) -> Path | None:
    if config.metrics_file is not None:
        return Path(config.metrics_file)
    return Path(config.model_file).with_name("metrics.json")


def resolve_regularized_calibration_file(config: KalshiRegularizedLogisticScorerConfig) -> Path | None:
    if config.calibration_file is not None:
        return Path(config.calibration_file)
    return Path(config.model_file).with_name("calibration.json")


def resolve_linear_svm_metrics_file(config: KalshiLinearSVMScorerConfig) -> Path | None:
    if config.metrics_file is not None:
        return Path(config.metrics_file)
    return Path(config.model_file).with_name("metrics.json")


def resolve_linear_svm_calibration_file(config: KalshiLinearSVMScorerConfig) -> Path | None:
    if config.calibration_file is not None:
        return Path(config.calibration_file)
    return Path(config.model_file).with_name("calibration.json")


def load_lightgbm_model_artifact(config: KalshiLightGBMScorerConfig | None = None) -> LoadedLightGBMModel:
    config = config or KalshiLightGBMScorerConfig()
    model_file = Path(config.model_file)
    if not model_file.exists():
        raise FileNotFoundError(f"LightGBM model file not found: {model_file}")

    best_iteration: int | None = None
    feature_names: tuple[str, ...] | None = None
    metrics_file = resolve_metrics_file(config)
    if config.use_best_iteration and metrics_file is not None and metrics_file.exists():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        best_iteration_value = metrics.get("best_iteration")
        if best_iteration_value is not None:
            best_iteration = int(best_iteration_value)
        feature_names_value = metrics.get("feature_names")
        if feature_names_value:
            feature_names = tuple(str(name) for name in feature_names_value)

    calibration: PlattCalibration | None = None
    calibration_file = resolve_calibration_file(config)
    if config.apply_calibration and calibration_file is not None and calibration_file.exists():
        calibration = load_platt_calibration(calibration_file)

    booster = Booster(model_file=str(model_file))
    return LoadedLightGBMModel(
        booster=booster,
        model_file=model_file,
        best_iteration=best_iteration,
        calibration=calibration,
        feature_names=feature_names,
    )


def load_regularized_logistic_model_artifact(
    config: KalshiRegularizedLogisticScorerConfig | None = None,
) -> LoadedRegularizedLogisticModel:
    config = config or KalshiRegularizedLogisticScorerConfig()
    model_file = Path(config.model_file)
    if not model_file.exists():
        raise FileNotFoundError(f"Regularized logistic model file not found: {model_file}")

    feature_names: tuple[str, ...] | None = None
    metrics_file = resolve_regularized_metrics_file(config)
    if metrics_file is not None and metrics_file.exists():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        feature_names_value = metrics.get("feature_names")
        if feature_names_value:
            feature_names = tuple(str(name) for name in feature_names_value)

    calibration: PlattCalibration | None = None
    calibration_file = resolve_regularized_calibration_file(config)
    if config.apply_calibration and calibration_file is not None and calibration_file.exists():
        calibration = load_platt_calibration(calibration_file)

    payload = joblib_load(model_file)
    return LoadedRegularizedLogisticModel(
        scaler=payload["scaler"],
        model=payload["model"],
        model_file=model_file,
        calibration=calibration,
        feature_names=feature_names,
    )


def load_linear_svm_model_artifact(config: KalshiLinearSVMScorerConfig | None = None) -> LoadedRegularizedLogisticModel:
    config = config or KalshiLinearSVMScorerConfig()
    model_file = Path(config.model_file)
    if not model_file.exists():
        raise FileNotFoundError(f"Linear SVM model file not found: {model_file}")

    feature_names: tuple[str, ...] | None = None
    metrics_file = resolve_linear_svm_metrics_file(config)
    if metrics_file is not None and metrics_file.exists():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        feature_names_value = metrics.get("feature_names")
        if feature_names_value:
            feature_names = tuple(str(name) for name in feature_names_value)

    calibration: PlattCalibration | None = None
    calibration_file = resolve_linear_svm_calibration_file(config)
    if config.apply_calibration and calibration_file is not None and calibration_file.exists():
        calibration = load_platt_calibration(calibration_file)

    payload = joblib_load(model_file)
    return LoadedRegularizedLogisticModel(
        scaler=payload["scaler"],
        model=payload["model"],
        model_file=model_file,
        calibration=calibration,
        feature_names=feature_names,
    )


def clip_prediction(
    probability: float,
    config: KalshiLightGBMScorerConfig | KalshiRegularizedLogisticScorerConfig | None = None,
) -> float:
    config = config or KalshiLightGBMScorerConfig()
    return float(np.clip(probability, config.prediction_clip_min, config.prediction_clip_max))


def predict_yes_probability(
    model: LoadedLightGBMModel,
    feature_row: np.ndarray,
    config: KalshiLightGBMScorerConfig | None = None,
) -> float:
    config = config or KalshiLightGBMScorerConfig()
    aligned_feature_row = _align_feature_row(model, feature_row)
    if config.use_best_iteration and model.best_iteration is not None:
        prediction = float(model.booster.predict(aligned_feature_row, num_iteration=model.best_iteration)[0])
    else:
        prediction = float(model.booster.predict(aligned_feature_row)[0])
    if config.apply_calibration and model.calibration is not None:
        prediction = float(model.calibration.apply(prediction))
    return clip_prediction(prediction, config)


def predict_regularized_logistic_yes_probability(
    model: LoadedRegularizedLogisticModel,
    feature_row: np.ndarray,
    config: KalshiRegularizedLogisticScorerConfig | None = None,
) -> float:
    config = config or KalshiRegularizedLogisticScorerConfig()
    aligned_feature_row = _align_feature_row_by_names(
        feature_names=model.feature_names,
        model_file=model.model_file,
        feature_row=feature_row,
    )
    transformed_feature_row = model.scaler.transform(aligned_feature_row)
    prediction = float(model.model.predict_proba(transformed_feature_row)[0, 1])
    if config.apply_calibration and model.calibration is not None:
        prediction = float(model.calibration.apply(prediction))
    return clip_prediction(prediction, config)


def _sigmoid_from_scores(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def predict_linear_svm_yes_probability(
    model: LoadedRegularizedLogisticModel,
    feature_row: np.ndarray,
    config: KalshiLinearSVMScorerConfig | None = None,
) -> float:
    config = config or KalshiLinearSVMScorerConfig()
    aligned_feature_row = _align_feature_row_by_names(
        feature_names=model.feature_names,
        model_file=model.model_file,
        feature_row=feature_row,
    )
    transformed_feature_row = model.scaler.transform(aligned_feature_row)
    decision_score = np.asarray(model.model.decision_function(transformed_feature_row), dtype=np.float64)
    prediction = float(_sigmoid_from_scores(decision_score)[0])
    if config.apply_calibration and model.calibration is not None:
        prediction = float(model.calibration.apply(prediction))
    return clip_prediction(prediction, config)


def _align_feature_row(model: LoadedLightGBMModel, feature_row: np.ndarray) -> np.ndarray:
    return _align_feature_row_by_names(
        feature_names=model.feature_names,
        model_file=model.model_file,
        feature_row=feature_row,
    )


def _align_feature_row_by_names(
    *,
    feature_names: tuple[str, ...] | None,
    model_file: Path,
    feature_row: np.ndarray,
) -> np.ndarray:
    if feature_names is None or feature_row.shape[1] == len(feature_names):
        return feature_row
    indices = []
    for feature_name in feature_names:
        try:
            indices.append(_FEATURE_NAME_TO_INDEX[feature_name])
        except KeyError as exc:
            raise ValueError(
                f"Feature '{feature_name}' expected by model {model_file} is not present in the canonical feature registry."
            ) from exc
    return feature_row[:, indices]


def _feature_names_require_hourly_context(feature_names: tuple[str, ...] | None) -> bool:
    if not feature_names:
        return False
    return any(feature_name in _HOURLY_CONTEXT_FEATURE_NAMES for feature_name in feature_names)


def _feature_state_has_required_context(
    feature_state: KalshiFeatureState | KalshiFeatureUpdate,
    feature_names: tuple[str, ...] | None,
) -> bool:
    if not _feature_names_require_hourly_context(feature_names):
        return True
    return bool(feature_state.kxbtcd_atm_ticker)


def score_feature_state(
    feature_state: KalshiFeatureState | KalshiFeatureUpdate,
    model: LoadedLightGBMModel,
    config: KalshiLightGBMScorerConfig | None = None,
) -> KalshiLightGBMScoreState:
    feature_row = feature_state.feature_row(model.feature_names)
    if feature_row is None or feature_state.market_prob is None or feature_state.tau_minutes is None:
        raise ValueError(f"Feature state for {feature_state.ticker} is not scoreable.")

    predicted_yes_probability = predict_yes_probability(model, feature_row, config=config)
    return KalshiLightGBMScoreState(
        ticker=feature_state.ticker,
        event_time=feature_state.event_time,
        market_prob=feature_state.market_prob,
        tau_minutes=feature_state.tau_minutes,
        predicted_yes_probability=predicted_yes_probability,
        model_edge=predicted_yes_probability - feature_state.market_prob,
        model_file=model.model_file,
        last_yes_price_cents=feature_state.last_yes_price_cents,
        last_price_cents=feature_state.last_price_cents,
        yes_bid_cents=feature_state.yes_bid_cents,
        yes_ask_cents=feature_state.yes_ask_cents,
        no_bid_cents=feature_state.no_bid_cents,
        no_ask_cents=feature_state.no_ask_cents,
        ticker_update_time=feature_state.ticker_update_time,
        trade_yes_prob=feature_state.trade_yes_prob,
        quote_mid_prob=feature_state.quote_mid_prob,
        quote_spread_cents=feature_state.quote_spread_cents,
        buy_yes_price_cents=feature_state.buy_yes_price_cents,
        buy_no_price_cents=feature_state.buy_no_price_cents,
        quote_age_seconds=feature_state.quote_age_seconds,
        last_to_mid_gap=feature_state.last_to_mid_gap,
        price_momentum=feature_state.price_momentum,
        signed_contracts_sum_300s=feature_state.signed_contracts_sum_300s,
        yes_taker_share_300s=feature_state.yes_taker_share_300s,
        received_at=feature_state.received_at,
        event_id=feature_state.event_id,
        raw_event_id=feature_state.raw_event_id,
        source=feature_state.source,
    )


def score_regularized_logistic_feature_state(
    feature_state: KalshiFeatureState | KalshiFeatureUpdate,
    model: LoadedRegularizedLogisticModel,
    config: KalshiRegularizedLogisticScorerConfig | None = None,
) -> KalshiLightGBMScoreState:
    feature_row = feature_state.feature_row(model.feature_names)
    if feature_row is None or feature_state.market_prob is None or feature_state.tau_minutes is None:
        raise ValueError(f"Feature state for {feature_state.ticker} is not scoreable.")

    predicted_yes_probability = predict_regularized_logistic_yes_probability(model, feature_row, config=config)
    return KalshiLightGBMScoreState(
        ticker=feature_state.ticker,
        event_time=feature_state.event_time,
        market_prob=feature_state.market_prob,
        tau_minutes=feature_state.tau_minutes,
        predicted_yes_probability=predicted_yes_probability,
        model_edge=predicted_yes_probability - feature_state.market_prob,
        model_file=model.model_file,
        last_yes_price_cents=feature_state.last_yes_price_cents,
        last_price_cents=feature_state.last_price_cents,
        yes_bid_cents=feature_state.yes_bid_cents,
        yes_ask_cents=feature_state.yes_ask_cents,
        no_bid_cents=feature_state.no_bid_cents,
        no_ask_cents=feature_state.no_ask_cents,
        ticker_update_time=feature_state.ticker_update_time,
        trade_yes_prob=feature_state.trade_yes_prob,
        quote_mid_prob=feature_state.quote_mid_prob,
        quote_spread_cents=feature_state.quote_spread_cents,
        buy_yes_price_cents=feature_state.buy_yes_price_cents,
        buy_no_price_cents=feature_state.buy_no_price_cents,
        quote_age_seconds=feature_state.quote_age_seconds,
        last_to_mid_gap=feature_state.last_to_mid_gap,
        price_momentum=feature_state.price_momentum,
        signed_contracts_sum_300s=feature_state.signed_contracts_sum_300s,
        yes_taker_share_300s=feature_state.yes_taker_share_300s,
        received_at=feature_state.received_at,
        event_id=feature_state.event_id,
        raw_event_id=feature_state.raw_event_id,
        source=feature_state.source,
    )


def score_linear_svm_feature_state(
    feature_state: KalshiFeatureState | KalshiFeatureUpdate,
    model: LoadedRegularizedLogisticModel,
    config: KalshiLinearSVMScorerConfig | None = None,
) -> KalshiLightGBMScoreState:
    feature_row = feature_state.feature_row(model.feature_names)
    if feature_row is None or feature_state.market_prob is None or feature_state.tau_minutes is None:
        raise ValueError(f"Feature state for {feature_state.ticker} is not scoreable.")

    predicted_yes_probability = predict_linear_svm_yes_probability(model, feature_row, config=config)
    return KalshiLightGBMScoreState(
        ticker=feature_state.ticker,
        event_time=feature_state.event_time,
        market_prob=feature_state.market_prob,
        tau_minutes=feature_state.tau_minutes,
        predicted_yes_probability=predicted_yes_probability,
        model_edge=predicted_yes_probability - feature_state.market_prob,
        model_file=model.model_file,
        last_yes_price_cents=feature_state.last_yes_price_cents,
        last_price_cents=feature_state.last_price_cents,
        yes_bid_cents=feature_state.yes_bid_cents,
        yes_ask_cents=feature_state.yes_ask_cents,
        no_bid_cents=feature_state.no_bid_cents,
        no_ask_cents=feature_state.no_ask_cents,
        ticker_update_time=feature_state.ticker_update_time,
        trade_yes_prob=feature_state.trade_yes_prob,
        quote_mid_prob=feature_state.quote_mid_prob,
        quote_spread_cents=feature_state.quote_spread_cents,
        buy_yes_price_cents=feature_state.buy_yes_price_cents,
        buy_no_price_cents=feature_state.buy_no_price_cents,
        quote_age_seconds=feature_state.quote_age_seconds,
        last_to_mid_gap=feature_state.last_to_mid_gap,
        price_momentum=feature_state.price_momentum,
        signed_contracts_sum_300s=feature_state.signed_contracts_sum_300s,
        yes_taker_share_300s=feature_state.yes_taker_share_300s,
        received_at=feature_state.received_at,
        event_id=feature_state.event_id,
        raw_event_id=feature_state.raw_event_id,
        source=feature_state.source,
    )


def score_update_from_state(state: KalshiLightGBMScoreState) -> KalshiLightGBMScoreUpdate:
    return KalshiLightGBMScoreUpdate(
        ticker=state.ticker,
        event_time=state.event_time,
        market_prob=state.market_prob,
        tau_minutes=state.tau_minutes,
        predicted_yes_probability=state.predicted_yes_probability,
        model_edge=state.model_edge,
        model_file=state.model_file,
        last_yes_price_cents=state.last_yes_price_cents,
        last_price_cents=state.last_price_cents,
        yes_bid_cents=state.yes_bid_cents,
        yes_ask_cents=state.yes_ask_cents,
        no_bid_cents=state.no_bid_cents,
        no_ask_cents=state.no_ask_cents,
        ticker_update_time=state.ticker_update_time,
        trade_yes_prob=state.trade_yes_prob,
        quote_mid_prob=state.quote_mid_prob,
        quote_spread_cents=state.quote_spread_cents,
        buy_yes_price_cents=state.buy_yes_price_cents,
        buy_no_price_cents=state.buy_no_price_cents,
        quote_age_seconds=state.quote_age_seconds,
        last_to_mid_gap=state.last_to_mid_gap,
        price_momentum=state.price_momentum,
        signed_contracts_sum_300s=state.signed_contracts_sum_300s,
        yes_taker_share_300s=state.yes_taker_share_300s,
        received_at=state.received_at,
        event_id=state.event_id,
        raw_event_id=state.raw_event_id,
        source=state.source,
    )


class KalshiLightGBMScorer:
    def __init__(
        self,
        feature_engine: KalshiFeatureStateEngine,
        config: KalshiLightGBMScorerConfig | None = None,
    ):
        self.feature_engine = feature_engine
        self.config = config or KalshiLightGBMScorerConfig()
        self._feature_queue: asyncio.Queue[KalshiFeatureUpdate] | None = None
        self._states: dict[str, KalshiLightGBMScoreState] = {}
        self._callbacks: list[Callback] = []
        self._queues: list[asyncio.Queue[KalshiLightGBMScoreUpdate]] = []
        self._task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._model: LoadedLightGBMModel | None = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return

        self._stop_event = asyncio.Event()
        self._ready_event.clear()
        if self._feature_queue is None:
            self._feature_queue = self.feature_engine.subscribe_queue()

        self._model = load_lightgbm_model_artifact(self.config)
        await self._bootstrap_from_feature_engine()
        self._task = asyncio.create_task(self._consume_loop(), name="kalshi-lightgbm-scorer")
        self._ready_event.set()

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def wait_until_ready(self) -> None:
        await self._ready_event.wait()

    def get_state(self, ticker: str) -> KalshiLightGBMScoreState | None:
        return self._states.get(ticker)

    def snapshot_states(self) -> dict[str, KalshiLightGBMScoreState]:
        return dict(self._states)

    def subscribe(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiLightGBMScoreUpdate]:
        queue: asyncio.Queue[KalshiLightGBMScoreUpdate] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue

    async def _bootstrap_from_feature_engine(self) -> None:
        for feature_state in self.feature_engine.snapshot_states().values():
            if not feature_state.is_scoreable or not _feature_state_has_required_context(feature_state, self._model.feature_names if self._model is not None else None):
                continue
            score_state = self._score(feature_state)
            self._states[score_state.ticker] = score_state
            await self._publish_update(score_update_from_state(score_state))

    async def _consume_loop(self) -> None:
        if self._feature_queue is None:
            return

        try:
            while not self._stop_event.is_set():
                update = await self._feature_queue.get()
                await self._handle_feature_update(update)
        except asyncio.CancelledError:
            raise

    async def _handle_feature_update(self, update: KalshiFeatureUpdate) -> None:
        if not update.is_scoreable or not _feature_state_has_required_context(update, self._model.feature_names if self._model is not None else None):
            return

        score_state = self._score(update)
        self._states[score_state.ticker] = score_state
        await self._publish_update(score_update_from_state(score_state))

    def _score(self, feature_state: KalshiFeatureState | KalshiFeatureUpdate) -> KalshiLightGBMScoreState:
        if self._model is None:
            raise RuntimeError("Scorer model is not loaded.")
        return score_feature_state(feature_state, model=self._model, config=self.config)

    async def _publish_update(self, update: KalshiLightGBMScoreUpdate) -> None:
        for callback in self._callbacks:
            result = callback(update)
            if inspect.isawaitable(result):
                await result
        for queue in self._queues:
            await queue.put(update)


class KalshiRegularizedLogisticScorer:
    def __init__(
        self,
        feature_engine: KalshiFeatureStateEngine,
        config: KalshiRegularizedLogisticScorerConfig | None = None,
    ):
        self.feature_engine = feature_engine
        self.config = config or KalshiRegularizedLogisticScorerConfig()
        self._feature_queue: asyncio.Queue[KalshiFeatureUpdate] | None = None
        self._states: dict[str, KalshiLightGBMScoreState] = {}
        self._callbacks: list[Callback] = []
        self._queues: list[asyncio.Queue[KalshiLightGBMScoreUpdate]] = []
        self._task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._model: LoadedRegularizedLogisticModel | None = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return

        self._stop_event = asyncio.Event()
        self._ready_event.clear()
        if self._feature_queue is None:
            self._feature_queue = self.feature_engine.subscribe_queue()

        self._model = load_regularized_logistic_model_artifact(self.config)
        await self._bootstrap_from_feature_engine()
        self._task = asyncio.create_task(self._consume_loop(), name="kalshi-regularized-logistic-scorer")
        self._ready_event.set()

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def wait_until_ready(self) -> None:
        await self._ready_event.wait()

    def get_state(self, ticker: str) -> KalshiLightGBMScoreState | None:
        return self._states.get(ticker)

    def snapshot_states(self) -> dict[str, KalshiLightGBMScoreState]:
        return dict(self._states)

    def subscribe(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiLightGBMScoreUpdate]:
        queue: asyncio.Queue[KalshiLightGBMScoreUpdate] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue

    async def _bootstrap_from_feature_engine(self) -> None:
        for feature_state in self.feature_engine.snapshot_states().values():
            if not feature_state.is_scoreable or not _feature_state_has_required_context(feature_state, self._model.feature_names if self._model is not None else None):
                continue
            score_state = self._score(feature_state)
            self._states[score_state.ticker] = score_state
            await self._publish_update(score_update_from_state(score_state))

    async def _consume_loop(self) -> None:
        if self._feature_queue is None:
            return

        try:
            while not self._stop_event.is_set():
                update = await self._feature_queue.get()
                await self._handle_feature_update(update)
        except asyncio.CancelledError:
            raise

    async def _handle_feature_update(self, update: KalshiFeatureUpdate) -> None:
        if not update.is_scoreable or not _feature_state_has_required_context(update, self._model.feature_names if self._model is not None else None):
            return

        score_state = self._score(update)
        self._states[score_state.ticker] = score_state
        await self._publish_update(score_update_from_state(score_state))

    def _score(self, feature_state: KalshiFeatureState | KalshiFeatureUpdate) -> KalshiLightGBMScoreState:
        if self._model is None:
            raise RuntimeError("Scorer model is not loaded.")
        return score_regularized_logistic_feature_state(feature_state, model=self._model, config=self.config)

    async def _publish_update(self, update: KalshiLightGBMScoreUpdate) -> None:
        for callback in self._callbacks:
            result = callback(update)
            if inspect.isawaitable(result):
                await result
        for queue in self._queues:
            await queue.put(update)


class KalshiLinearSVMScorer:
    def __init__(
        self,
        feature_engine: KalshiFeatureStateEngine,
        config: KalshiLinearSVMScorerConfig | None = None,
    ):
        self.feature_engine = feature_engine
        self.config = config or KalshiLinearSVMScorerConfig()
        self._feature_queue: asyncio.Queue[KalshiFeatureUpdate] | None = None
        self._states: dict[str, KalshiLightGBMScoreState] = {}
        self._callbacks: list[Callback] = []
        self._queues: list[asyncio.Queue[KalshiLightGBMScoreUpdate]] = []
        self._task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._model: LoadedRegularizedLogisticModel | None = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return

        self._stop_event = asyncio.Event()
        self._ready_event.clear()
        if self._feature_queue is None:
            self._feature_queue = self.feature_engine.subscribe_queue()

        self._model = load_linear_svm_model_artifact(self.config)
        await self._bootstrap_from_feature_engine()
        self._task = asyncio.create_task(self._consume_loop(), name="kalshi-linear-svm-scorer")
        self._ready_event.set()

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def wait_until_ready(self) -> None:
        await self._ready_event.wait()

    def get_state(self, ticker: str) -> KalshiLightGBMScoreState | None:
        return self._states.get(ticker)

    def snapshot_states(self) -> dict[str, KalshiLightGBMScoreState]:
        return dict(self._states)

    def subscribe(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiLightGBMScoreUpdate]:
        queue: asyncio.Queue[KalshiLightGBMScoreUpdate] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue

    async def _bootstrap_from_feature_engine(self) -> None:
        for feature_state in self.feature_engine.snapshot_states().values():
            if not feature_state.is_scoreable or not _feature_state_has_required_context(feature_state, self._model.feature_names if self._model is not None else None):
                continue
            score_state = self._score(feature_state)
            self._states[score_state.ticker] = score_state
            await self._publish_update(score_update_from_state(score_state))

    async def _consume_loop(self) -> None:
        if self._feature_queue is None:
            return

        try:
            while not self._stop_event.is_set():
                update = await self._feature_queue.get()
                await self._handle_feature_update(update)
        except asyncio.CancelledError:
            raise

    async def _handle_feature_update(self, update: KalshiFeatureUpdate) -> None:
        if not update.is_scoreable or not _feature_state_has_required_context(update, self._model.feature_names if self._model is not None else None):
            return

        score_state = self._score(update)
        self._states[score_state.ticker] = score_state
        await self._publish_update(score_update_from_state(score_state))

    def _score(self, feature_state: KalshiFeatureState | KalshiFeatureUpdate) -> KalshiLightGBMScoreState:
        if self._model is None:
            raise RuntimeError("Scorer model is not loaded.")
        return score_linear_svm_feature_state(feature_state, model=self._model, config=self.config)

    async def _publish_update(self, update: KalshiLightGBMScoreUpdate) -> None:
        for callback in self._callbacks:
            result = callback(update)
            if inspect.isawaitable(result):
                await result
        for queue in self._queues:
            await queue.put(update)
