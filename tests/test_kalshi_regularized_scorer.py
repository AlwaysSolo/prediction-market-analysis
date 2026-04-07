from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from joblib import dump as joblib_dump
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.run_kalshi_regularized_execution_engine import default_model_file_for_family, signal_config_from_env_and_policy
from src.live.kalshi import (
    DEFAULT_BAGGED_LASSO_MODEL_FILE,
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiFeatureEngineConfig,
    KalshiFeatureStateEngine,
    KalshiLinearSVMScorer,
    KalshiMarketDataCollector,
    KalshiRegularizedLogisticScorer,
)
from src.live.kalshi.calibration import PlattCalibration, save_platt_calibration
from src.live.kalshi.features import build_feature_row
from src.live.kalshi.scorer import (
    KalshiLinearSVMScorerConfig,
    KalshiRegularizedLogisticScorerConfig,
    LoadedRegularizedLogisticModel,
    load_linear_svm_model_artifact,
    load_regularized_logistic_model_artifact,
    predict_linear_svm_yes_probability,
    predict_regularized_logistic_yes_probability,
)
from src.live.kalshi.types import KalshiTickerUpdate


def _collector_config(tmp_path: Path) -> KalshiCollectorConfig:
    return KalshiCollectorConfig(
        environment=KalshiEnvironment.DEMO,
        credentials=KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
        log_dir=tmp_path / "logs",
        series_tickers=("KXBTC15M",),
        metadata_refresh_interval_seconds=3600.0,
    )


def _ticker_update(
    *,
    ticker: str = "KXBTC15M-TEST",
    event_time: datetime | None = None,
    last_yes_price_cents: int = 55,
    previous_yes_price_cents: int | None = 54,
    close_time: datetime | None = None,
    is_open: bool = True,
) -> KalshiTickerUpdate:
    event_time = event_time or datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    if close_time is None:
        close_time = event_time + timedelta(minutes=7)
    return KalshiTickerUpdate(
        ticker=ticker,
        event_time=event_time,
        last_yes_price_cents=last_yes_price_cents,
        previous_yes_price_cents=previous_yes_price_cents,
        close_time=close_time,
        is_open=is_open,
        market_prob=None,
        previous_market_prob=None,
        price_momentum=None,
        tau_minutes=None,
    )


class _TrackingScaler:
    def __init__(self):
        self.feature_shapes: list[tuple[int, ...]] = []
        self.feature_rows: list[np.ndarray] = []

    def transform(self, feature_row: np.ndarray) -> np.ndarray:
        self.feature_shapes.append(tuple(feature_row.shape))
        self.feature_rows.append(np.asarray(feature_row, dtype=np.float64).copy())
        return feature_row


class _PredictingClassifier:
    def __init__(self, prediction: float):
        self.prediction = prediction
        self.feature_shapes: list[tuple[int, ...]] = []
        self.feature_rows: list[np.ndarray] = []

    def predict_proba(self, feature_row: np.ndarray) -> np.ndarray:
        self.feature_shapes.append(tuple(feature_row.shape))
        self.feature_rows.append(np.asarray(feature_row, dtype=np.float64).copy())
        return np.array([[1.0 - self.prediction, self.prediction]], dtype=np.float64)


class _DecisionFunctionClassifier:
    def __init__(self, score: float):
        self.score = score
        self.feature_shapes: list[tuple[int, ...]] = []

    def decision_function(self, feature_row: np.ndarray) -> np.ndarray:
        self.feature_shapes.append(tuple(feature_row.shape))
        return np.array([self.score], dtype=np.float64)


def test_load_regularized_logistic_model_artifact_reads_feature_names_and_calibration(tmp_path: Path):
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.8],
            [0.8, 0.2],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 1, 0, 1], dtype=np.int8)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(solver="lbfgs", random_state=42).fit(scaler.transform(X), y)

    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    calibration_path = tmp_path / "calibration.json"
    joblib_dump({"scaler": scaler, "model": model}, model_path)
    metrics_path.write_text(json.dumps({"feature_names": ["z_implied", "tau_minutes"]}), encoding="utf-8")
    save_platt_calibration(calibration_path, PlattCalibration(a=1.0, b=0.0))

    artifact = load_regularized_logistic_model_artifact(
        KalshiRegularizedLogisticScorerConfig(
            model_file=model_path,
            metrics_file=metrics_path,
            calibration_file=calibration_path,
        )
    )

    assert artifact.model_file == model_path
    assert artifact.feature_names == ("z_implied", "tau_minutes")
    assert artifact.calibration is not None


def test_predict_regularized_logistic_yes_probability_aligns_feature_row():
    feature_row = build_feature_row(
        market_prob=0.55,
        tau_minutes=7.0,
        previous_market_prob=0.54,
        last_trade_count=3.0,
        last_trade_side_sign=1.0,
        last_trade_signed_count=3.0,
        trade_count_30s=2.0,
    )
    scaler = _TrackingScaler()
    model = _PredictingClassifier(0.63)

    probability = predict_regularized_logistic_yes_probability(
        LoadedRegularizedLogisticModel(
            scaler=scaler,
            model=model,
            model_file=Path("dummy.joblib"),
            feature_names=(
                "z_implied",
                "tau_minutes",
                "price_momentum",
                "abs_price_momentum",
                "price_direction",
                "distance_from_mid",
            ),
        ),
        feature_row,
    )

    assert probability == pytest.approx(0.63)
    assert scaler.feature_shapes == [(1, 6)]
    assert model.feature_shapes == [(1, 6)]


def test_load_linear_svm_model_artifact_reads_feature_names_and_calibration(tmp_path: Path):
    scaler = StandardScaler().fit(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64))
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    calibration_path = tmp_path / "calibration.json"
    joblib_dump({"scaler": scaler, "model": _DecisionFunctionClassifier(0.5)}, model_path)
    metrics_path.write_text(json.dumps({"feature_names": ["z_implied", "tau_minutes"]}), encoding="utf-8")
    save_platt_calibration(calibration_path, PlattCalibration(a=1.0, b=0.0))

    artifact = load_linear_svm_model_artifact(
        KalshiLinearSVMScorerConfig(
            model_file=model_path,
            metrics_file=metrics_path,
            calibration_file=calibration_path,
        )
    )

    assert artifact.model_file == model_path
    assert artifact.feature_names == ("z_implied", "tau_minutes")
    assert artifact.calibration is not None


def test_predict_linear_svm_yes_probability_aligns_feature_row():
    feature_row = build_feature_row(
        market_prob=0.55,
        tau_minutes=7.0,
        previous_market_prob=0.54,
        last_trade_count=3.0,
        last_trade_side_sign=1.0,
        last_trade_signed_count=3.0,
        trade_count_30s=2.0,
    )
    scaler = _TrackingScaler()
    model = _DecisionFunctionClassifier(0.7)

    probability = predict_linear_svm_yes_probability(
        LoadedRegularizedLogisticModel(
            scaler=scaler,
            model=model,
            model_file=Path("dummy-svm.joblib"),
            feature_names=(
                "z_implied",
                "tau_minutes",
                "price_momentum",
                "abs_price_momentum",
                "price_direction",
                "distance_from_mid",
            ),
        ),
        feature_row,
    )

    assert probability == pytest.approx(1.0 / (1.0 + np.exp(-0.7)))
    assert scaler.feature_shapes == [(1, 6)]
    assert model.feature_shapes == [(1, 6)]


def test_regularized_logistic_scorer_consumes_feature_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    engine = KalshiFeatureStateEngine(collector)
    fake_model = LoadedRegularizedLogisticModel(
        scaler=_TrackingScaler(),
        model=_PredictingClassifier(0.62),
        model_file=Path("fake-model.joblib"),
    )
    monkeypatch.setattr("src.live.kalshi.scorer.load_regularized_logistic_model_artifact", lambda _config: fake_model)
    scorer = KalshiRegularizedLogisticScorer(engine)
    queue = scorer.subscribe_queue()

    async def run() -> None:
        await engine.start()
        await scorer.start()
        await collector._publish_update(_ticker_update())
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.ticker == "KXBTC15M-TEST"
        assert update.market_prob == pytest.approx(0.55)
        assert update.tau_minutes == pytest.approx(7.0)
        assert update.predicted_yes_probability == pytest.approx(0.62)
        assert update.model_edge == pytest.approx(0.07)
        await scorer.stop()
        await engine.stop()

    asyncio.run(run())


def test_regularized_logistic_scorer_supports_hourly_context_features_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(
        KalshiCollectorConfig(
            environment=KalshiEnvironment.DEMO,
            credentials=KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
            log_dir=tmp_path / "logs",
            series_tickers=("KXBTC15M", "KXBTCD"),
            metadata_refresh_interval_seconds=3600.0,
        )
    )
    engine = KalshiFeatureStateEngine(
        collector,
        config=KalshiFeatureEngineConfig(
            publish_series_tickers=("KXBTC15M",),
            hourly_context_target_series_ticker="KXBTC15M",
            hourly_context_series_ticker="KXBTCD",
        ),
    )
    tracking_scaler = _TrackingScaler()
    predicting_model = _PredictingClassifier(0.64)
    fake_model = LoadedRegularizedLogisticModel(
        scaler=tracking_scaler,
        model=predicting_model,
        model_file=Path("fake-hourly-context-model.joblib"),
        feature_names=("z_implied", "kxbtcd_atm_z_implied", "k15_minus_k1h_atm_z", "k15_k1h_z_product"),
    )
    monkeypatch.setattr("src.live.kalshi.scorer.load_regularized_logistic_model_artifact", lambda _config: fake_model)
    scorer = KalshiRegularizedLogisticScorer(engine)
    queue = scorer.subscribe_queue()

    async def run() -> None:
        await engine.start()
        await scorer.start()
        context_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        target_time = context_time + timedelta(seconds=20)

        await collector._publish_update(
            KalshiTickerUpdate(
                ticker="KXBTCD-CTX-T95000.00",
                event_time=context_time,
                last_yes_price_cents=52,
                previous_yes_price_cents=51,
                close_time=datetime(2026, 1, 1, 13, 0, tzinfo=UTC),
                is_open=True,
                market_prob=None,
                previous_market_prob=None,
                price_momentum=None,
                tau_minutes=None,
                open_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                trade_id="ctx-1",
                count=5,
                taker_side="yes",
                last_trade_time=context_time,
                last_price_cents=52,
                source="trade",
            )
        )
        await collector._publish_update(
            KalshiTickerUpdate(
                ticker="KXBTC15M-TEST",
                event_time=target_time,
                last_yes_price_cents=55,
                previous_yes_price_cents=54,
                close_time=target_time + timedelta(minutes=7),
                is_open=True,
                market_prob=None,
                previous_market_prob=None,
                price_momentum=None,
                tau_minutes=None,
                trade_id="target-1",
                count=2,
                taker_side="yes",
                last_trade_time=target_time,
                last_price_cents=55,
                source="trade",
            )
        )
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.ticker == "KXBTC15M-TEST"
        assert update.predicted_yes_probability == pytest.approx(0.64)
        assert tracking_scaler.feature_shapes == [(1, 4)]
        assert predicting_model.feature_shapes == [(1, 4)]
        assert tracking_scaler.feature_rows[0][0, 1] != 0.0
        assert tracking_scaler.feature_rows[0][0, 3] != 0.0
        await scorer.stop()
        await engine.stop()

    asyncio.run(run())


def test_regularized_logistic_hourly_context_model_waits_for_context_before_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    collector = KalshiMarketDataCollector(
        KalshiCollectorConfig(
            environment=KalshiEnvironment.DEMO,
            credentials=KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
            log_dir=tmp_path / "logs",
            series_tickers=("KXBTC15M", "KXBTCD"),
            metadata_refresh_interval_seconds=3600.0,
        )
    )
    engine = KalshiFeatureStateEngine(
        collector,
        config=KalshiFeatureEngineConfig(
            publish_series_tickers=("KXBTC15M",),
            hourly_context_target_series_ticker="KXBTC15M",
            hourly_context_series_ticker="KXBTCD",
        ),
    )
    tracking_scaler = _TrackingScaler()
    predicting_model = _PredictingClassifier(0.64)
    fake_model = LoadedRegularizedLogisticModel(
        scaler=tracking_scaler,
        model=predicting_model,
        model_file=Path("fake-hourly-context-model.joblib"),
        feature_names=("z_implied", "kxbtcd_atm_z_implied", "k15_minus_k1h_atm_z", "k15_k1h_z_product"),
    )
    monkeypatch.setattr("src.live.kalshi.scorer.load_regularized_logistic_model_artifact", lambda _config: fake_model)
    scorer = KalshiRegularizedLogisticScorer(engine)
    queue = scorer.subscribe_queue()

    async def run() -> None:
        await engine.start()
        await scorer.start()
        target_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        await collector._publish_update(
            KalshiTickerUpdate(
                ticker="KXBTC15M-TEST",
                event_time=target_time,
                last_yes_price_cents=55,
                previous_yes_price_cents=54,
                close_time=target_time + timedelta(minutes=7),
                is_open=True,
                market_prob=None,
                previous_market_prob=None,
                price_momentum=None,
                tau_minutes=None,
                trade_id="target-1",
                count=2,
                taker_side="yes",
                last_trade_time=target_time,
                last_price_cents=55,
                source="trade",
                open_time=target_time - timedelta(minutes=1),
            )
        )

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.get(), timeout=0.1)

        context_time = target_time + timedelta(seconds=10)
        await collector._publish_update(
            KalshiTickerUpdate(
                ticker="KXBTCD-CTX-T95000.00",
                event_time=context_time,
                last_yes_price_cents=52,
                previous_yes_price_cents=51,
                close_time=target_time + timedelta(hours=1),
                is_open=True,
                market_prob=None,
                previous_market_prob=None,
                price_momentum=None,
                tau_minutes=None,
                open_time=target_time,
                trade_id="ctx-1",
                count=5,
                taker_side="yes",
                last_trade_time=context_time,
                last_price_cents=52,
                source="trade",
            )
        )

        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.ticker == "KXBTC15M-TEST"
        assert update.predicted_yes_probability == pytest.approx(0.64)
        assert tracking_scaler.feature_shapes == [(1, 4)]
        assert tracking_scaler.feature_rows[0][0, 1] != 0.0
        assert tracking_scaler.feature_rows[0][0, 3] != 0.0

        await scorer.stop()
        await engine.stop()

    asyncio.run(run())


def test_linear_svm_scorer_consumes_feature_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    engine = KalshiFeatureStateEngine(collector)
    fake_model = LoadedRegularizedLogisticModel(
        scaler=_TrackingScaler(),
        model=_DecisionFunctionClassifier(0.5),
        model_file=Path("fake-linear-svm.joblib"),
    )
    monkeypatch.setattr("src.live.kalshi.scorer.load_linear_svm_model_artifact", lambda _config: fake_model)
    scorer = KalshiLinearSVMScorer(engine)
    queue = scorer.subscribe_queue()

    async def run() -> None:
        await engine.start()
        await scorer.start()
        await collector._publish_update(_ticker_update())
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.ticker == "KXBTC15M-TEST"
        assert update.market_prob == pytest.approx(0.55)
        assert update.tau_minutes == pytest.approx(7.0)
        assert update.predicted_yes_probability == pytest.approx(1.0 / (1.0 + np.exp(-0.5)))
        assert update.model_edge == pytest.approx(update.predicted_yes_probability - 0.55)
        await scorer.stop()
        await engine.stop()

    asyncio.run(run())


def test_signal_config_from_env_and_policy_overrides_env_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_EDGE_THRESHOLD_CENTS", "9.0")
    monkeypatch.setenv("KALSHI_DEMO_SIGNAL_ALLOW_STACKING", "true")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "config": {
                    "edge_threshold_cents": 4.5,
                    "min_tau_minutes": 0.0,
                    "max_tau_minutes": 12.0,
                    "price_band_min_cents": 10,
                    "price_band_max_cents": 90,
                    "reserve_cash_pct": 30.0,
                    "starting_cash_dollars": 10000.0,
                    "contracts_per_order": 1,
                    "slippage_pct": 1.0,
                    "allow_stacking": False,
                    "capital_pct_per_order": None,
                }
            }
        ),
        encoding="utf-8",
    )

    config, loaded_policy = signal_config_from_env_and_policy(KalshiEnvironment.DEMO, policy_path)

    assert loaded_policy == policy_path
    assert config.edge_threshold_cents == pytest.approx(4.5)
    assert config.allow_stacking is False
    assert config.price_band_min_cents == 10
    assert config.price_band_max_cents == 90


def test_default_model_file_for_family_supports_bagged_lasso():
    assert default_model_file_for_family("bagged_lasso") == DEFAULT_BAGGED_LASSO_MODEL_FILE
