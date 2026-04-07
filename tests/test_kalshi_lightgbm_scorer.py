from __future__ import annotations

import asyncio
import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from lightgbm.basic import LightGBMError

from src.live.kalshi import (
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiFeatureStateEngine,
    KalshiLightGBMScorer,
    KalshiLightGBMScorerConfig,
    KalshiMarketDataCollector,
)
from src.live.kalshi.features import build_feature_row
from src.live.kalshi.scorer import (
    DEFAULT_LIGHTGBM_MODEL_FILE,
    LoadedLightGBMModel,
    load_lightgbm_model_artifact,
    predict_yes_probability,
)
from src.live.kalshi.types import KalshiTickerState, KalshiTickerUpdate


def _collector_config(tmp_path: Path) -> KalshiCollectorConfig:
    return KalshiCollectorConfig(
        environment=KalshiEnvironment.DEMO,
        credentials=KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
        log_dir=tmp_path / "logs",
        series_tickers=("KXBTC15M",),
        metadata_refresh_interval_seconds=3600.0,
    )


def _load_score_helper():
    helper_path = Path("Gemini scripts") / "load_lightgbm_and_score.py"
    spec = importlib.util.spec_from_file_location("load_lightgbm_and_score", helper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


class _PredictingBooster:
    def __init__(self, prediction: float):
        self.prediction = prediction
        self.calls: list[int | None] = []
        self.feature_shapes: list[tuple[int, ...]] = []

    def predict(self, feature_row: np.ndarray, num_iteration: int | None = None) -> np.ndarray:
        self.calls.append(num_iteration)
        self.feature_shapes.append(tuple(feature_row.shape))
        return np.array([self.prediction], dtype=np.float64)


def test_load_model_artifact_reads_default_model_and_best_iteration():
    artifact = load_lightgbm_model_artifact()

    assert artifact.model_file == DEFAULT_LIGHTGBM_MODEL_FILE
    assert artifact.best_iteration is not None
    assert artifact.best_iteration > 0


def test_load_model_artifact_missing_model_fails_fast(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_lightgbm_model_artifact(
            KalshiLightGBMScorerConfig(model_file=tmp_path / "missing-model.txt")
        )


def test_load_model_artifact_invalid_model_fails_fast(tmp_path: Path):
    invalid_model = tmp_path / "invalid-model.txt"
    invalid_model.write_text("not a lightgbm model", encoding="utf-8")

    with pytest.raises(LightGBMError):
        load_lightgbm_model_artifact(
            KalshiLightGBMScorerConfig(model_file=invalid_model, metrics_file=None)
        )


def test_predict_yes_probability_clips_output():
    feature_row = build_feature_row(market_prob=0.55, tau_minutes=7.0, previous_market_prob=0.54)

    high_prediction = predict_yes_probability(
        LoadedLightGBMModel(
            booster=_PredictingBooster(1.5),  # type: ignore[arg-type]
            model_file=Path("dummy.txt"),
            best_iteration=None,
        ),
        feature_row,
    )
    low_prediction = predict_yes_probability(
        LoadedLightGBMModel(
            booster=_PredictingBooster(-0.5),  # type: ignore[arg-type]
            model_file=Path("dummy.txt"),
            best_iteration=None,
        ),
        feature_row,
    )

    assert high_prediction == pytest.approx(1.0 - 1e-6)
    assert low_prediction == pytest.approx(1e-6)


def test_predict_yes_probability_uses_best_iteration():
    feature_row = build_feature_row(market_prob=0.55, tau_minutes=7.0, previous_market_prob=0.54)
    booster = _PredictingBooster(0.42)

    probability = predict_yes_probability(
        LoadedLightGBMModel(
            booster=booster,  # type: ignore[arg-type]
            model_file=Path("dummy.txt"),
            best_iteration=42,
        ),
        feature_row,
    )

    assert probability == pytest.approx(0.42)
    assert booster.calls == [42]


def test_load_model_artifact_reads_best_iteration_from_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_path = tmp_path / "model.txt"
    metrics_path = tmp_path / "metrics.json"
    model_path.write_text("placeholder model contents", encoding="utf-8")
    metrics_path.write_text(json.dumps({"best_iteration": 77}), encoding="utf-8")

    class FakeBooster:
        def __init__(self, model_file: str):
            self.model_file = model_file

        def predict(self, _feature_row: np.ndarray, num_iteration: int | None = None) -> np.ndarray:
            return np.array([0.5], dtype=np.float64)

    monkeypatch.setattr("src.live.kalshi.scorer.Booster", FakeBooster)
    artifact = load_lightgbm_model_artifact(
        KalshiLightGBMScorerConfig(model_file=model_path, metrics_file=metrics_path)
    )

    assert artifact.best_iteration == 77
    assert artifact.model_file == model_path
    assert artifact.booster.model_file == str(model_path)


def test_predict_yes_probability_applies_platt_calibration():
    feature_row = build_feature_row(market_prob=0.55, tau_minutes=7.0, previous_market_prob=0.54)
    probability = predict_yes_probability(
        LoadedLightGBMModel(
            booster=_PredictingBooster(0.60),  # type: ignore[arg-type]
            model_file=Path("dummy.txt"),
            best_iteration=None,
            calibration=type("Calibration", (), {"apply": staticmethod(lambda value: 0.75)})(),
        ),
        feature_row,
        config=KalshiLightGBMScorerConfig(apply_calibration=True),
    )

    assert probability == pytest.approx(0.75)


def test_predict_yes_probability_aligns_feature_row_for_legacy_model():
    booster = _PredictingBooster(0.60)
    feature_row = build_feature_row(
        market_prob=0.55,
        tau_minutes=7.0,
        previous_market_prob=0.54,
        last_trade_count=3.0,
        last_trade_side_sign=1.0,
        last_trade_signed_count=3.0,
        trade_count_30s=2.0,
    )

    probability = predict_yes_probability(
        LoadedLightGBMModel(
            booster=booster,  # type: ignore[arg-type]
            model_file=Path("dummy.txt"),
            best_iteration=None,
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

    assert probability == pytest.approx(0.60)
    assert booster.feature_shapes == [(1, 6)]


def test_scorer_matches_standalone_helper():
    helper = _load_score_helper()
    config = KalshiLightGBMScorerConfig()
    artifact = load_lightgbm_model_artifact(config)
    feature_row = build_feature_row(market_prob=0.55, tau_minutes=7.0, previous_market_prob=0.54)

    scorer_probability = predict_yes_probability(artifact, feature_row, config=config)
    helper_probability = helper.score_yes_probability(
        model_file=config.model_file,
        market_prob=0.55,
        tau_minutes=7.0,
        previous_market_prob=0.54,
    )

    assert scorer_probability == pytest.approx(helper_probability)


def test_lightgbm_scorer_bootstraps_existing_feature_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._states[ticker] = KalshiTickerState(
        ticker=ticker,
        last_yes_price_cents=55,
        last_trade_time=datetime(2026, 1, 1, 11, 58, tzinfo=UTC),
        previous_yes_price_cents=54,
        close_time=datetime.now(UTC) + timedelta(minutes=10),
        is_open=True,
    )

    fake_model = LoadedLightGBMModel(
        booster=_PredictingBooster(0.61),  # type: ignore[arg-type]
        model_file=Path("fake-model.txt"),
        best_iteration=None,
    )
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: fake_model)

    engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiLightGBMScorer(engine)
    queue = scorer.subscribe_queue()

    async def run() -> None:
        await engine.start()
        await scorer.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.ticker == ticker
        assert update.predicted_yes_probability == pytest.approx(0.61)
        assert update.model_edge == pytest.approx(0.06)
        await scorer.stop()
        await engine.stop()

    asyncio.run(run())


def test_lightgbm_scorer_consumes_feature_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    engine = KalshiFeatureStateEngine(collector)
    fake_model = LoadedLightGBMModel(
        booster=_PredictingBooster(0.60),  # type: ignore[arg-type]
        model_file=Path("fake-model.txt"),
        best_iteration=None,
    )
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: fake_model)
    scorer = KalshiLightGBMScorer(engine)
    queue = scorer.subscribe_queue()

    async def run() -> None:
        await engine.start()
        await scorer.start()
        await collector._publish_update(_ticker_update())
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert update.ticker == "KXBTC15M-TEST"
        assert update.market_prob == pytest.approx(0.55)
        assert update.tau_minutes == pytest.approx(7.0)
        assert update.predicted_yes_probability == pytest.approx(0.60)
        assert update.model_edge == pytest.approx(0.05)
        assert scorer.get_state(update.ticker) is not None
        await scorer.stop()
        await engine.stop()

    asyncio.run(run())


def test_lightgbm_scorer_ignores_non_scoreable_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    engine = KalshiFeatureStateEngine(collector)
    fake_model = LoadedLightGBMModel(
        booster=_PredictingBooster(0.60),  # type: ignore[arg-type]
        model_file=Path("fake-model.txt"),
        best_iteration=None,
    )
    monkeypatch.setattr("src.live.kalshi.scorer.load_lightgbm_model_artifact", lambda _config: fake_model)
    scorer = KalshiLightGBMScorer(engine)
    queue = scorer.subscribe_queue()

    async def run() -> None:
        await engine.start()
        await scorer.start()
        await collector._publish_update(
            _ticker_update(
                close_time=datetime(2026, 1, 1, 12, 16, tzinfo=UTC),
            )
        )
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.get(), timeout=0.1)
        await scorer.stop()
        await engine.stop()

    asyncio.run(run())
