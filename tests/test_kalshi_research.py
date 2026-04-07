from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_kalshi_research_multi_model import _run as run_research_multi_model
from src.indexers.kalshi.models import Market
from src.live.kalshi import (
    FEATURE_ORDER,
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiMarketDataCollector,
    KalshiResearchLedger,
    KalshiResearchSampler,
    KalshiResearchSamplerConfig,
    KalshiTickerState,
)
from src.live.kalshi.scorer import KalshiLightGBMScoreState, LoadedRegularizedLogisticModel


def _collector_config(tmp_path: Path) -> KalshiCollectorConfig:
    return KalshiCollectorConfig(
        environment=KalshiEnvironment.DEMO,
        credentials=KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
        log_dir=tmp_path / "logs",
        series_tickers=("KXBTC15M",),
        metadata_refresh_interval_seconds=3600.0,
    )


def _market(*, ticker: str, result: str = "", status: str = "open") -> Market:
    return Market(
        ticker=ticker,
        event_ticker="KXBTC15M",
        market_type="binary",
        title="t",
        yes_sub_title="y",
        no_sub_title="n",
        status=status,
        yes_bid=None,
        yes_ask=None,
        no_bid=None,
        no_ask=None,
        last_price=None,
        volume=0,
        volume_24h=0,
        open_interest=0,
        result=result,
        created_time=None,
        open_time=None,
        close_time=datetime.now(UTC) + timedelta(minutes=10),
    )


def _score_state(
    *,
    ticker: str = "KXBTC15M-TEST",
    predicted_yes_probability: float = 0.75,
    market_prob: float = 0.55,
    tau_minutes: float = 7.0,
    yes_bid_cents: int = 54,
    yes_ask_cents: int = 56,
    event_time: datetime | None = None,
) -> KalshiLightGBMScoreState:
    event_time = event_time or datetime.now(UTC)
    return KalshiLightGBMScoreState(
        ticker=ticker,
        event_time=event_time,
        market_prob=market_prob,
        tau_minutes=tau_minutes,
        predicted_yes_probability=predicted_yes_probability,
        model_edge=predicted_yes_probability - market_prob,
        model_file=Path("fake-model.joblib"),
        last_yes_price_cents=int(round(market_prob * 100)),
        last_price_cents=int(round(market_prob * 100)),
        yes_bid_cents=yes_bid_cents,
        yes_ask_cents=yes_ask_cents,
        ticker_update_time=event_time,
        trade_yes_prob=market_prob,
        quote_mid_prob=(yes_bid_cents + yes_ask_cents) / 200.0,
        quote_spread_cents=yes_ask_cents - yes_bid_cents,
        buy_yes_price_cents=yes_ask_cents,
        buy_no_price_cents=100 - yes_bid_cents,
        quote_age_seconds=0.0,
        last_to_mid_gap=market_prob - ((yes_bid_cents + yes_ask_cents) / 200.0),
    )


class _FakeScorer:
    def __init__(self, collector: KalshiMarketDataCollector, snapshot_states: dict[str, KalshiLightGBMScoreState]):
        self.feature_engine = SimpleNamespace(collector=collector)
        self._states = dict(snapshot_states)
        self._queues: list[asyncio.Queue[KalshiLightGBMScoreState]] = []

    def snapshot_states(self) -> dict[str, KalshiLightGBMScoreState]:
        return dict(self._states)

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiLightGBMScoreState]:
        queue: asyncio.Queue[KalshiLightGBMScoreState] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue

    async def push(self, state: KalshiLightGBMScoreState) -> None:
        for queue in self._queues:
            await queue.put(state)


class _IdentityScaler:
    def transform(self, value: np.ndarray) -> np.ndarray:
        return value


class _PredictingModel:
    def __init__(self, prediction: float):
        self.prediction = prediction

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.array([[1.0 - self.prediction, self.prediction]], dtype=np.float64)


def _fake_regularized_model(prediction: float) -> LoadedRegularizedLogisticModel:
    return LoadedRegularizedLogisticModel(
        scaler=_IdentityScaler(),
        model=_PredictingModel(prediction),
        model_file=Path("fake-model.joblib"),
        calibration=None,
        feature_names=FEATURE_ORDER,
    )


def test_research_sampler_picks_yes_when_yes_edge_is_larger(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    scorer = _FakeScorer(collector, {"KXBTC15M-TEST": _score_state(predicted_yes_probability=0.78)})
    sampler = KalshiResearchSampler(scorer, log_dir=tmp_path / "research")
    queue = sampler.subscribe_queue()

    async def run() -> None:
        await sampler.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        await sampler.stop()

        assert update.status == "recorded"
        assert update.sample is not None
        assert update.sample.side == "YES"
        assert update.sample.chosen_side_edge_bucket in {"10-20", "20-40", "40-60", "60+"}

    asyncio.run(run())


def test_research_sampler_picks_no_when_no_edge_is_larger(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    scorer = _FakeScorer(collector, {"KXBTC15M-TEST": _score_state(predicted_yes_probability=0.22)})
    sampler = KalshiResearchSampler(scorer, log_dir=tmp_path / "research")
    queue = sampler.subscribe_queue()

    async def run() -> None:
        await sampler.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        await sampler.stop()

        assert update.status == "recorded"
        assert update.sample is not None
        assert update.sample.side == "NO"

    asyncio.run(run())

def test_research_sampler_skips_when_both_sides_fail_filters(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    state = _score_state(predicted_yes_probability=0.52, yes_bid_cents=49, yes_ask_cents=51)
    scorer = _FakeScorer(collector, {"KXBTC15M-TEST": state})
    sampler = KalshiResearchSampler(scorer, log_dir=tmp_path / "research")
    queue = sampler.subscribe_queue()

    async def run() -> None:
        await sampler.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        await sampler.stop()

        assert update.status == "skipped"
        assert update.reason == "insufficient_post_cost_edge"

    asyncio.run(run())


def test_research_sampler_blocks_yes_in_downtrend_regime(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    scorer = _FakeScorer(
        collector,
        {
            "KXBTC15M-TEST": KalshiLightGBMScoreState(
                **{
                    **_score_state(predicted_yes_probability=0.78).__dict__,
                    "price_momentum": -0.03,
                    "signed_contracts_sum_300s": -6.0,
                    "yes_taker_share_300s": 0.30,
                }
            )
        },
    )
    sampler = KalshiResearchSampler(
        scorer,
        log_dir=tmp_path / "research",
        config=KalshiResearchSamplerConfig(apply_regime_hard_gate=True),
    )
    queue = sampler.subscribe_queue()

    async def run() -> None:
        await sampler.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        await sampler.stop()

        assert update.status == "skipped"
        assert update.reason == "blocked_by_regime_downtrend"
        assert update.regime_label == "downtrend"
        assert update.side == "YES"

    asyncio.run(run())


def test_research_sampler_blocks_bucket_ban_policy_for_yes_price_bucket(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    scorer = _FakeScorer(
        collector,
        {
            "KXBTC15M-TEST": _score_state(
                predicted_yes_probability=0.44,
                market_prob=0.30,
                tau_minutes=8.0,
                yes_bid_cents=29,
                yes_ask_cents=31,
            )
        },
    )
    sampler = KalshiResearchSampler(scorer, log_dir=tmp_path / "research")
    queue = sampler.subscribe_queue()

    async def run() -> None:
        await sampler.start()
        update = await asyncio.wait_for(queue.get(), timeout=0.5)
        await sampler.stop()

        assert update.status == "skipped"
        assert update.reason == "blocked_by_bucket_policy"
        assert update.side == "YES"
        assert update.price_bucket == "30-40"
        assert update.chosen_side_probability_bucket == "40-50"
        assert update.bucket_policy_dimension == "price"
        assert update.bucket_policy_bucket == "30-40"
        assert update.bucket_policy_side == "YES"

    asyncio.run(run())

def test_research_sampler_signature_gating_allows_new_bucket_only(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    initial_state = _score_state(event_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    scorer = _FakeScorer(collector, {"KXBTC15M-TEST": initial_state})
    sampler = KalshiResearchSampler(scorer, log_dir=tmp_path / "research")
    queue = sampler.subscribe_queue()

    async def run() -> None:
        await sampler.start()
        first = await asyncio.wait_for(queue.get(), timeout=0.5)
        await scorer.push(initial_state)
        duplicate = await asyncio.wait_for(queue.get(), timeout=0.5)
        next_tau_state = _score_state(
            tau_minutes=9.0,
            event_time=datetime(2026, 1, 1, 12, 1, tzinfo=UTC),
        )
        await scorer.push(next_tau_state)
        second = await asyncio.wait_for(queue.get(), timeout=0.5)
        await sampler.stop()

        assert first.status == "recorded"
        assert duplicate.status == "skipped"
        assert duplicate.reason == "duplicate_signature"
        assert second.status == "recorded"
        assert second.sample is not None
        assert second.sample.tau_bucket == "8-10"

    asyncio.run(run())

def test_research_ledger_settles_samples_observation_only(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._markets[ticker] = _market(ticker=ticker, result="YES", status="settled")
    scorer = _FakeScorer(collector, {ticker: _score_state(ticker=ticker, predicted_yes_probability=0.78)})
    sampler = KalshiResearchSampler(scorer, log_dir=tmp_path / "research")
    ledger = KalshiResearchLedger(collector, sampler, log_dir=tmp_path / "research")
    sample_queue = sampler.subscribe_queue()
    settlement_queue = ledger.subscribe_queue()

    async def run() -> None:
        await ledger.start()
        await sampler.start()
        sample_update = await asyncio.wait_for(sample_queue.get(), timeout=0.5)
        settlement = await asyncio.wait_for(settlement_queue.get(), timeout=0.5)
        await sampler.stop()
        await ledger.stop()

        assert sample_update.status == "recorded"
        assert settlement.side == "YES"
        assert settlement.settlement_result == "YES"
        assert settlement.is_win is True
        assert settlement.realized_pnl_dollars == pytest.approx(
            1.0 - sample_update.sample.estimated_cash_required_dollars  # type: ignore[union-attr]
        )

    asyncio.run(run())

def test_research_runner_writes_research_logs_without_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_collector_start(self: KalshiMarketDataCollector) -> None:
        ticker = "KXBTC15M-TEST"
        now = datetime.now(UTC)
        self._states[ticker] = KalshiTickerState(
            ticker=ticker,
            last_yes_price_cents=55,
            last_trade_time=now,
            previous_yes_price_cents=54,
            close_time=now + timedelta(minutes=7),
            is_open=True,
            open_time=now,
            trade_id=None,
            count=0,
            taker_side=None,
            last_price_cents=55,
            yes_bid_cents=54,
            yes_ask_cents=56,
            volume=0,
            open_interest=0,
            dollar_volume=0,
            dollar_open_interest=0,
            ticker_update_time=now,
        )
        self._markets[ticker] = _market(ticker=ticker)

    monkeypatch.setattr(
        "scripts.run_kalshi_research_multi_model.KalshiCredentials.from_env",
        lambda environment: KalshiCredentials(api_key_id="demo-key", private_key_path=Path("tests/fixtures/demo.pem")),
    )
    monkeypatch.setattr("scripts.run_kalshi_research_multi_model.KalshiMarketDataCollector.start", fake_collector_start)
    monkeypatch.setattr(
        "src.live.kalshi.scorer.load_regularized_logistic_model_artifact",
        lambda _config: _fake_regularized_model(0.78),
    )

    args = argparse.Namespace(
        environment="demo",
        model_family=["lasso"],
        run_dir=[],
        runtime_run_dir=[],
        model_file=[],
        series=["KXBTC15M"],
        hourly_context_series="KXBTCD",
        ticker=[],
        log_root=str(tmp_path / "live_research"),
        metadata_refresh_interval_seconds=300.0,
        min_edge_cents=None,
        min_tau_minutes=None,
        max_tau_minutes=None,
        price_band_min_cents=None,
        price_band_max_cents=None,
        quote_max_age_seconds=None,
        contracts_per_sample=None,
        slippage_pct=None,
        max_runtime_seconds=0.05,
    )

    async def run() -> None:
        await run_research_multi_model(args)

    asyncio.run(run())

    research_logs = list((tmp_path / "live_research" / "research" / "lasso" / "demo").rglob("events.jsonl"))
    assert research_logs
    content = research_logs[0].read_text(encoding="utf-8")
    assert "research_sample_recorded" in content
    assert not (tmp_path / "live_research" / "execution").exists()


def test_research_ledger_recovers_open_samples_from_existing_logs(tmp_path: Path) -> None:
    collector = KalshiMarketDataCollector(_collector_config(tmp_path))
    ticker = "KXBTC15M-TEST"
    collector._markets[ticker] = _market(ticker=ticker, result="YES", status="settled")
    scorer = _FakeScorer(collector, {})
    sampler = KalshiResearchSampler(scorer, log_dir=tmp_path / "research")

    events_path = tmp_path / "research" / "demo" / "2026-01-01" / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    recorded_row = {
        "logged_at": "2026-01-01T12:00:00+00:00",
        "event_type": "research_sample_recorded",
        "payload": {
            "sample_id": "sample-1",
            "ticker": ticker,
            "side": "YES",
            "contracts": 1,
            "reference_price_cents": 56,
            "max_acceptable_entry_price_cents": 70,
            "predicted_yes_probability": 0.78,
            "predicted_no_probability": 0.22,
            "chosen_side_probability": 0.78,
            "feature_basis_market_prob": 0.55,
            "raw_model_edge": 0.23,
            "yes_post_cost_edge": 0.18,
            "no_post_cost_edge": -0.28,
            "chosen_post_cost_edge": 0.18,
            "last_yes_price_cents": 55,
            "last_price_cents": 55,
            "yes_bid_cents": 54,
            "yes_ask_cents": 56,
            "buy_yes_price_cents": 56,
            "buy_no_price_cents": 46,
            "quote_mid_prob": 0.55,
            "quote_spread_cents": 2,
            "quote_age_seconds": 0.1,
            "tau_minutes": 7.0,
            "tau_bucket": "6-8",
            "price_bucket": "50-60",
            "chosen_side_probability_bucket": "70-80",
            "chosen_side_edge_bucket": "10-20",
            "estimated_entry_cost_dollars": 0.57,
            "estimated_fees_dollars": 0.02,
            "estimated_cash_required_dollars": 0.59,
        },
    }
    events_path.write_text(json.dumps(recorded_row) + "\n", encoding="utf-8")

    ledger = KalshiResearchLedger(collector, sampler, log_dir=tmp_path / "research")
    settlement_queue = ledger.subscribe_queue()

    async def run() -> None:
        await ledger.start()
        settlement = await asyncio.wait_for(settlement_queue.get(), timeout=0.5)
        await ledger.stop()

        assert settlement.sample_id == "sample-1"
        assert settlement.ticker == ticker
        assert settlement.settlement_result == "YES"
        assert settlement.is_win is True

    asyncio.run(run())
