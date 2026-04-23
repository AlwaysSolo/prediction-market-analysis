from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import replace
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from src.live.data.btc_spot_feed import BTCSpotFeed, BTCSpotUpdate
from src.live.kalshi.collector import KalshiMarketDataCollector
from src.live.kalshi.features import (
    SPOT_V1_FEATURE_SCHEMA,
    KalshiFeatureEngineConfig,
    KalshiFeatureState,
    KalshiTradeFeatureAccumulator,
    KalshiFeatureUpdate,
    feature_update_from_state,
)
from src.live.kalshi.spot_features import SpotFeatureTracker
from src.live.kalshi.strike import strike_price_from_market
from src.live.kalshi.types import KalshiTickerState, KalshiTickerUpdate, utc_now

Callback = Callable[[KalshiFeatureUpdate], Awaitable[None] | None]


class KalshiFeatureStateEngine:
    def __init__(
        self,
        collector: KalshiMarketDataCollector,
        config: KalshiFeatureEngineConfig | None = None,
        spot_feed: BTCSpotFeed | None = None,
    ):
        self.collector = collector
        self.config = config or KalshiFeatureEngineConfig()
        self.spot_feed = spot_feed
        self._collector_queue: asyncio.Queue[KalshiTickerUpdate] | None = None
        self._states: dict[str, KalshiFeatureState] = {}
        self._accumulators: dict[str, KalshiTradeFeatureAccumulator] = {}
        self._spot_trackers: dict[str, SpotFeatureTracker] = {}
        self._callbacks: list[Callback] = []
        self._queues: list[asyncio.Queue[KalshiFeatureUpdate]] = []
        self._task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._logger = logging.getLogger(__name__)

    def _matches_series(self, ticker: str, series_ticker: str | None) -> bool:
        return bool(series_ticker) and ticker.startswith(f"{series_ticker}-")

    def _ticker_state_from_feature_state(self, feature_state: KalshiFeatureState) -> KalshiTickerState:
        return KalshiTickerState(
            ticker=feature_state.ticker,
            last_yes_price_cents=feature_state.last_yes_price_cents,
            last_trade_time=feature_state.last_trade_time,
            previous_yes_price_cents=feature_state.previous_yes_price_cents,
            close_time=feature_state.close_time,
            is_open=feature_state.is_open,
            open_time=feature_state.open_time,
            trade_id=feature_state.trade_id,
            count=feature_state.count,
            taker_side=feature_state.taker_side,
            last_price_cents=feature_state.last_price_cents,
            yes_bid_cents=feature_state.yes_bid_cents,
            yes_ask_cents=feature_state.yes_ask_cents,
            no_bid_cents=feature_state.no_bid_cents,
            no_ask_cents=feature_state.no_ask_cents,
            volume=feature_state.volume,
            open_interest=feature_state.open_interest,
            dollar_volume=feature_state.dollar_volume,
            dollar_open_interest=feature_state.dollar_open_interest,
            ticker_update_time=feature_state.ticker_update_time,
            received_at=feature_state.received_at,
            event_id=feature_state.event_id,
            raw_event_id=feature_state.raw_event_id,
        )

    def _should_publish_ticker(self, ticker: str) -> bool:
        configured_series = self.config.normalized_publish_series_tickers()
        if not configured_series:
            return True
        return any(self._matches_series(ticker, series_ticker) for series_ticker in configured_series)

    def _context_trade_age_seconds(self, state: KalshiFeatureState, target_time) -> float:
        reference_time = state.last_trade_time or state.event_time
        return max(0.0, (target_time - reference_time).total_seconds())

    def _empty_hourly_context_payload(self) -> dict[str, object]:
        return {
            "kxbtcd_atm_ticker": None,
            "kxbtcd_atm_z_implied": 0.0,
            "kxbtcd_atm_price_momentum": 0.0,
            "kxbtcd_atm_abs_price_momentum": 0.0,
            "kxbtcd_atm_distance_from_mid": 0.0,
            "kxbtcd_atm_trade_count_300s": 0.0,
            "kxbtcd_atm_signed_contracts_sum_300s": 0.0,
            "kxbtcd_atm_price_return_300s": 0.0,
            "kxbtcd_atm_price_volatility_300s": 0.0,
            "kxbtcd_atm_yes_taker_share_300s": 0.0,
            "k15_minus_k1h_atm_z": 0.0,
            "k15_minus_k1h_atm_price_return_300s": 0.0,
            "k15_k1h_atm_direction_agreement": 0.0,
        }

    def _direction_agreement(self, left_value: float, right_value: float) -> float:
        return float(np.sign(left_value) * np.sign(right_value))

    def _hourly_context_payload(self, target_state: KalshiFeatureState, context_state: KalshiFeatureState | None) -> dict[str, object]:
        if context_state is None:
            return self._empty_hourly_context_payload()
        target_z_implied = float(target_state.z_implied or 0.0)
        target_price_return_300s = float(target_state.price_return_300s or 0.0)
        context_z_implied = float(context_state.z_implied or 0.0)
        context_price_return_300s = float(context_state.price_return_300s or 0.0)
        return {
            "kxbtcd_atm_ticker": context_state.ticker,
            "kxbtcd_atm_z_implied": context_z_implied,
            "kxbtcd_atm_price_momentum": float(context_state.price_momentum or 0.0),
            "kxbtcd_atm_abs_price_momentum": float(context_state.abs_price_momentum or 0.0),
            "kxbtcd_atm_distance_from_mid": float(context_state.distance_from_mid or 0.0),
            "kxbtcd_atm_trade_count_300s": float(context_state.trade_count_300s or 0.0),
            "kxbtcd_atm_signed_contracts_sum_300s": float(context_state.signed_contracts_sum_300s or 0.0),
            "kxbtcd_atm_price_return_300s": context_price_return_300s,
            "kxbtcd_atm_price_volatility_300s": float(context_state.price_volatility_300s or 0.0),
            "kxbtcd_atm_yes_taker_share_300s": float(context_state.yes_taker_share_300s or 0.0),
            "k15_minus_k1h_atm_z": target_z_implied - context_z_implied,
            "k15_minus_k1h_atm_price_return_300s": target_price_return_300s - context_price_return_300s,
            "k15_k1h_atm_direction_agreement": self._direction_agreement(
                target_price_return_300s,
                context_price_return_300s,
            ),
        }

    def _select_hourly_context_state(self, target_state: KalshiFeatureState) -> KalshiFeatureState | None:
        context_series_ticker = self.config.hourly_context_series_ticker
        if not context_series_ticker or target_state.event_time is None:
            return None
        target_time = target_state.event_time
        active_context_states = [
            state
            for state in self._states.values()
            if self._matches_series(state.ticker, context_series_ticker)
            and state.open_time is not None
            and state.close_time is not None
            and state.open_time <= target_time
            and state.close_time > target_time
        ]
        if not active_context_states:
            return None
        nearest_close_time = min(state.close_time for state in active_context_states if state.close_time is not None)
        expiry_candidates = [state for state in active_context_states if state.close_time == nearest_close_time]

        best_rank: tuple[float, float, float, float, str] | None = None
        selected_state: KalshiFeatureState | None = None
        for state in expiry_candidates:
            if state.market_prob is None:
                continue
            age_seconds = self._context_trade_age_seconds(state, target_time)
            if age_seconds > self.config.hourly_context_max_staleness_seconds:
                continue
            trade_count_300s = float(state.trade_count_300s or 0.0)
            if trade_count_300s < self.config.hourly_context_min_recent_trade_count_300s:
                continue
            rank = (
                abs(float(state.market_prob) - 0.5),
                -trade_count_300s,
                -float(state.contracts_sum_300s or 0.0),
                age_seconds,
                state.ticker,
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                selected_state = state
        return selected_state

    def _apply_hourly_context(self, feature_state: KalshiFeatureState) -> KalshiFeatureState:
        target_series_ticker = self.config.hourly_context_target_series_ticker
        if not self._matches_series(feature_state.ticker, target_series_ticker):
            return feature_state
        selected_context = self._select_hourly_context_state(feature_state)
        return replace(feature_state, **self._hourly_context_payload(feature_state, selected_context))

    async def _publish_if_scoreable(self, feature_state: KalshiFeatureState) -> None:
        if not feature_state.is_scoreable or not self._should_publish_ticker(feature_state.ticker):
            return
        await self._publish_update(feature_update_from_state(feature_state))

    def _spot_tracker(self, ticker: str) -> SpotFeatureTracker:
        tracker = self._spot_trackers.get(ticker)
        if tracker is None:
            tracker = SpotFeatureTracker(
                detrend_halflife_seconds=self.config.external_spot_divergence_detrend_halflife_seconds,
                alert_window_seconds=self.config.external_spot_divergence_lookback_seconds,
            )
            self._spot_trackers[ticker] = tracker
        return tracker

    def _spot_snapshot_for_time(self, event_time) -> BTCSpotUpdate | None:
        if self.spot_feed is None:
            return None
        lookup = getattr(self.spot_feed, "lookup_at_or_before", None)
        if callable(lookup):
            return lookup(event_time, max_age_ms=self.config.external_spot_max_age_ms)
        snapshot = self.spot_feed.snapshot_state()
        if snapshot is None:
            return None
        age_ms = max(0.0, (event_time - snapshot.event_time).total_seconds() * 1000.0)
        if age_ms > float(self.config.external_spot_max_age_ms):
            return None
        return snapshot

    def _apply_external_spot(self, feature_state: KalshiFeatureState) -> KalshiFeatureState:
        requires_spot = (
            self.config.external_spot_required_for_schema
            or self.config.feature_schema == SPOT_V1_FEATURE_SCHEMA
        )
        if self.spot_feed is None:
            if requires_spot:
                return replace(feature_state, is_scoreable=False, btc_spot_is_fresh=False)
            return feature_state
        snapshot = self._spot_snapshot_for_time(feature_state.event_time)
        if snapshot is None:
            if requires_spot:
                return replace(feature_state, is_scoreable=False, btc_spot_is_fresh=False)
            return feature_state
        strike_price = strike_price_from_market(self.collector.get_market(feature_state.ticker))
        tracker = self._spot_tracker(feature_state.ticker)
        spot_fields = tracker.enrich(
            event_time=feature_state.event_time,
            quote_mid_prob=feature_state.quote_mid_prob,
            tau_minutes=feature_state.tau_minutes,
            strike_price=strike_price,
            spot_update=snapshot,
        )
        enriched_state = replace(feature_state, **spot_fields)
        if requires_spot and not bool(enriched_state.btc_spot_is_fresh):
            enriched_state = replace(enriched_state, is_scoreable=False)
        zscore = tracker.primary_detrended_zscore(feature_state.event_time)
        if zscore is not None and abs(zscore) >= float(self.config.external_spot_divergence_sigma_threshold):
            self._logger.warning(
                "External spot divergence z-score breached threshold for %s: %.3f",
                feature_state.ticker,
                zscore,
            )
        return enriched_state

    async def _refresh_context_dependent_targets(self, event_time) -> None:
        target_series_ticker = self.config.hourly_context_target_series_ticker
        if not target_series_ticker:
            return
        for ticker, current_state in list(self._states.items()):
            if not self._matches_series(ticker, target_series_ticker):
                continue
            ticker_state = self.collector.get_state(ticker)
            if ticker_state is None:
                ticker_state = self._ticker_state_from_feature_state(current_state)
            accumulator = self._accumulators.setdefault(
                ticker,
                KalshiTradeFeatureAccumulator(config=self.config),
            )
            rebuilt_state = accumulator.build_feature_state_from_state(ticker_state, event_time=event_time)
            enriched_state = self._apply_hourly_context(rebuilt_state)
            if enriched_state == current_state:
                continue
            self._states[ticker] = enriched_state
            await self._publish_if_scoreable(enriched_state)

    async def start(self) -> None:
        if self._task and not self._task.done():
            return

        self._stop_event = asyncio.Event()
        self._ready_event.clear()
        if self._collector_queue is None:
            self._collector_queue = self.collector.subscribe_queue()

        await self._bootstrap_from_collector()
        self._task = asyncio.create_task(self._consume_loop(), name="kalshi-feature-engine")
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

    def get_state(self, ticker: str) -> KalshiFeatureState | None:
        return self._states.get(ticker)

    def snapshot_states(self) -> dict[str, KalshiFeatureState]:
        return dict(self._states)

    def subscribe(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    def subscribe_queue(self, maxsize: int = 0) -> asyncio.Queue[KalshiFeatureUpdate]:
        queue: asyncio.Queue[KalshiFeatureUpdate] = asyncio.Queue(maxsize=maxsize)
        self._queues.append(queue)
        return queue

    async def _bootstrap_from_collector(self) -> None:
        snapshot_time = utc_now()
        for ticker_state in self.collector.snapshot_states().values():
            accumulator = self._accumulators.setdefault(
                ticker_state.ticker,
                KalshiTradeFeatureAccumulator(config=self.config),
            )
            feature_state = accumulator.build_feature_state_from_state(ticker_state, event_time=snapshot_time)
            feature_state = self._apply_hourly_context(feature_state)
            feature_state = self._apply_external_spot(feature_state)
            self._states[feature_state.ticker] = feature_state
        for feature_state in self._states.values():
            await self._publish_if_scoreable(feature_state)

    async def _consume_loop(self) -> None:
        if self._collector_queue is None:
            return

        try:
            while not self._stop_event.is_set():
                update = await self._collector_queue.get()
                await self._handle_ticker_update(update)
        except asyncio.CancelledError:
            raise

    async def _handle_ticker_update(self, update: KalshiTickerUpdate) -> None:
        accumulator = self._accumulators.setdefault(
            update.ticker,
            KalshiTradeFeatureAccumulator(config=self.config),
        )
        feature_state = accumulator.build_feature_state_from_update(update)
        feature_state = self._apply_hourly_context(feature_state)
        feature_state = self._apply_external_spot(feature_state)
        current = self._states.get(feature_state.ticker)
        if feature_state == current:
            if self._matches_series(update.ticker, self.config.hourly_context_series_ticker):
                await self._refresh_context_dependent_targets(update.event_time)
            return

        self._states[feature_state.ticker] = feature_state
        await self._publish_if_scoreable(feature_state)
        if self._matches_series(update.ticker, self.config.hourly_context_series_ticker):
            await self._refresh_context_dependent_targets(update.event_time)

    async def _publish_update(self, update: KalshiFeatureUpdate) -> None:
        for callback in self._callbacks:
            result = callback(update)
            if inspect.isawaitable(result):
                await result
        for queue in self._queues:
            await queue.put(update)
