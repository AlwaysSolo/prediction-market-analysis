from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class KalshiEnvironment(str, Enum):
    DEMO = "demo"
    PRODUCTION = "production"

    @property
    def api_base_url(self) -> str:
        if self is KalshiEnvironment.DEMO:
            return "https://demo-api.kalshi.co/trade-api/v2"
        return "https://api.elections.kalshi.com/trade-api/v2"

    @property
    def websocket_url(self) -> str:
        if self is KalshiEnvironment.DEMO:
            return "wss://demo-api.kalshi.co/trade-api/ws/v2"
        return "wss://api.elections.kalshi.com/trade-api/ws/v2"


@dataclass(frozen=True)
class KalshiCredentials:
    api_key_id: str
    private_key_path: Path

    @classmethod
    def from_env(cls, environment: KalshiEnvironment) -> KalshiCredentials:
        if environment is KalshiEnvironment.DEMO:
            key_env = "KALSHI_DEMO_API_KEY_ID"
            path_env = "KALSHI_DEMO_PRIVATE_KEY_PATH"
        else:
            key_env = "KALSHI_PROD_API_KEY_ID"
            path_env = "KALSHI_PROD_PRIVATE_KEY_PATH"

        api_key_id = os.getenv(key_env) or os.getenv("KALSHI_API_KEY_ID")
        private_key_path = os.getenv(path_env) or os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if not api_key_id or not private_key_path:
            raise ValueError(
                f"Missing Kalshi credentials for {environment.value}. "
                f"Expected {key_env}/{path_env} or shared KALSHI_API_KEY_ID/KALSHI_PRIVATE_KEY_PATH."
            )
        return cls(api_key_id=api_key_id, private_key_path=Path(private_key_path))


@dataclass(frozen=True)
class KalshiReconnectConfig:
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0


@dataclass(frozen=True)
class KalshiCollectorConfig:
    environment: KalshiEnvironment
    credentials: KalshiCredentials
    log_dir: Path = Path("output/live/kalshi/raw")
    series_tickers: tuple[str, ...] = ()
    market_tickers: tuple[str, ...] = ()
    metadata_refresh_interval_seconds: float = 300.0
    max_stale_message_age_seconds: float = 30.0
    log_stream_messages: bool = False
    reconnect: KalshiReconnectConfig = field(default_factory=KalshiReconnectConfig)

    def normalized_market_tickers(self) -> tuple[str, ...]:
        return tuple(sorted({ticker for ticker in self.market_tickers if ticker}))

    def normalized_series_tickers(self) -> tuple[str, ...]:
        return tuple(sorted({ticker for ticker in self.series_tickers if ticker}))
