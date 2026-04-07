from collections.abc import Generator
from typing import Any, Optional

import httpx

from src.common.client import retry_request
from src.indexers.kalshi.models import Market, Trade

KALSHI_API_HOST = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiClient:
    def __init__(self, host: str = KALSHI_API_HOST):
        self.host = host
        self.client = httpx.Client(base_url=host, timeout=30.0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def close(self):
        self.client.close()

    @retry_request()
    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """Make a GET request with retry/backoff."""
        response = self.client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    def get_market(self, ticker: str) -> Market:
        data = self._get(f"/markets/{ticker}")
        return Market.from_dict(data["market"])

    def get_historical_cutoff(self) -> dict[str, Any]:
        return self._get("/historical/cutoff")

    def get_historical_market(self, ticker: str) -> Market:
        data = self._get(f"/historical/markets/{ticker}")
        return Market.from_dict(data["market"])

    def get_markets_page(
        self,
        *,
        limit: int = 200,
        cursor: str | None = None,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        tickers: tuple[str, ...] | None = None,
        status: str | None = None,
        min_created_ts: int | None = None,
        max_created_ts: int | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        min_settled_ts: int | None = None,
        max_settled_ts: int | None = None,
        min_updated_ts: int | None = None,
        mve_filter: str | None = None,
    ) -> tuple[dict[str, Any], list[Market]]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if tickers:
            params["tickers"] = ",".join(tickers)
        if status:
            params["status"] = status
        if min_created_ts is not None:
            params["min_created_ts"] = min_created_ts
        if max_created_ts is not None:
            params["max_created_ts"] = max_created_ts
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        if max_close_ts is not None:
            params["max_close_ts"] = max_close_ts
        if min_settled_ts is not None:
            params["min_settled_ts"] = min_settled_ts
        if max_settled_ts is not None:
            params["max_settled_ts"] = max_settled_ts
        if min_updated_ts is not None:
            params["min_updated_ts"] = min_updated_ts
        if mve_filter:
            params["mve_filter"] = mve_filter
        data = self._get("/markets", params=params)
        return data, [Market.from_dict(item) for item in data.get("markets", [])]

    def iter_markets_pages(
        self,
        *,
        limit: int = 200,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        tickers: tuple[str, ...] | None = None,
        status: str | None = None,
        min_created_ts: int | None = None,
        max_created_ts: int | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        min_settled_ts: int | None = None,
        max_settled_ts: int | None = None,
        min_updated_ts: int | None = None,
        mve_filter: str | None = None,
    ) -> Generator[tuple[dict[str, Any], list[Market]], None, None]:
        cursor = None
        while True:
            payload, markets = self.get_markets_page(
                limit=limit,
                cursor=cursor,
                event_ticker=event_ticker,
                series_ticker=series_ticker,
                tickers=tickers,
                status=status,
                min_created_ts=min_created_ts,
                max_created_ts=max_created_ts,
                min_close_ts=min_close_ts,
                max_close_ts=max_close_ts,
                min_settled_ts=min_settled_ts,
                max_settled_ts=max_settled_ts,
                min_updated_ts=min_updated_ts,
                mve_filter=mve_filter,
            )
            yield payload, markets
            cursor = payload.get("cursor")
            if not cursor:
                break

    def get_historical_markets_page(
        self,
        *,
        limit: int = 200,
        cursor: str | None = None,
        tickers: tuple[str, ...] | None = None,
        event_ticker: str | None = None,
        mve_filter: str | None = None,
    ) -> tuple[dict[str, Any], list[Market]]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if tickers:
            params["tickers"] = ",".join(tickers)
        if event_ticker:
            params["event_ticker"] = event_ticker
        if mve_filter:
            params["mve_filter"] = mve_filter
        data = self._get("/historical/markets", params=params)
        return data, [Market.from_dict(item) for item in data.get("markets", [])]

    def iter_historical_markets_pages(
        self,
        *,
        limit: int = 200,
        tickers: tuple[str, ...] | None = None,
        event_ticker: str | None = None,
        mve_filter: str | None = None,
    ) -> Generator[tuple[dict[str, Any], list[Market]], None, None]:
        cursor = None
        while True:
            payload, markets = self.get_historical_markets_page(
                limit=limit,
                cursor=cursor,
                tickers=tickers,
                event_ticker=event_ticker,
                mve_filter=mve_filter,
            )
            yield payload, markets
            cursor = payload.get("cursor")
            if not cursor:
                break

    def get_market_trades(
        self,
        ticker: str,
        limit: int = 1000,
        verbose: bool = True,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> list[Trade]:
        all_trades = []
        cursor = None

        while True:
            params = {"ticker": ticker, "limit": limit}
            if cursor:
                params["cursor"] = cursor
            if min_ts is not None:
                params["min_ts"] = min_ts
            if max_ts is not None:
                params["max_ts"] = max_ts

            data = self._get("/markets/trades", params=params)

            trades = [Trade.from_dict(t) for t in data.get("trades", [])]
            if trades:
                all_trades.extend(trades)
                if verbose:
                    print(f"Fetched {len(trades)} trades (total: {len(all_trades)})")

            cursor = data.get("cursor")
            if not cursor:
                break

        return all_trades

    def get_market_trades_page(
        self,
        *,
        ticker: str,
        limit: int = 1000,
        cursor: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
    ) -> tuple[dict[str, Any], list[Trade]]:
        params: dict[str, Any] = {"ticker": ticker, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        data = self._get("/markets/trades", params=params)
        return data, [Trade.from_dict(item) for item in data.get("trades", [])]

    def iter_market_trades_pages(
        self,
        *,
        ticker: str,
        limit: int = 1000,
        min_ts: int | None = None,
        max_ts: int | None = None,
    ) -> Generator[tuple[dict[str, Any], list[Trade]], None, None]:
        cursor = None
        while True:
            payload, trades = self.get_market_trades_page(
                ticker=ticker,
                limit=limit,
                cursor=cursor,
                min_ts=min_ts,
                max_ts=max_ts,
            )
            yield payload, trades
            cursor = payload.get("cursor")
            if not cursor:
                break

    def get_historical_trades_page(
        self,
        *,
        ticker: str,
        limit: int = 1000,
        cursor: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
    ) -> tuple[dict[str, Any], list[Trade]]:
        params: dict[str, Any] = {"ticker": ticker, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        data = self._get("/historical/trades", params=params)
        return data, [Trade.from_dict(item) for item in data.get("trades", [])]

    def iter_historical_trades_pages(
        self,
        *,
        ticker: str,
        limit: int = 1000,
        min_ts: int | None = None,
        max_ts: int | None = None,
    ) -> Generator[tuple[dict[str, Any], list[Trade]], None, None]:
        cursor = None
        while True:
            payload, trades = self.get_historical_trades_page(
                ticker=ticker,
                limit=limit,
                cursor=cursor,
                min_ts=min_ts,
                max_ts=max_ts,
            )
            yield payload, trades
            cursor = payload.get("cursor")
            if not cursor:
                break

    def get_market_candlesticks(
        self,
        *,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
        include_latest_before_start: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        if include_latest_before_start:
            params["include_latest_before_start"] = "true"
        return self._get(f"/series/{series_ticker}/markets/{ticker}/candlesticks", params=params)

    def get_historical_market_candlesticks(
        self,
        *,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> dict[str, Any]:
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        return self._get(f"/historical/markets/{ticker}/candlesticks", params=params)

    def list_markets(self, limit: int = 20, **kwargs) -> list[Market]:
        params = {"limit": limit, **kwargs}
        data = self._get("/markets", params=params)
        return [Market.from_dict(m) for m in data.get("markets", [])]

    def list_all_markets(self, limit: int = 200) -> list[Market]:
        all_markets = []
        cursor = None

        while True:
            params = {"limit": limit}
            if cursor:
                params["cursor"] = cursor

            data = self._get("/markets", params=params)

            markets = [Market.from_dict(m) for m in data.get("markets", [])]
            if markets:
                all_markets.extend(markets)
                print(f"Fetched {len(markets)} markets (total: {len(all_markets)})")

            cursor = data.get("cursor")
            if not cursor:
                break

        return all_markets

    def iter_markets(
        self,
        limit: int = 200,
        cursor: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
    ) -> Generator[tuple[list[Market], Optional[str]], None, None]:
        while True:
            params = {"limit": limit}
            if cursor:
                params["cursor"] = cursor
            if min_close_ts is not None:
                params["min_close_ts"] = min_close_ts
            if max_close_ts is not None:
                params["max_close_ts"] = max_close_ts

            data = self._get("/markets", params=params)

            markets = [Market.from_dict(m) for m in data.get("markets", [])]
            cursor = data.get("cursor")

            yield markets, cursor

            if not cursor:
                break

    def get_recent_trades(self, limit: int = 100) -> list[Trade]:
        data = self._get("/markets/trades", params={"limit": limit})
        return [Trade.from_dict(t) for t in data.get("trades", [])]
