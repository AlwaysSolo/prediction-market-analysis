from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx

from src.common.client import retry_request
from src.indexers.kalshi.models import Market
from src.live.kalshi.auth import build_auth_headers
from src.live.kalshi.config import KalshiCredentials, KalshiEnvironment


class _HasKalshiAuth(Protocol):
    environment: KalshiEnvironment
    credentials: KalshiCredentials


class KalshiLiveRestClient:
    def __init__(self, config: _HasKalshiAuth):
        self.environment = config.environment
        self.credentials = config.credentials
        self.client = httpx.Client(base_url=config.environment.api_base_url, timeout=30.0)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> KalshiLiveRestClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @retry_request()
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timestamp_ms = int(datetime.now(UTC).timestamp() * 1000)
        signing_path = self._signing_path(path)
        headers = build_auth_headers(self.credentials, method, signing_path, timestamp_ms)
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        response = self.client.request(method, path, params=params, json=json_body, headers=headers)
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()

    @staticmethod
    def _signing_path(path: str) -> str:
        if path.startswith("/trade-api/"):
            return path
        return f"/trade-api/v2{path}"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("GET", path, params=params)

    def _post(self, path: str, json_body: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("POST", path, params=params, json_body=json_body)

    def _delete(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("DELETE", path, params=params)

    def get_markets_page(
        self,
        *,
        limit: int = 200,
        status: str = "open",
        cursor: str | None = None,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        tickers: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, Any], list[Market]]:
        params: dict[str, Any] = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if tickers:
            params["tickers"] = ",".join(tickers)
        payload = self._get("/markets", params=params)
        markets = [Market.from_dict(item) for item in payload.get("markets", [])]
        return payload, markets

    def iter_markets(
        self,
        limit: int = 200,
        status: str = "open",
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        tickers: tuple[str, ...] | None = None,
    ) -> Generator[tuple[dict, list[Market]], None, None]:
        cursor = None
        while True:
            payload, markets = self.get_markets_page(
                limit=limit,
                status=status,
                cursor=cursor,
                event_ticker=event_ticker,
                series_ticker=series_ticker,
                tickers=tickers,
            )
            yield payload, markets
            cursor = payload.get("cursor")
            if not cursor:
                break

    def create_order(self, order: dict[str, Any], *, subaccount: int | None = None) -> dict[str, Any]:
        payload = dict(order)
        if subaccount is not None:
            payload["subaccount"] = subaccount
        return self._post("/portfolio/orders", payload)

    def get_market(self, ticker: str) -> Market:
        payload = self._get(f"/markets/{ticker}")
        return Market.from_dict(payload["market"])

    def get_market_orderbook(self, ticker: str, *, depth: int = 1) -> dict[str, Any]:
        params = {"depth": depth} if depth > 0 else None
        return self._get(f"/markets/{ticker}/orderbook", params=params)

    def get_public_market(self, ticker: str, environment: KalshiEnvironment | None = None) -> Market:
        target_environment = environment or self.environment
        with httpx.Client(base_url=target_environment.api_base_url, timeout=30.0) as public_client:
            response = public_client.get(f"/markets/{ticker}")
            response.raise_for_status()
            payload = response.json()
        return Market.from_dict(payload["market"])

    def get_balance(self, *, subaccount: int = 0) -> dict[str, Any]:
        return self._get("/portfolio/balance", params={"subaccount": subaccount})

    def get_subaccount_balances(self) -> list[dict[str, Any]]:
        payload = self._get("/portfolio/subaccounts/balances")
        return list(payload.get("subaccount_balances", []))

    def iter_positions(
        self,
        *,
        limit: int = 100,
        subaccount: int = 0,
        ticker: str | None = None,
        count_filter: str = "position,total_traded",
    ) -> Generator[dict[str, Any], None, None]:
        cursor = None
        while True:
            params: dict[str, Any] = {
                "limit": limit,
                "subaccount": subaccount,
                "count_filter": count_filter,
            }
            if ticker:
                params["ticker"] = ticker
            if cursor:
                params["cursor"] = cursor
            payload = self._get("/portfolio/positions", params=params)
            yield payload
            cursor = payload.get("cursor")
            if not cursor:
                break

    def get_positions(
        self,
        *,
        limit: int = 100,
        subaccount: int = 0,
        ticker: str | None = None,
        count_filter: str = "position,total_traded",
    ) -> list[dict[str, Any]]:
        market_positions: list[dict[str, Any]] = []
        for payload in self.iter_positions(
            limit=limit,
            subaccount=subaccount,
            ticker=ticker,
            count_filter=count_filter,
        ):
            market_positions.extend(payload.get("market_positions", []))
        return market_positions

    def iter_orders(
        self,
        *,
        limit: int = 100,
        ticker: str | None = None,
        status: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        subaccount: int | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        cursor = None
        while True:
            params: dict[str, Any] = {"limit": limit}
            if ticker:
                params["ticker"] = ticker
            if status:
                params["status"] = status
            if min_ts is not None:
                params["min_ts"] = min_ts
            if max_ts is not None:
                params["max_ts"] = max_ts
            if subaccount is not None:
                params["subaccount"] = subaccount
            if cursor:
                params["cursor"] = cursor
            payload = self._get("/portfolio/orders", params=params)
            yield payload
            cursor = payload.get("cursor")
            if not cursor:
                break

    def get_orders(
        self,
        *,
        limit: int = 100,
        ticker: str | None = None,
        status: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        subaccount: int | None = None,
    ) -> list[dict[str, Any]]:
        orders: list[dict[str, Any]] = []
        for payload in self.iter_orders(
            limit=limit,
            ticker=ticker,
            status=status,
            min_ts=min_ts,
            max_ts=max_ts,
            subaccount=subaccount,
        ):
            orders.extend(payload.get("orders", []))
        return orders

    def get_order(self, order_id: str, *, subaccount: int | None = None) -> dict[str, Any]:
        params = {"subaccount": subaccount} if subaccount is not None else None
        return self._get(f"/portfolio/orders/{order_id}", params=params)

    def cancel_order(self, order_id: str, *, subaccount: int | None = None) -> dict[str, Any]:
        params = {"subaccount": subaccount} if subaccount is not None else None
        return self._delete(f"/portfolio/orders/{order_id}", params=params)
