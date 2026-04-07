from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indexers.kalshi.client import KalshiClient  # noqa: E402
from src.indexers.kalshi.models import Market  # noqa: E402


def iter_series_markets(series_ticker: str, status: str, limit: int = 1000) -> list[Market]:
    """Fetch all markets for a Kalshi series and status."""
    client = KalshiClient()
    markets: list[Market] = []
    cursor: str | None = None

    try:
        while True:
            params: dict[str, object] = {
                "series_ticker": series_ticker,
                "status": status,
                "limit": limit,
            }
            if cursor:
                params["cursor"] = cursor

            data = client._get("/markets", params=params)
            batch = [Market.from_dict(m) for m in data.get("markets", [])]
            markets.extend(batch)
            cursor = data.get("cursor")

            if not cursor:
                break
    finally:
        client.close()

    return markets


def fetch_series_trades(series_ticker: str, max_workers: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch all settled and currently open markets for a series, then fetch trades."""
    settled_markets = iter_series_markets(series_ticker, "settled")
    open_markets = iter_series_markets(series_ticker, "open")

    by_ticker: dict[str, Market] = {}
    for market in settled_markets + open_markets:
        by_ticker[market.ticker] = market

    markets = sorted(by_ticker.values(), key=lambda m: (m.close_time or datetime.min, m.ticker))
    markets_df = pd.DataFrame([asdict(m) for m in markets])

    def fetch_market_trades(ticker: str) -> list[dict]:
        client = KalshiClient()
        try:
            trades = client.get_market_trades(ticker, verbose=False)
            fetched_at = datetime.now(UTC)
            return [{**asdict(trade), "_fetched_at": fetched_at} for trade in trades]
        finally:
            client.close()

    all_trades: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_market_trades, market.ticker): market.ticker for market in markets}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                all_trades.extend(future.result())
            except Exception as exc:
                print(f"Error fetching trades for {ticker}: {exc}")

    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["created_time", "trade_id"]).reset_index(drop=True)

    return markets_df, trades_df


def fetch_series_trades_incremental(series_ticker: str, output_dir: Path, max_workers: int = 2) -> tuple[pd.DataFrame, int]:
    """Fetch series trades market-by-market so interrupted runs can resume."""
    settled_markets = iter_series_markets(series_ticker, "settled")
    open_markets = iter_series_markets(series_ticker, "open")

    by_ticker: dict[str, Market] = {}
    for market in settled_markets + open_markets:
        by_ticker[market.ticker] = market

    markets = sorted(by_ticker.values(), key=lambda m: (m.close_time or datetime.min, m.ticker))
    markets_df = pd.DataFrame([asdict(m) for m in markets])
    markets_parquet = output_dir / f"{series_ticker}_markets.parquet"
    markets_csv = output_dir / f"{series_ticker}_markets.csv"
    markets_df.to_parquet(markets_parquet, index=False)
    markets_df.to_csv(markets_csv, index=False)

    trades_dir = output_dir / series_ticker / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)

    def fetch_market_trades(market: Market) -> tuple[str, int]:
        output_path = trades_dir / f"{market.ticker}.parquet"
        if output_path.exists():
            existing = pd.read_parquet(output_path)
            return market.ticker, len(existing)

        client = KalshiClient()
        try:
            trades = client.get_market_trades(market.ticker, verbose=False)
            fetched_at = datetime.now(UTC)
            rows = [{**asdict(trade), "_fetched_at": fetched_at} for trade in trades]
            df = pd.DataFrame(rows)
            df.to_parquet(output_path, index=False)
            return market.ticker, len(df)
        finally:
            client.close()

    total_trades = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_market_trades, market): market.ticker for market in markets}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                _, trade_count = future.result()
                total_trades += trade_count
                print({"ticker": ticker, "trades": trade_count, "running_total": total_trades})
            except Exception as exc:
                print(f"Error fetching trades for {ticker}: {exc}")

    return markets_df, total_trades


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Kalshi markets and trades for a single series ticker.")
    parser.add_argument("series_ticker", help="Kalshi series ticker, for example KXBTC15M")
    parser.add_argument(
        "--output-dir",
        default="output/kalshi_series",
        help="Directory where parquet/csv files will be written",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Concurrent workers for per-market trade fetches",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Save one parquet file per market so interrupted runs can resume",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markets_parquet = output_dir / f"{args.series_ticker}_markets.parquet"
    markets_csv = output_dir / f"{args.series_ticker}_markets.csv"
    if args.incremental:
        markets_df, total_trades = fetch_series_trades_incremental(
            args.series_ticker,
            output_dir=output_dir,
            max_workers=args.max_workers,
        )
        markets_df.to_parquet(markets_parquet, index=False)
        markets_df.to_csv(markets_csv, index=False)
        print(
            {
                "series_ticker": args.series_ticker,
                "markets": len(markets_df),
                "trades_saved_incrementally": total_trades,
                "markets_parquet": str(markets_parquet),
                "trades_dir": str(output_dir / args.series_ticker / "trades"),
            }
        )
    else:
        markets_df, trades_df = fetch_series_trades(args.series_ticker, max_workers=args.max_workers)
        trades_parquet = output_dir / f"{args.series_ticker}_trades.parquet"
        trades_csv = output_dir / f"{args.series_ticker}_trades.csv"
        markets_df.to_parquet(markets_parquet, index=False)
        markets_df.to_csv(markets_csv, index=False)
        trades_df.to_parquet(trades_parquet, index=False)
        trades_df.to_csv(trades_csv, index=False)
        print(
            {
                "series_ticker": args.series_ticker,
                "markets": len(markets_df),
                "trades": len(trades_df),
                "markets_parquet": str(markets_parquet),
                "trades_parquet": str(trades_parquet),
            }
        )


if __name__ == "__main__":
    main()
