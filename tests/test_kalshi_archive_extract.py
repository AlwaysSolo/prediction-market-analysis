from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.indexers.kalshi.archive_extract import detect_archive_root, extract_series_from_archive


def test_extract_series_from_archive_filters_master_parquet(tmp_path: Path):
    archive_root = tmp_path / "kalshi"
    markets_dir = archive_root / "markets"
    trades_dir = archive_root / "trades"
    markets_dir.mkdir(parents=True)
    trades_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"ticker": "KXBTCD-26MAR2801-T80000.00", "event_ticker": "KXBTCD-26MAR2801", "status": "finalized"},
            {"ticker": "KXBTC15M-26MAR280115-00", "event_ticker": "KXBTC15M-26MAR280115", "status": "finalized"},
        ]
    ).to_parquet(markets_dir / "markets_0_10000.parquet", index=False)
    pd.DataFrame(
        [
            {"trade_id": "t1", "ticker": "KXBTCD-26MAR2801-T80000.00", "created_time": "2026-03-28T00:01:00Z"},
            {"trade_id": "t2", "ticker": "KXBTC15M-26MAR280115-00", "created_time": "2026-03-28T00:01:00Z"},
        ]
    ).to_parquet(trades_dir / "trades_0_10000.parquet", index=False)

    result = extract_series_from_archive("KXBTCD", archive_root=archive_root, output_dir=tmp_path / "out")

    markets_df = pd.read_parquet(result.markets_parquet)
    trades_df = pd.read_parquet(result.trades_parquet)

    assert result.markets_count == 1
    assert result.trades_count == 1
    assert set(markets_df["ticker"]) == {"KXBTCD-26MAR2801-T80000.00"}
    assert set(trades_df["ticker"]) == {"KXBTCD-26MAR2801-T80000.00"}


def test_detect_archive_root_finds_markets_and_trades(tmp_path: Path):
    archive_root = tmp_path / "data" / "kalshi"
    (archive_root / "markets").mkdir(parents=True)
    (archive_root / "trades").mkdir(parents=True)

    assert detect_archive_root(archive_root) == archive_root
