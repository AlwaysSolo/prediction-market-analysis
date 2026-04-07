from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indexers.kalshi.archive_extract import extract_series_from_archive  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a Kalshi series from the local master parquet archive into trainer-ready parquet files."
    )
    parser.add_argument("series_ticker", help="Kalshi series ticker, for example KXBTCD")
    parser.add_argument(
        "--archive-root",
        default=None,
        help='Root of the local master Kalshi parquet archive. Defaults to auto-detecting "data/data/data/kalshi" or "data/kalshi".',
    )
    parser.add_argument(
        "--output-dir",
        default="output/kalshi_series",
        help='Directory where "<series>_markets.parquet" and "<series>_trades.parquet" will be written.',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted series files in the output directory.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec to use for the extracted files.",
    )
    args = parser.parse_args()

    result = extract_series_from_archive(
        args.series_ticker,
        archive_root=args.archive_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        compression=args.compression,
    )
    print(json.dumps(asdict(result), indent=2, default=str))


if __name__ == "__main__":
    main()
