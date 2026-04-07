from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.kalshi.live_archive import repair_live_archive  # noqa: E402


async def _run(args: argparse.Namespace) -> None:
    await repair_live_archive(
        Path(args.archive_root),
        run_name=args.run_name,
        environment=args.environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact leftover staged Kalshi live archive shards into Parquet.")
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--environment", default="production")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
