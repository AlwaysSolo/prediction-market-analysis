from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class SeriesExtractionResult:
    series_ticker: str
    archive_root: Path
    output_dir: Path
    markets_parquet: Path
    trades_parquet: Path
    markets_count: int
    trades_count: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_archive_candidates() -> tuple[Path, ...]:
    repo_root = _repo_root()
    return (
        repo_root / "data" / "data" / "data" / "kalshi",
        repo_root / "data" / "kalshi",
    )


def detect_archive_root(archive_root: str | Path | None = None) -> Path:
    candidates = [Path(archive_root).expanduser()] if archive_root else list(_default_archive_candidates())
    for candidate in candidates:
        if (candidate / "markets").exists() and (candidate / "trades").exists():
            return candidate
    searched = "\n".join(f'  - "{candidate}"' for candidate in candidates)
    raise FileNotFoundError(
        "Could not find a Kalshi parquet archive with both markets/ and trades/.\n"
        f"Searched:\n{searched}"
    )


def _sql_literal(path_or_value: str | Path) -> str:
    return str(path_or_value).replace("\\", "/").replace("'", "''")


def extract_series_from_archive(
    series_ticker: str,
    *,
    archive_root: str | Path | None = None,
    output_dir: str | Path = "output/kalshi_series",
    overwrite: bool = False,
    compression: str = "zstd",
) -> SeriesExtractionResult:
    resolved_archive_root = detect_archive_root(archive_root)
    resolved_output_dir = Path(output_dir).expanduser()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    markets_glob = resolved_archive_root / "markets" / "*.parquet"
    trades_glob = resolved_archive_root / "trades" / "*.parquet"
    markets_parquet = resolved_output_dir / f"{series_ticker}_markets.parquet"
    trades_parquet = resolved_output_dir / f"{series_ticker}_trades.parquet"

    if not overwrite:
        existing = [path for path in (markets_parquet, trades_parquet) if path.exists()]
        if existing:
            lines = "\n".join(f'  - "{path}"' for path in existing)
            raise FileExistsError(
                "Refusing to overwrite existing extracted series files without --overwrite.\n"
                f"Existing files:\n{lines}"
            )

    pattern = f"{series_ticker}-%"
    markets_glob_sql = _sql_literal(markets_glob)
    trades_glob_sql = _sql_literal(trades_glob)
    markets_parquet_sql = _sql_literal(markets_parquet)
    trades_parquet_sql = _sql_literal(trades_parquet)
    pattern_sql = pattern.replace("'", "''")
    compression_sql = compression.replace("'", "''")

    connection = duckdb.connect()
    try:
        markets_count = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{markets_glob_sql}')
                WHERE ticker LIKE '{pattern_sql}'
                """
            ).fetchone()[0]
        )
        trades_count = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{trades_glob_sql}')
                WHERE ticker LIKE '{pattern_sql}'
                """
            ).fetchone()[0]
        )
        if markets_count <= 0:
            raise ValueError(
                f'No market rows found for series "{series_ticker}" in archive "{resolved_archive_root}".'
            )
        if trades_count <= 0:
            raise ValueError(
                f'No trade rows found for series "{series_ticker}" in archive "{resolved_archive_root}".'
            )

        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{markets_glob_sql}')
                WHERE ticker LIKE '{pattern_sql}'
            ) TO '{markets_parquet_sql}' (FORMAT PARQUET, COMPRESSION {compression_sql})
            """
        )
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{trades_glob_sql}')
                WHERE ticker LIKE '{pattern_sql}'
            ) TO '{trades_parquet_sql}' (FORMAT PARQUET, COMPRESSION {compression_sql})
            """
        )
    finally:
        connection.close()

    return SeriesExtractionResult(
        series_ticker=series_ticker,
        archive_root=resolved_archive_root,
        output_dir=resolved_output_dir,
        markets_parquet=markets_parquet,
        trades_parquet=trades_parquet,
        markets_count=markets_count,
        trades_count=trades_count,
    )
