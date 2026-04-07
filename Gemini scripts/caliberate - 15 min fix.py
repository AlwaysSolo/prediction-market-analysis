from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERIES = "KXBTC15M"
DEFAULT_OUTPUT = Path(__file__).with_name("result.md")


def resolve_default_inputs(series_ticker: str) -> tuple[Path, Path]:
    candidates = [
        REPO_ROOT / "output" / "kalshi_series_backfill",
        REPO_ROOT / "output" / "kalshi_series",
    ]

    for base in candidates:
        markets_path = base / f"{series_ticker}_markets.parquet"
        trades_path = base / series_ticker / "trades"
        if markets_path.exists() and trades_path.exists():
            return markets_path, trades_path

    raise FileNotFoundError(
        f"Could not find series files for {series_ticker}. "
        "Pass --markets-path and --trades-path explicitly."
    )


def resolve_inputs(
    series_ticker: str,
    markets_path: str | None,
    trades_path: str | None,
) -> tuple[Path, Path]:
    if markets_path and trades_path:
        return Path(markets_path), Path(trades_path)

    if markets_path or trades_path:
        raise ValueError("Pass both --markets-path and --trades-path together.")

    return resolve_default_inputs(series_ticker)


def collect_trade_files(trades_path: Path) -> tuple[list[str], int]:
    if trades_path.is_file():
        return [str(trades_path)], 0

    if not trades_path.exists():
        raise FileNotFoundError(f"Trades path not found: {trades_path}")

    valid_files: list[str] = []
    skipped_files = 0

    for file_path in sorted(trades_path.glob("*.parquet")):
        try:
            metadata = pq.read_metadata(file_path)
        except Exception as exc:
            print(f"Skipping unreadable trade file {file_path.name}: {exc}")
            skipped_files += 1
            continue

        if metadata.num_rows == 0 or metadata.num_columns == 0:
            skipped_files += 1
            continue

        valid_files.append(str(file_path))

    if not valid_files:
        raise FileNotFoundError(f"No non-empty parquet trade files found under {trades_path}")

    return valid_files, skipped_files


def register_markets_view(con: duckdb.DuckDBPyConnection, view_name: str, markets_path: Path) -> None:
    suffix = markets_path.suffix.lower()
    if suffix == ".parquet":
        con.from_parquet(str(markets_path)).create_view(view_name, replace=True)
        return

    if suffix == ".csv":
        con.from_csv_auto(str(markets_path), header=True).create_view(view_name, replace=True)
        return

    raise ValueError(f"Unsupported markets file type: {markets_path}")


def build_report(
    series_ticker: str,
    analysis_rows,
    total_points: int,
) -> str:
    lines: list[str] = [
        f"# {series_ticker} Pure Calibration Report",
        "",
        f"- **Target Product:** {series_ticker} (Native 15m Binaries)",
        f"- **Total Trades Analyzed:** {total_points:,}",
        "- **Goal:** Systematic bias correction for Gaussian model drift.",
        "",
        "## TTE Calibration Table",
        "",
        "| TTE (min) | Market Prob | Realized Win% | Bias (Rust Offset) | Sample Size |",
        "|-----------|-------------|---------------|-------------------|-------------|",
    ]

    for row in analysis_rows:
        bias = row["realized_win_rate"] - row["avg_market_price"]
        lines.append(
            f"| {row['tte_bucket']:>9} | "
            f"{row['avg_market_price']:>11.4f} | "
            f"{row['realized_win_rate']:>13.4f} | "
            f"{bias:>+17.4f} | "
            f"{row['sample_size']:>11,} |"
        )

    lines.extend(
        [
            "",
            "## Rust Implementation Snippet",
            "Copy this into `src/domain/calibration.rs` or directly into the lookup logic:",
            "",
            "```rust",
            "let calibration_offset = match tau_minutes.round() as i32 {",
        ]
    )

    for row in analysis_rows:
        bias = row["realized_win_rate"] - row["avg_market_price"]
        lines.append(f"    {row['tte_bucket']} => {bias:.4f},")

    lines.extend(
        [
            "    _ => 0.0,",
            "};",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def run_calibration(
    series_ticker: str,
    markets_path: Path,
    trades_path: Path,
    output_path: Path,
    min_sample_size: int,
) -> None:
    trade_files, skipped_files = collect_trade_files(trades_path)

    print(f"Using markets file: {markets_path}")
    print(f"Using trades path: {trades_path}")
    print(f"Loaded {len(trade_files):,} non-empty trade parquet files")
    if skipped_files:
        print(f"Skipped {skipped_files:,} empty or unreadable trade parquet files")

    con = duckdb.connect()
    register_markets_view(con, "markets_source", markets_path)
    con.from_parquet(trade_files, union_by_name=True).create_view("trades_source", replace=True)

    query_prefix = f"""
        WITH target_markets AS (
            SELECT
                ticker,
                LOWER(CAST(result AS VARCHAR)) AS result,
                CAST(close_time AS TIMESTAMPTZ) AS close_time
            FROM markets_source
            WHERE ticker LIKE '{series_ticker}-%'
        ),
        trade_points AS (
            SELECT
                t.ticker,
                CAST(t.created_time AS TIMESTAMPTZ) AS created_time,
                CAST(t.yes_price AS DOUBLE) / 100.0 AS market_prob,
                CASE WHEN m.result = 'yes' THEN 1 ELSE 0 END AS actual_outcome,
                DATE_DIFF(
                    'second',
                    CAST(t.created_time AS TIMESTAMPTZ),
                    m.close_time
                ) / 60.0 AS tau_minutes
            FROM trades_source t
            INNER JOIN target_markets m USING (ticker)
            WHERE t.yes_price IS NOT NULL
        )
    """

    total_points = con.execute(
        query_prefix
        + """
        SELECT COUNT(*)
        FROM trade_points
        WHERE tau_minutes BETWEEN 0 AND 16
        """
    ).fetchone()[0]

    analysis = con.execute(
        query_prefix
        + f"""
        SELECT
            CAST(ROUND(tau_minutes) AS INTEGER) AS tte_bucket,
            AVG(actual_outcome) AS realized_win_rate,
            AVG(market_prob) AS avg_market_price,
            COUNT(*) AS sample_size
        FROM trade_points
        WHERE tau_minutes BETWEEN 0 AND 16
        GROUP BY 1
        HAVING COUNT(*) > {min_sample_size}
        ORDER BY 1
        """
    ).fetchdf()

    print(f"Analyzing {total_points:,} trades in the 0-16 minute window")
    print(f"Computed {len(analysis):,} calibration buckets")

    report = build_report(series_ticker, analysis.to_dict("records"), total_points)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibration on a Kalshi series backfill.")
    parser.add_argument("--series", default=DEFAULT_SERIES, help="Series ticker prefix, e.g. KXBTC15M")
    parser.add_argument("--markets-path", help="Path to the series markets parquet/csv file")
    parser.add_argument("--trades-path", help="Path to a trades directory or trades parquet file")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Markdown report output path",
    )
    parser.add_argument(
        "--min-sample-size",
        type=int,
        default=50,
        help="Minimum samples required for a TTE bucket to appear in the report",
    )
    args = parser.parse_args()

    markets_path, trades_path = resolve_inputs(args.series, args.markets_path, args.trades_path)
    run_calibration(
        series_ticker=args.series,
        markets_path=markets_path,
        trades_path=trades_path,
        output_path=Path(args.output),
        min_sample_size=args.min_sample_size,
    )


if __name__ == "__main__":
    main()
