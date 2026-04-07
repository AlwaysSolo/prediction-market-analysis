from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERIES = "KXBTC15M"
DEFAULT_MARKETS_PATH = REPO_ROOT / "output" / "kalshi_series_backfill" / f"{DEFAULT_SERIES}_markets.parquet"
DEFAULT_TRADES_PATH = REPO_ROOT / "output" / "kalshi_series_backfill" / f"{DEFAULT_SERIES}_trades.parquet"
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("magnetism_penalty_matrix.csv")


def build_query(
    markets_path: Path,
    trades_path: Path,
    tau_bucket_minutes: float,
    distance_bucket_cents: float,
    min_trades: int,
) -> str:
    return f"""
    WITH joined AS (
        SELECT
            t.trade_id,
            t.ticker,
            CAST(t.created_time AS TIMESTAMPTZ) AS created_time,
            CAST(t.yes_price AS DOUBLE) AS yes_price_cents,
            LOWER(CAST(m.result AS VARCHAR)) AS result,
            CAST(m.close_time AS TIMESTAMPTZ) AS close_time
        FROM read_parquet('{trades_path.as_posix()}') AS t
        INNER JOIN read_parquet('{markets_path.as_posix()}') AS m USING (ticker)
        WHERE t.yes_price IS NOT NULL
          AND m.close_time IS NOT NULL
          AND LOWER(CAST(m.result AS VARCHAR)) IN ('yes', 'no')
    ),
    base AS (
        SELECT
            trade_id,
            ticker,
            CASE WHEN result = 'yes' THEN 1 ELSE 0 END AS actual_outcome,
            yes_price_cents / 100.0 AS market_prob,
            DATE_DIFF('second', created_time, close_time) / 60.0 AS tau_minutes,
            ABS(yes_price_cents - 50.0) AS distance_from_mid
        FROM joined
        WHERE created_time IS NOT NULL
    ),
    binned AS (
        SELECT
            ROUND(tau_minutes / {tau_bucket_minutes}, 0) * {tau_bucket_minutes} AS tau_bin,
            ROUND(distance_from_mid / {distance_bucket_cents}, 0) * {distance_bucket_cents} AS distance_bin,
            market_prob,
            actual_outcome
        FROM base
        WHERE tau_minutes > 0
    )
    SELECT
        tau_bin,
        distance_bin,
        COUNT(*) AS trade_count,
        AVG(market_prob) AS avg_implied_prob,
        AVG(actual_outcome) AS actual_win_rate,
        AVG(market_prob) - AVG(actual_outcome) AS magnetism_penalty
    FROM binned
    GROUP BY tau_bin, distance_bin
    HAVING COUNT(*) >= {min_trades}
    ORDER BY tau_bin, distance_bin
    """


def run_analysis(
    markets_path: Path,
    trades_path: Path,
    output_path: Path,
    tau_bucket_minutes: float,
    distance_bucket_cents: float,
    min_trades: int,
) -> None:
    if not markets_path.exists():
        raise FileNotFoundError(f"Markets parquet not found: {markets_path}")
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades parquet not found: {trades_path}")

    print("Loading Kalshi trades for strike magnetism analysis...")
    print(f"Markets: {markets_path}")
    print(f"Trades: {trades_path}")

    query = build_query(
        markets_path=markets_path,
        trades_path=trades_path,
        tau_bucket_minutes=tau_bucket_minutes,
        distance_bucket_cents=distance_bucket_cents,
        min_trades=min_trades,
    )

    con = duckdb.connect()

    base_stats = con.execute(
        f"""
        SELECT
            COUNT(*) AS trade_rows,
            MIN(CAST(created_time AS TIMESTAMPTZ)) AS first_trade_time,
            MAX(CAST(created_time AS TIMESTAMPTZ)) AS last_trade_time
        FROM read_parquet('{trades_path.as_posix()}')
        """
    ).fetchone()
    print(f"Trade rows scanned: {base_stats[0]:,}")
    print(f"Trade window: {base_stats[1]} -> {base_stats[2]}")

    print("Calculating penalty matrix...")
    analysis = con.execute(query).fetchdf()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(output_path, index=False)
    print(f"Saved penalty matrix to {output_path}")
    print(f"Matrix rows: {len(analysis):,}")

    if analysis.empty:
        print("No buckets met the minimum trade count threshold.")
        con.close()
        return

    print("\nHighest penalties (market overpriced relative to realized win rate):")
    print(analysis.sort_values("magnetism_penalty", ascending=False).head(10).to_string(index=False))

    print("\nMost negative penalties (strongest realized magnetism):")
    print(analysis.sort_values("magnetism_penalty", ascending=True).head(10).to_string(index=False))

    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate strike magnetism buckets for Kalshi trades.")
    parser.add_argument("--markets-path", default=str(DEFAULT_MARKETS_PATH), help="Path to markets parquet")
    parser.add_argument("--trades-path", default=str(DEFAULT_TRADES_PATH), help="Path to merged trades parquet")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="CSV output path")
    parser.add_argument(
        "--tau-bucket-minutes",
        type=float,
        default=1.0,
        help="Bucket width for time-to-close in minutes",
    )
    parser.add_argument(
        "--distance-bucket-cents",
        type=float,
        default=5.0,
        help="Bucket width for distance from the 50-cent strike, measured in cents",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=1000,
        help="Minimum trades required for a bucket to be kept",
    )
    args = parser.parse_args()

    run_analysis(
        markets_path=Path(args.markets_path),
        trades_path=Path(args.trades_path),
        output_path=Path(args.output_path),
        tau_bucket_minutes=args.tau_bucket_minutes,
        distance_bucket_cents=args.distance_bucket_cents,
        min_trades=args.min_trades,
    )


if __name__ == "__main__":
    main()
