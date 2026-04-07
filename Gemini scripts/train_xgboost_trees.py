from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERIES = "KXBTC15M"
DEFAULT_SAMPLE_SIZE = 500_000
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("xgboost_output")
RANDOM_SEED = 42


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


def stratified_sample_indices(actuals: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or len(actuals) <= sample_size:
        return np.arange(len(actuals))

    rng = np.random.default_rng(seed)
    positive_idx = np.flatnonzero(actuals == 1)
    negative_idx = np.flatnonzero(actuals == 0)

    if len(positive_idx) == 0 or len(negative_idx) == 0:
        sampled = rng.choice(len(actuals), size=sample_size, replace=False)
        sampled.sort()
        return sampled

    positive_target = int(round(sample_size * (len(positive_idx) / len(actuals))))
    positive_target = min(max(positive_target, 1), len(positive_idx))
    negative_target = min(sample_size - positive_target, len(negative_idx))

    if positive_target + negative_target < sample_size:
        remaining = sample_size - (positive_target + negative_target)
        if len(positive_idx) - positive_target >= len(negative_idx) - negative_target:
            positive_target = min(positive_target + remaining, len(positive_idx))
        else:
            negative_target = min(negative_target + remaining, len(negative_idx))

    sampled_positive = rng.choice(positive_idx, size=positive_target, replace=False)
    sampled_negative = rng.choice(negative_idx, size=negative_target, replace=False)
    sampled = np.concatenate([sampled_positive, sampled_negative])
    rng.shuffle(sampled)
    return sampled


def load_feature_frame(
    series_ticker: str,
    markets_path: Path,
    trades_path: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    trade_files, skipped_files = collect_trade_files(trades_path)

    print(f"Using markets file: {markets_path}")
    print(f"Using trades path: {trades_path}")
    print(f"Loaded {len(trade_files):,} non-empty trade parquet files")
    if skipped_files:
        print(f"Skipped {skipped_files:,} empty or unreadable trade parquet files")

    con = duckdb.connect()
    register_markets_view(con, "markets_source", markets_path)
    con.from_parquet(trade_files, union_by_name=True).create_view("trades_source", replace=True)

    df = con.execute(
        f"""
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
                LEAST(GREATEST(CAST(t.yes_price AS DOUBLE) / 100.0, 0.01), 0.99) AS market_prob,
                CASE WHEN m.result = 'yes' THEN 1 ELSE 0 END AS actual_outcome,
                DATE_DIFF(
                    'second',
                    CAST(t.created_time AS TIMESTAMPTZ),
                    m.close_time
                ) / 60.0 AS tau_minutes
            FROM trades_source t
            INNER JOIN target_markets m USING (ticker)
            WHERE t.yes_price IS NOT NULL
        ),
        filtered AS (
            SELECT
                ticker,
                created_time,
                market_prob,
                actual_outcome,
                tau_minutes,
                COALESCE(
                    market_prob - LAG(market_prob) OVER (
                        PARTITION BY ticker
                        ORDER BY created_time
                    ),
                    0.0
                ) AS price_momentum
            FROM trade_points
            WHERE tau_minutes > 0 AND tau_minutes <= 15
        )
        SELECT
            actual_outcome,
            market_prob,
            tau_minutes,
            price_momentum
        FROM filtered
        """
    ).fetchdf()

    print(f"Loaded {len(df):,} feature rows from the series backfill")

    features = pd.DataFrame(
        {
            "market_prob": df["market_prob"].astype(float),
            "tau_minutes": df["tau_minutes"].astype(float),
            "price_momentum": df["price_momentum"].astype(float),
        }
    )
    features["abs_price_momentum"] = features["price_momentum"].abs()
    features["price_direction"] = np.sign(features["price_momentum"]).astype(float)
    features["distance_from_mid"] = (features["market_prob"] - 0.5).abs()
    features["z_implied"] = norm.ppf(features["market_prob"])

    actuals = df["actual_outcome"].to_numpy(dtype=np.int8)
    return features, actuals


def train_xgboost(
    features: pd.DataFrame,
    actuals: np.ndarray,
    sample_size: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
) -> tuple[XGBClassifier, dict, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    sample_indices = stratified_sample_indices(actuals, sample_size, RANDOM_SEED)
    features_sampled = features.iloc[sample_indices].reset_index(drop=True)
    actuals_sampled = actuals[sample_indices]

    print(f"Training sample size: {len(features_sampled):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        features_sampled,
        actuals_sampled,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=actuals_sampled,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    test_pred = model.predict_proba(X_test)[:, 1]
    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "sample_rows": int(len(features_sampled)),
        "feature_names": list(features_sampled.columns),
        "log_loss": float(log_loss(y_test, test_pred)),
        "brier_score": float(brier_score_loss(y_test, test_pred)),
        "roc_auc": float(roc_auc_score(y_test, test_pred)),
        "positive_rate_test": float(np.mean(y_test)),
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "tree_method": "hist",
            "random_seed": RANDOM_SEED,
        },
    }

    return model, metrics, X_train, X_test, y_train, y_test


def write_outputs(
    model: XGBClassifier,
    metrics: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    booster = model.get_booster()
    model_path = output_dir / "model.json"
    trees_txt_path = output_dir / "trees.txt"
    trees_json_path = output_dir / "trees.json"
    trees_csv_path = output_dir / "trees.csv"
    importance_path = output_dir / "feature_importance.json"
    metrics_path = output_dir / "metrics.json"

    booster.save_model(model_path)

    text_dump = booster.get_dump(with_stats=True, dump_format="text")
    trees_txt_path.write_text(
        "\n\n".join(f"booster[{i}]\n{tree}" for i, tree in enumerate(text_dump)),
        encoding="utf-8",
    )

    json_dump = booster.get_dump(with_stats=True, dump_format="json")
    trees_json_path.write_text("[\n" + ",\n".join(json_dump) + "\n]\n", encoding="utf-8")

    trees_df = booster.trees_to_dataframe()
    trees_df.to_csv(trees_csv_path, index=False)

    importance = {
        "gain": booster.get_score(importance_type="gain"),
        "weight": booster.get_score(importance_type="weight"),
        "cover": booster.get_score(importance_type="cover"),
    }
    importance_path.write_text(json.dumps(importance, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to {model_path}")
    print(f"Saved text tree dump to {trees_txt_path}")
    print(f"Saved JSON tree dump to {trees_json_path}")
    print(f"Saved tree table to {trees_csv_path}")
    print(f"Saved feature importance to {importance_path}")
    print(f"Saved metrics to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost model on a Kalshi series backfill.")
    parser.add_argument("--series", default=DEFAULT_SERIES, help="Series ticker prefix, e.g. KXBTC15M")
    parser.add_argument("--markets-path", help="Path to the series markets parquet/csv file")
    parser.add_argument("--trades-path", help="Path to a trades directory or trades parquet file")
    parser.add_argument("--output-dir", help="Directory for artifacts")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Stratified sample size used for training (0 uses all rows)",
    )
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of boosting rounds")
    parser.add_argument("--max-depth", type=int, default=4, help="Tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Boosting learning rate")
    parser.add_argument("--subsample", type=float, default=0.9, help="Row subsample rate per tree")
    parser.add_argument("--colsample-bytree", type=float, default=0.9, help="Column subsample rate per tree")
    args = parser.parse_args()

    markets_path, trades_path = resolve_inputs(args.series, args.markets_path, args.trades_path)
    output_dir = Path(args.output_dir) if args.output_dir else (DEFAULT_OUTPUT_DIR / args.series)

    features, actuals = load_feature_frame(
        series_ticker=args.series,
        markets_path=markets_path,
        trades_path=trades_path,
    )
    model, metrics, *_ = train_xgboost(
        features=features,
        actuals=actuals,
        sample_size=args.sample_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
    )
    write_outputs(model=model, metrics=metrics, output_dir=output_dir)

    print("\nMetrics")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
