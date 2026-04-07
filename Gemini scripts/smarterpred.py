from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import norm, t

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier, early_stopping
except ImportError:
    LGBMClassifier = None
    early_stopping = None

try:
    from pysr import PySRRegressor
except ImportError:
    PySRRegressor = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
except ImportError:
    LogisticRegression = None
    train_test_split = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERIES = "KXBTC15M"
DEFAULT_SAMPLE_ROWS = 1_000_000
DEFAULT_ARTIFACTS_DIR = Path(__file__).with_name("model_tournament_output")
ML_RANDOM_SEED = 42
MODEL_NAME_TO_LABEL = {
    "1. Standard Gaussian (The Market)": "Standard Gaussian",
    "2. Fat Tails (Student-T, df=5)": "Student-T df=5",
    "3. Extreme Fat Tails (Student-T, df=3)": "Student-T df=3",
    "4. Momentum-Adjusted Gaussian": "Momentum-Adjusted Gaussian",
    "5. Machine Learning (Logistic Regression)": "Machine Learning (Logistic Regression)",
    "6. Gradient Boosted Trees (XGBoost)": "XGBoost",
    "7. LightGBM": "LightGBM",
    "8. CatBoost": "CatBoost",
    "9. PySR (Symbolic Regression)": "PySR (Symbolic Regression)",
}

warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted")
warnings.filterwarnings("ignore", message="Note: Setting `random_state` without also setting `deterministic=True`")


def binary_log_loss(actuals: np.ndarray, predictions: np.ndarray) -> float:
    clipped = np.clip(predictions, 1e-12, 1 - 1e-12)
    return float(-np.mean(actuals * np.log(clipped) + (1 - actuals) * np.log(1 - clipped)))


def clean_probabilities(predictions: np.ndarray) -> np.ndarray:
    cleaned = np.nan_to_num(predictions, nan=0.5, posinf=0.99, neginf=0.01)
    return np.clip(cleaned.astype(np.float64, copy=False), 0.01, 0.99)


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


def build_feature_matrix(
    market_probs: np.ndarray,
    tau: np.ndarray,
    momentum: np.ndarray,
) -> np.ndarray:
    z_implied = norm.ppf(market_probs)
    z_clean = np.nan_to_num(z_implied, posinf=3.0, neginf=-3.0)
    abs_momentum = np.abs(momentum)
    price_direction = np.sign(momentum)
    distance_from_mid = np.abs(market_probs - 0.5)
    return np.column_stack(
        (z_clean, tau, momentum, abs_momentum, price_direction, distance_from_mid)
    ).astype(np.float32, copy=False)


def feature_names() -> list[str]:
    return [
        "z_implied",
        "tau_minutes",
        "price_momentum",
        "abs_price_momentum",
        "price_direction",
        "distance_from_mid",
    ]


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


def resolve_artifacts_dir(artifacts_dir: str | None, xgb_output_dir: str | None, series_ticker: str) -> Path:
    if artifacts_dir and xgb_output_dir and Path(artifacts_dir) != Path(xgb_output_dir):
        raise ValueError("Pass either --artifacts-dir or --xgb-output-dir, not both with different values.")

    if artifacts_dir:
        return Path(artifacts_dir)

    if xgb_output_dir:
        return Path(xgb_output_dir)

    return DEFAULT_ARTIFACTS_DIR / f"{series_ticker}_tournament"


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


def load_feature_frame(
    series_ticker: str,
    markets_path: Path,
    trades_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    con.close()

    print(f"Processing {len(df):,} trades through the model tournament")

    actuals = df["actual_outcome"].to_numpy(dtype=np.int8)
    market_probs = df["market_prob"].to_numpy(dtype=np.float32)
    tau = df["tau_minutes"].to_numpy(dtype=np.float32)
    momentum = df["price_momentum"].to_numpy(dtype=np.float32)
    del df

    return actuals, market_probs, tau, momentum


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def write_summary_markdown(
    path: Path,
    title: str,
    results: list[tuple[str, float]],
    metadata: dict[str, object],
) -> None:
    lines = [f"# {title}", ""]
    lines.append(f"- Series: `{metadata['series']}`")
    lines.append(f"- Rows used: `{metadata['sample_rows']:,}`")
    lines.append(f"- Test rows: `{metadata['test_rows']:,}`")
    lines.append("")
    lines.append("| Model | Log-Loss |")
    lines.append("| --- | ---: |")
    for name, loss in results:
        lines.append(f"| {MODEL_NAME_TO_LABEL[name]} | {loss:.5f} |")
    best_name, best_loss = min(results, key=lambda item: item[1])
    lines.extend(["", f"Best predictor: **{MODEL_NAME_TO_LABEL[best_name]} ({best_loss:.5f})**", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tournament_outputs(
    artifacts_dir: Path,
    results: list[tuple[str, float]],
    metadata: dict[str, object],
) -> None:
    tournament_payload = {
        "metadata": metadata,
        "results": [{"model": name, "log_loss": loss} for name, loss in results],
        "best_model": min(results, key=lambda item: item[1])[0],
    }
    write_json(artifacts_dir / "tournament_results.json", tournament_payload)
    write_summary_markdown(
        artifacts_dir / "summary.md",
        title="Model Tournament Results",
        results=results,
        metadata=metadata,
    )


def save_feature_importance(path: Path, names: list[str], values: np.ndarray) -> None:
    payload = {name: float(value) for name, value in zip(names, values)}
    write_json(path, payload)


def write_xgboost_artifacts(model: XGBClassifier, output_dir: Path, metrics: dict[str, object]) -> None:
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
    write_json(importance_path, importance)
    write_json(metrics_path, metrics)

    print(f"Saved XGBoost model to {model_path}")


def write_lightgbm_artifacts(
    model: LGBMClassifier,
    output_dir: Path,
    feature_names_list: list[str],
    metrics: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(output_dir / "model.txt"))
    save_feature_importance(output_dir / "feature_importance.json", feature_names_list, model.feature_importances_)
    write_json(output_dir / "metrics.json", metrics)
    print(f"Saved LightGBM model to {output_dir / 'model.txt'}")


def write_catboost_artifacts(
    model: CatBoostClassifier,
    output_dir: Path,
    feature_names_list: list[str],
    metrics: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "model.cbm"))
    save_feature_importance(
        output_dir / "feature_importance.json",
        feature_names_list,
        np.asarray(model.get_feature_importance()),
    )
    write_json(output_dir / "metrics.json", metrics)
    print(f"Saved CatBoost model to {output_dir / 'model.cbm'}")


def write_pysr_artifacts(
    model: PySRRegressor,
    output_dir: Path,
    metrics: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    equations_path = output_dir / "equations.csv"
    best_equation_path = output_dir / "best_equation.txt"
    metrics_path = output_dir / "metrics.json"

    model.equations_.to_csv(equations_path, index=False)
    best_equation_path.write_text(str(model.sympy()), encoding="utf-8")
    write_json(metrics_path, metrics)

    run_directory = Path(model.output_directory_) / model.run_id_
    for source_name in ("hall_of_fame.csv", "hall_of_fame.csv.bak"):
        source = run_directory / source_name
        if source.exists():
            shutil.copy2(source, output_dir / source_name)

    print(f"Saved PySR equations to {equations_path}")


def split_dataset(
    feature_matrix: np.ndarray,
    actuals: np.ndarray,
    market_probs: np.ndarray,
    tau: np.ndarray,
    momentum: np.ndarray,
) -> dict[str, np.ndarray]:
    if train_test_split is None:
        raise RuntimeError("scikit-learn is required to run the tournament split.")

    (
        X_train_val,
        X_test,
        y_train_val,
        y_test,
        market_prob_train_val,
        market_prob_test,
        tau_train_val,
        tau_test,
        momentum_train_val,
        momentum_test,
    ) = train_test_split(
        feature_matrix,
        actuals,
        market_probs,
        tau,
        momentum,
        test_size=0.2,
        random_state=ML_RANDOM_SEED,
        stratify=actuals,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        random_state=ML_RANDOM_SEED,
        stratify=y_train_val,
    )

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "X_train_val": X_train_val,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "market_prob_test": market_prob_test,
        "tau_test": tau_test,
        "momentum_test": momentum_test,
        "train_rows": len(X_train),
        "validation_rows": len(X_valid),
        "test_rows": len(X_test),
    }


def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[LGBMClassifier, dict[str, object]]:
    candidate_params = [
        {
            "n_estimators": 1200,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_samples": 200,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 1600,
            "learning_rate": 0.02,
            "num_leaves": 127,
            "max_depth": -1,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_samples": 300,
            "reg_lambda": 2.0,
        },
    ]

    best_model = None
    best_loss = None
    best_params = None
    trial_results = []

    for idx, params in enumerate(candidate_params, start=1):
        print(f"Training LightGBM candidate {idx}/{len(candidate_params)}...")
        model = LGBMClassifier(
            objective="binary",
            random_state=ML_RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
        callbacks = [early_stopping(100, verbose=False)] if early_stopping is not None else None
        fit_kwargs = {
            "eval_set": [(X_valid, y_valid)],
            "eval_metric": "binary_logloss",
        }
        if callbacks is not None:
            fit_kwargs["callbacks"] = callbacks

        model.fit(X_train, y_train, **fit_kwargs)
        valid_prob = clean_probabilities(model.predict_proba(X_valid)[:, 1])
        valid_loss = binary_log_loss(y_valid, valid_prob)
        trial_results.append({"candidate": idx, "params": params, "validation_log_loss": valid_loss})

        if best_loss is None or valid_loss < best_loss:
            best_model = model
            best_loss = valid_loss
            best_params = params

    assert best_model is not None and best_loss is not None and best_params is not None

    metrics = {
        "validation_log_loss": best_loss,
        "best_iteration": int(getattr(best_model, "best_iteration_", -1) or -1),
        "best_params": best_params,
        "candidates": trial_results,
    }
    return best_model, metrics


def tune_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[CatBoostClassifier, dict[str, object]]:
    candidate_params = [
        {
            "iterations": 1500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
        },
        {
            "iterations": 2000,
            "depth": 8,
            "learning_rate": 0.02,
            "l2_leaf_reg": 5.0,
        },
    ]

    best_model = None
    best_loss = None
    best_params = None
    trial_results = []

    for idx, params in enumerate(candidate_params, start=1):
        print(f"Training CatBoost candidate {idx}/{len(candidate_params)}...")
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=ML_RANDOM_SEED,
            verbose=False,
            thread_count=-1,
            **params,
        )
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True, verbose=False)
        valid_prob = clean_probabilities(model.predict_proba(X_valid)[:, 1])
        valid_loss = binary_log_loss(y_valid, valid_prob)
        trial_results.append({"candidate": idx, "params": params, "validation_log_loss": valid_loss})

        if best_loss is None or valid_loss < best_loss:
            best_model = model
            best_loss = valid_loss
            best_params = params

    assert best_model is not None and best_loss is not None and best_params is not None

    metrics = {
        "validation_log_loss": best_loss,
        "best_iteration": int(best_model.get_best_iteration()),
        "best_params": best_params,
        "candidates": trial_results,
    }
    return best_model, metrics


def tune_pysr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    output_dir: Path,
) -> tuple[PySRRegressor, dict[str, object]]:
    candidate_params = [
        {
            "niterations": 10,
            "populations": 6,
            "population_size": 24,
            "maxsize": 18,
            "timeout_in_seconds": 300,
        },
        {
            "niterations": 14,
            "populations": 8,
            "population_size": 30,
            "maxsize": 22,
            "timeout_in_seconds": 420,
        },
    ]

    best_model = None
    best_loss = None
    best_params = None
    trial_results = []

    for idx, params in enumerate(candidate_params, start=1):
        print(f"Training PySR candidate {idx}/{len(candidate_params)}...")
        model = PySRRegressor(
            model_selection="best",
            binary_operators=["+", "-", "*", "/"],
            ncycles_per_iteration=200,
            batching=True,
            batch_size=100_000,
            precision=32,
            progress=False,
            verbosity=0,
            parallelism="multithreading",
            random_state=ML_RANDOM_SEED + idx,
            deterministic=False,
            output_directory=str(output_dir),
            run_id=f"candidate_{idx:02d}",
            **params,
        )
        model.fit(X_train, y_train.astype(np.float32, copy=False))
        valid_prob = clean_probabilities(model.predict(X_valid))
        valid_loss = binary_log_loss(y_valid, valid_prob)
        trial_results.append({"candidate": idx, "params": params, "validation_log_loss": valid_loss})

        if best_loss is None or valid_loss < best_loss:
            best_model = model
            best_loss = valid_loss
            best_params = params

    assert best_model is not None and best_loss is not None and best_params is not None

    metrics = {
        "validation_log_loss": best_loss,
        "best_params": best_params,
        "candidates": trial_results,
    }
    return best_model, metrics


def run_tournament(
    series_ticker: str,
    markets_path: Path,
    trades_path: Path,
    sample_size: int,
    artifacts_dir: Path,
) -> None:
    print("Loading data for the model tournament...")
    actuals, market_probs, tau, momentum = load_feature_frame(
        series_ticker=series_ticker,
        markets_path=markets_path,
        trades_path=trades_path,
    )

    eval_indices = stratified_sample_indices(actuals, sample_size, ML_RANDOM_SEED)
    using_sampled_tournament = len(eval_indices) < len(actuals)
    if using_sampled_tournament:
        print(
            f"Using a stratified sample of {len(eval_indices):,} rows for the model tournament "
            "to keep the tournament practical."
        )
    else:
        print(f"Using all {len(eval_indices):,} available rows for the model tournament.")

    actuals_eval = actuals[eval_indices]
    market_probs_eval = market_probs[eval_indices]
    tau_eval = tau[eval_indices]
    momentum_eval = momentum[eval_indices]

    feature_matrix = build_feature_matrix(market_probs_eval, tau_eval, momentum_eval)
    splits = split_dataset(feature_matrix, actuals_eval, market_probs_eval, tau_eval, momentum_eval)
    X_train = splits["X_train"]
    X_valid = splits["X_valid"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_valid = splits["y_valid"]
    y_test = splits["y_test"]
    market_prob_test = splits["market_prob_test"]
    momentum_test = splits["momentum_test"]

    reset_directory(artifacts_dir)
    feature_names_list = feature_names()

    z_implied_test = norm.ppf(market_prob_test)
    prob_gaussian = clean_probabilities(market_prob_test)
    prob_t5 = clean_probabilities(t.cdf(z_implied_test, df=5))
    prob_t3 = clean_probabilities(t.cdf(z_implied_test, df=3))
    prob_momentum = clean_probabilities(prob_gaussian + (np.sign(momentum_test) * 0.02))

    prob_ml_lr = None
    if LogisticRegression is None:
        print("\nSkipping ML model: scikit-learn is not installed in this environment.")
    else:
        print("\nTraining Machine Learning Model (Logistic Regression)...")
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, y_train)
        prob_ml_lr = clean_probabilities(lr_model.predict_proba(X_test)[:, 1])

    prob_xgb = None
    if XGBClassifier is None:
        print("\nSkipping XGBoost: xgboost is not installed in this environment.")
    else:
        print("Training Gradient Boosted Trees (XGBoost)...")
        xgb_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=2000,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=ML_RANDOM_SEED,
            n_jobs=-1,
            early_stopping_rounds=50,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        prob_xgb = clean_probabilities(xgb_model.predict_proba(X_test)[:, 1])
        xgb_metrics = {
            "series": series_ticker,
            "sample_rows": int(len(actuals_eval)),
            "train_rows": int(splits["train_rows"]),
            "validation_rows": int(splits["validation_rows"]),
            "test_rows": int(splits["test_rows"]),
            "feature_names": feature_names_list,
            "best_iteration": int(getattr(xgb_model, "best_iteration", -1)),
            "best_score": (
                float(getattr(xgb_model, "best_score"))
                if getattr(xgb_model, "best_score", None) is not None
                else None
            ),
            "log_loss_test": float(binary_log_loss(y_test, prob_xgb)),
            "params": xgb_model.get_xgb_params(),
        }
        write_xgboost_artifacts(xgb_model, artifacts_dir / "xgboost", xgb_metrics)

    prob_lgbm = None
    if LGBMClassifier is None:
        print("\nSkipping LightGBM: lightgbm is not installed in this environment.")
    else:
        lgbm_model, lgbm_metrics = tune_lightgbm(X_train, y_train, X_valid, y_valid)
        prob_lgbm = clean_probabilities(lgbm_model.predict_proba(X_test)[:, 1])
        lgbm_metrics.update(
            {
                "series": series_ticker,
                "sample_rows": int(len(actuals_eval)),
                "train_rows": int(splits["train_rows"]),
                "validation_rows": int(splits["validation_rows"]),
                "test_rows": int(splits["test_rows"]),
                "feature_names": feature_names_list,
                "log_loss_test": float(binary_log_loss(y_test, prob_lgbm)),
            }
        )
        write_lightgbm_artifacts(lgbm_model, artifacts_dir / "lightgbm", feature_names_list, lgbm_metrics)

    prob_catboost = None
    if CatBoostClassifier is None:
        print("\nSkipping CatBoost: catboost is not installed in this environment.")
    else:
        catboost_model, catboost_metrics = tune_catboost(X_train, y_train, X_valid, y_valid)
        prob_catboost = clean_probabilities(catboost_model.predict_proba(X_test)[:, 1])
        catboost_metrics.update(
            {
                "series": series_ticker,
                "sample_rows": int(len(actuals_eval)),
                "train_rows": int(splits["train_rows"]),
                "validation_rows": int(splits["validation_rows"]),
                "test_rows": int(splits["test_rows"]),
                "feature_names": feature_names_list,
                "log_loss_test": float(binary_log_loss(y_test, prob_catboost)),
            }
        )
        write_catboost_artifacts(
            catboost_model,
            artifacts_dir / "catboost",
            feature_names_list,
            catboost_metrics,
        )

    prob_pysr = None
    if PySRRegressor is None:
        print("\nSkipping PySR: pysr is not installed in this environment.")
    else:
        pysr_model, pysr_metrics = tune_pysr(X_train, y_train, X_valid, y_valid, artifacts_dir / "pysr_runs")
        prob_pysr = clean_probabilities(pysr_model.predict(X_test))
        pysr_metrics.update(
            {
                "series": series_ticker,
                "sample_rows": int(len(actuals_eval)),
                "train_rows": int(splits["train_rows"]),
                "validation_rows": int(splits["validation_rows"]),
                "test_rows": int(splits["test_rows"]),
                "feature_names": feature_names_list,
                "log_loss_test": float(binary_log_loss(y_test, prob_pysr)),
            }
        )
        write_pysr_artifacts(pysr_model, artifacts_dir / "pysr", pysr_metrics)

    models = {
        "1. Standard Gaussian (The Market)": prob_gaussian,
        "2. Fat Tails (Student-T, df=5)": prob_t5,
        "3. Extreme Fat Tails (Student-T, df=3)": prob_t3,
        "4. Momentum-Adjusted Gaussian": prob_momentum,
        "5. Machine Learning (Logistic Regression)": prob_ml_lr,
        "6. Gradient Boosted Trees (XGBoost)": prob_xgb,
        "7. LightGBM": prob_lgbm,
        "8. CatBoost": prob_catboost,
        "9. PySR (Symbolic Regression)": prob_pysr,
    }
    models = {name: predictions for name, predictions in models.items() if predictions is not None}

    results = [(name, binary_log_loss(y_test, predictions)) for name, predictions in models.items()]
    results.sort(key=lambda item: item[1])

    metadata = {
        "series": series_ticker,
        "sample_rows": int(len(actuals_eval)),
        "train_rows": int(splits["train_rows"]),
        "validation_rows": int(splits["validation_rows"]),
        "test_rows": int(splits["test_rows"]),
        "using_sample": using_sampled_tournament,
        "feature_names": feature_names_list,
    }
    write_tournament_outputs(artifacts_dir, results, metadata)

    print("\n" + "=" * 60)
    if using_sampled_tournament:
        title = f"MODEL COMPETITION RESULTS ({len(actuals_eval):,}-Row Sample)"
    else:
        title = "MODEL COMPETITION RESULTS (Overall 15m Horizon)"
    print(f"{title:^60}")
    print("=" * 60)
    print(f"{'Model Name':<40} | {'Log-Loss':<10}")
    print("-" * 60)

    for name, loss in results:
        print(f"{name:<40} | {loss:.5f}")

    print("\nCompact Summary")
    for name, loss in results:
        print(f"{MODEL_NAME_TO_LABEL[name]}: {loss:.5f}")

    best_name, best_loss = min(results, key=lambda item: item[1])
    print(f"\nBest predictor: {MODEL_NAME_TO_LABEL[best_name]} ({best_loss:.5f})")

    print("\n" + "=" * 60)
    print(f"{'STRIKE MAGNETISM (PINNING) TEST':^60}")
    print("=" * 60)

    danger_zone_mask = (tau < 3.0) & (market_probs >= 0.40) & (market_probs <= 0.60)
    dz_actuals = actuals[danger_zone_mask]
    dz_market_probs = market_probs[danger_zone_mask]

    if len(dz_actuals) == 0:
        print("Not enough data in the danger zone.")
        return

    dz_win_rate = np.mean(dz_actuals)
    dz_avg_prob = np.mean(dz_market_probs)
    print(f"Sample Size (Near Expiry, ATM): {len(dz_actuals):,}")
    print(f"Market Implied Probability : {dz_avg_prob:.4f}")
    print(f"Actual Realized Win Rate   : {dz_win_rate:.4f}")

    diff = dz_win_rate - dz_avg_prob
    if abs(diff) > 0.02:
        direction = (
            "Underpriced (momentum breaks through)"
            if diff > 0
            else "Overpriced (magnetism pins to NO)"
        )
        print(f"Conclusion: The market is wrong here by {diff * 100:+.2f}%. {direction}")
    else:
        print("Conclusion: Gaussian pricing holds up well near the strike.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model tournament on a Kalshi series backfill.")
    parser.add_argument("--series", default=DEFAULT_SERIES, help="Series ticker prefix, e.g. KXBTC15M")
    parser.add_argument("--markets-path", help="Path to the series markets parquet/csv file")
    parser.add_argument("--trades-path", help="Path to a trades directory or trades parquet file")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
        help="Stratified sample size for the tournament. Use 0 to use all rows.",
    )
    parser.add_argument("--artifacts-dir", help="Directory for saving tournament artifacts")
    parser.add_argument("--xgb-output-dir", help="Backward-compatible alias for --artifacts-dir")
    args = parser.parse_args()

    markets_path, trades_path = resolve_inputs(args.series, args.markets_path, args.trades_path)
    artifacts_dir = resolve_artifacts_dir(args.artifacts_dir, args.xgb_output_dir, args.series)
    run_tournament(
        series_ticker=args.series,
        markets_path=markets_path,
        trades_path=trades_path,
        sample_size=args.sample_size,
        artifacts_dir=artifacts_dir,
    )


if __name__ == "__main__":
    main()
