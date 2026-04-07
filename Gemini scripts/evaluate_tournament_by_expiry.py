from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import Booster
from pysr import PySRRegressor
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
SMARTERPRED_PATH = Path(__file__).with_name("smarterpred.py")
DEFAULT_SERIES = "KXBTC15M"
DEFAULT_ARTIFACTS_DIR = Path(__file__).with_name("model_tournament_output") / f"{DEFAULT_SERIES}_all_rows"
SCORE_COLUMNS = [
    "Standard Gaussian",
    "Student-T df=5",
    "Student-T df=3",
    "Momentum-Adjusted Gaussian",
    "Machine Learning (Logistic Regression)",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "PySR (Symbolic Regression)",
]


def load_smarterpred_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smarterpred_module", SMARTERPRED_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load smarterpred module from {SMARTERPRED_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_artifacts_dir(artifacts_dir: str | None, series_ticker: str) -> Path:
    if artifacts_dir:
        return Path(artifacts_dir)
    return Path(__file__).with_name("model_tournament_output") / f"{series_ticker}_all_rows"


def select_pysr_run_directory(artifacts_dir: Path) -> Path:
    metrics_path = artifacts_dir / "pysr" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best_params = metrics["best_params"]

    for candidate in metrics["candidates"]:
        if candidate["params"] == best_params:
            return artifacts_dir / "pysr_runs" / f"candidate_{candidate['candidate']:02d}"

    raise ValueError(f"Could not identify the winning PySR candidate from {metrics_path}")


def build_dataset(
    smarterpred: ModuleType,
    series_ticker: str,
    markets_path: Path,
    trades_path: Path,
    sample_size: int,
) -> dict[str, np.ndarray]:
    actuals, market_probs, tau, momentum = smarterpred.load_feature_frame(
        series_ticker=series_ticker,
        markets_path=markets_path,
        trades_path=trades_path,
    )

    eval_indices = smarterpred.stratified_sample_indices(actuals, sample_size, smarterpred.ML_RANDOM_SEED)
    actuals_eval = actuals[eval_indices]
    market_probs_eval = market_probs[eval_indices]
    tau_eval = tau[eval_indices]
    momentum_eval = momentum[eval_indices]

    feature_matrix = smarterpred.build_feature_matrix(market_probs_eval, tau_eval, momentum_eval)
    splits = smarterpred.split_dataset(feature_matrix, actuals_eval, market_probs_eval, tau_eval, momentum_eval)
    return splits


def load_saved_predictions(
    smarterpred: ModuleType,
    artifacts_dir: Path,
    splits: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    market_prob_test = splits["market_prob_test"]
    momentum_test = splits["momentum_test"]

    predictions: dict[str, np.ndarray] = {}

    z_implied_test = norm.ppf(market_prob_test)
    predictions["1. Standard Gaussian (The Market)"] = smarterpred.clean_probabilities(market_prob_test)
    predictions["2. Fat Tails (Student-T, df=5)"] = smarterpred.clean_probabilities(smarterpred.t.cdf(z_implied_test, df=5))
    predictions["3. Extreme Fat Tails (Student-T, df=3)"] = smarterpred.clean_probabilities(smarterpred.t.cdf(z_implied_test, df=3))
    predictions["4. Momentum-Adjusted Gaussian"] = smarterpred.clean_probabilities(
        predictions["1. Standard Gaussian (The Market)"] + (np.sign(momentum_test) * 0.02)
    )

    print("Training Logistic Regression for evaluation...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    predictions["5. Machine Learning (Logistic Regression)"] = smarterpred.clean_probabilities(
        lr_model.predict_proba(X_test)[:, 1]
    )

    print("Loading XGBoost artifact...")
    xgb_model = XGBClassifier()
    xgb_model.load_model(artifacts_dir / "xgboost" / "model.json")
    predictions["6. Gradient Boosted Trees (XGBoost)"] = smarterpred.clean_probabilities(
        xgb_model.predict_proba(X_test)[:, 1]
    )

    print("Loading LightGBM artifact...")
    lightgbm_model = Booster(model_file=str(artifacts_dir / "lightgbm" / "model.txt"))
    predictions["7. LightGBM"] = smarterpred.clean_probabilities(lightgbm_model.predict(X_test))

    print("Loading CatBoost artifact...")
    catboost_model = CatBoostClassifier()
    catboost_model.load_model(str(artifacts_dir / "catboost" / "model.cbm"))
    predictions["8. CatBoost"] = smarterpred.clean_probabilities(catboost_model.predict_proba(X_test)[:, 1])

    print("Loading PySR artifact...")
    pysr_model = PySRRegressor.from_file(run_directory=select_pysr_run_directory(artifacts_dir))
    predictions["9. PySR (Symbolic Regression)"] = smarterpred.clean_probabilities(pysr_model.predict(X_test))

    overall_rows = []
    for model_name, model_predictions in predictions.items():
        overall_rows.append(
            {
                "model": smarterpred.MODEL_NAME_TO_LABEL[model_name],
                "log_loss": smarterpred.binary_log_loss(y_test, model_predictions),
            }
        )
    overall_df = pd.DataFrame(overall_rows).sort_values("log_loss").reset_index(drop=True)
    print("\nOverall held-out log-loss (reconstructed from artifacts):")
    print(overall_df.to_string(index=False))

    return predictions


def evaluate_by_expiry_bucket(
    smarterpred: ModuleType,
    predictions: dict[str, np.ndarray],
    y_test: np.ndarray,
    tau_test: np.ndarray,
    minute_start: int,
    minute_end: int,
) -> pd.DataFrame:
    tte_bucket = np.rint(tau_test).astype(np.int32)
    rows: list[dict[str, object]] = []

    for minute in range(minute_start, minute_end - 1, -1):
        bucket_mask = tte_bucket == minute
        sample_size = int(bucket_mask.sum())
        if sample_size == 0:
            continue

        bucket_losses = {
            model_name: smarterpred.binary_log_loss(y_test[bucket_mask], model_predictions[bucket_mask])
            for model_name, model_predictions in predictions.items()
        }
        best_model_name, best_loss = min(bucket_losses.items(), key=lambda item: item[1])

        row: dict[str, object] = {
            "tte_bucket": minute,
            "sample_size": sample_size,
            "best_model": smarterpred.MODEL_NAME_TO_LABEL[best_model_name],
            "best_log_loss": float(best_loss),
        }
        for model_name, loss in bucket_losses.items():
            row[smarterpred.MODEL_NAME_TO_LABEL[model_name]] = float(loss)
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_by_probability_bucket(
    smarterpred: ModuleType,
    predictions: dict[str, np.ndarray],
    y_test: np.ndarray,
    market_prob_test: np.ndarray,
    bucket_size_cents: int,
) -> pd.DataFrame:
    market_prob_cents = market_prob_test * 100.0
    bucket_lower = (np.floor(market_prob_cents / bucket_size_cents) * bucket_size_cents).astype(np.int32)
    rows: list[dict[str, object]] = []

    for lower in range(0, 100, bucket_size_cents):
        upper = min(lower + bucket_size_cents, 100)
        if upper == 100:
            bucket_mask = (bucket_lower == lower) & (market_prob_cents <= 100.0)
        else:
            bucket_mask = bucket_lower == lower

        sample_size = int(bucket_mask.sum())
        if sample_size == 0:
            continue

        bucket_losses = {
            model_name: smarterpred.binary_log_loss(y_test[bucket_mask], model_predictions[bucket_mask])
            for model_name, model_predictions in predictions.items()
        }
        best_model_name, best_loss = min(bucket_losses.items(), key=lambda item: item[1])

        row: dict[str, object] = {
            "bucket_label": f"{lower}-{upper}c",
            "bucket_start_cents": lower,
            "bucket_end_cents": upper,
            "sample_size": sample_size,
            "best_model": smarterpred.MODEL_NAME_TO_LABEL[best_model_name],
            "best_log_loss": float(best_loss),
        }
        for model_name, loss in bucket_losses.items():
            row[smarterpred.MODEL_NAME_TO_LABEL[model_name]] = float(loss)
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_by_market_side(
    smarterpred: ModuleType,
    predictions: dict[str, np.ndarray],
    y_test: np.ndarray,
    market_prob_test: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    side_buckets = [
        ("NO-leaning", market_prob_test < 0.50),
        ("YES-leaning", market_prob_test >= 0.50),
    ]

    for side_label, bucket_mask in side_buckets:
        sample_size = int(bucket_mask.sum())
        if sample_size == 0:
            continue

        bucket_losses = {
            model_name: smarterpred.binary_log_loss(y_test[bucket_mask], model_predictions[bucket_mask])
            for model_name, model_predictions in predictions.items()
        }
        best_model_name, best_loss = min(bucket_losses.items(), key=lambda item: item[1])

        row: dict[str, object] = {
            "market_side": side_label,
            "sample_size": sample_size,
            "best_model": smarterpred.MODEL_NAME_TO_LABEL[best_model_name],
            "best_log_loss": float(best_loss),
        }
        for model_name, loss in bucket_losses.items():
            row[smarterpred.MODEL_NAME_TO_LABEL[model_name]] = float(loss)
        rows.append(row)

    return pd.DataFrame(rows)


def format_signed_momentum_bucket(lower_cents: int, upper_exclusive_cents: int) -> str:
    if upper_exclusive_cents - lower_cents == 1:
        return f"{lower_cents:+d}c"
    return f"{lower_cents:+d}c to {upper_exclusive_cents - 1:+d}c"


def evaluate_by_expiry_probability_bucket(
    smarterpred: ModuleType,
    predictions: dict[str, np.ndarray],
    y_test: np.ndarray,
    tau_test: np.ndarray,
    market_prob_test: np.ndarray,
    minute_start: int,
    minute_end: int,
    bucket_size_cents: int,
) -> pd.DataFrame:
    tte_bucket = np.rint(tau_test).astype(np.int32)
    market_prob_cents = market_prob_test * 100.0
    prob_bucket_start = (np.floor(market_prob_cents / bucket_size_cents) * bucket_size_cents).astype(np.int32)
    rows: list[dict[str, object]] = []

    for minute in range(minute_start, minute_end - 1, -1):
        minute_mask = tte_bucket == minute
        if not minute_mask.any():
            continue

        for lower in range(0, 100, bucket_size_cents):
            upper = min(lower + bucket_size_cents, 100)
            if upper == 100:
                bucket_mask = minute_mask & (prob_bucket_start == lower) & (market_prob_cents <= 100.0)
            else:
                bucket_mask = minute_mask & (prob_bucket_start == lower)

            sample_size = int(bucket_mask.sum())
            if sample_size == 0:
                continue

            bucket_losses = {
                model_name: smarterpred.binary_log_loss(y_test[bucket_mask], model_predictions[bucket_mask])
                for model_name, model_predictions in predictions.items()
            }
            best_model_name, best_loss = min(bucket_losses.items(), key=lambda item: item[1])

            row: dict[str, object] = {
                "tte_bucket": minute,
                "probability_bucket": f"{lower}-{upper}c",
                "probability_bucket_start_cents": lower,
                "probability_bucket_end_cents": upper,
                "sample_size": sample_size,
                "best_model": smarterpred.MODEL_NAME_TO_LABEL[best_model_name],
                "best_log_loss": float(best_loss),
            }
            for model_name, loss in bucket_losses.items():
                row[smarterpred.MODEL_NAME_TO_LABEL[model_name]] = float(loss)
            rows.append(row)

    return pd.DataFrame(rows)


def evaluate_by_probability_market_side_bucket(
    smarterpred: ModuleType,
    predictions: dict[str, np.ndarray],
    y_test: np.ndarray,
    market_prob_test: np.ndarray,
    bucket_size_cents: int,
) -> pd.DataFrame:
    market_prob_cents = market_prob_test * 100.0
    prob_bucket_start = (np.floor(market_prob_cents / bucket_size_cents) * bucket_size_cents).astype(np.int32)
    market_side = np.where(market_prob_test < 0.50, "NO-leaning", "YES-leaning")
    rows: list[dict[str, object]] = []

    for side_label in ["NO-leaning", "YES-leaning"]:
        side_mask = market_side == side_label
        if not side_mask.any():
            continue

        for lower in range(0, 100, bucket_size_cents):
            upper = min(lower + bucket_size_cents, 100)
            if upper == 100:
                bucket_mask = side_mask & (prob_bucket_start == lower) & (market_prob_cents <= 100.0)
            else:
                bucket_mask = side_mask & (prob_bucket_start == lower)

            sample_size = int(bucket_mask.sum())
            if sample_size == 0:
                continue

            bucket_losses = {
                model_name: smarterpred.binary_log_loss(y_test[bucket_mask], model_predictions[bucket_mask])
                for model_name, model_predictions in predictions.items()
            }
            best_model_name, best_loss = min(bucket_losses.items(), key=lambda item: item[1])

            row: dict[str, object] = {
                "market_side": side_label,
                "probability_bucket": f"{lower}-{upper}c",
                "probability_bucket_start_cents": lower,
                "probability_bucket_end_cents": upper,
                "sample_size": sample_size,
                "best_model": smarterpred.MODEL_NAME_TO_LABEL[best_model_name],
                "best_log_loss": float(best_loss),
            }
            for model_name, loss in bucket_losses.items():
                row[smarterpred.MODEL_NAME_TO_LABEL[model_name]] = float(loss)
            rows.append(row)

    return pd.DataFrame(rows)


def evaluate_by_expiry_momentum_bucket(
    smarterpred: ModuleType,
    predictions: dict[str, np.ndarray],
    y_test: np.ndarray,
    tau_test: np.ndarray,
    momentum_test: np.ndarray,
    minute_start: int,
    minute_end: int,
    bucket_size_cents: int,
) -> pd.DataFrame:
    tte_bucket = np.rint(tau_test).astype(np.int32)
    momentum_cents = np.rint(momentum_test * 100.0).astype(np.int32)
    bucket_start = np.floor_divide(momentum_cents, bucket_size_cents) * bucket_size_cents
    rows: list[dict[str, object]] = []

    min_bucket_start = int(bucket_start.min())
    max_bucket_start = int(bucket_start.max())

    for minute in range(minute_start, minute_end - 1, -1):
        minute_mask = tte_bucket == minute
        if not minute_mask.any():
            continue

        for lower in range(min_bucket_start, max_bucket_start + 1, bucket_size_cents):
            upper_exclusive = lower + bucket_size_cents
            bucket_mask = minute_mask & (bucket_start == lower)
            sample_size = int(bucket_mask.sum())
            if sample_size == 0:
                continue

            bucket_losses = {
                model_name: smarterpred.binary_log_loss(y_test[bucket_mask], model_predictions[bucket_mask])
                for model_name, model_predictions in predictions.items()
            }
            best_model_name, best_loss = min(bucket_losses.items(), key=lambda item: item[1])

            row: dict[str, object] = {
                "tte_bucket": minute,
                "momentum_bucket": format_signed_momentum_bucket(lower, upper_exclusive),
                "momentum_bucket_start_cents": lower,
                "momentum_bucket_end_cents_exclusive": upper_exclusive,
                "sample_size": sample_size,
                "best_model": smarterpred.MODEL_NAME_TO_LABEL[best_model_name],
                "best_log_loss": float(best_loss),
            }
            for model_name, loss in bucket_losses.items():
                row[smarterpred.MODEL_NAME_TO_LABEL[model_name]] = float(loss)
            rows.append(row)

    return pd.DataFrame(rows)


def evaluate_by_expiry_distance_bucket(
    smarterpred: ModuleType,
    predictions: dict[str, np.ndarray],
    y_test: np.ndarray,
    tau_test: np.ndarray,
    market_prob_test: np.ndarray,
    minute_start: int,
    minute_end: int,
    bucket_size_cents: int,
) -> pd.DataFrame:
    tte_bucket = np.rint(tau_test).astype(np.int32)
    distance_from_mid_cents = np.abs((market_prob_test * 100.0) - 50.0)
    distance_bucket_start = (np.floor(distance_from_mid_cents / bucket_size_cents) * bucket_size_cents).astype(np.int32)
    rows: list[dict[str, object]] = []

    max_bucket_start = int(np.floor(50 / bucket_size_cents) * bucket_size_cents)
    for minute in range(minute_start, minute_end - 1, -1):
        minute_mask = tte_bucket == minute
        if not minute_mask.any():
            continue

        for lower in range(0, max_bucket_start + 1, bucket_size_cents):
            upper = min(lower + bucket_size_cents, 50)
            if upper == 50:
                bucket_mask = minute_mask & (distance_bucket_start == lower) & (distance_from_mid_cents <= 50.0)
            else:
                bucket_mask = minute_mask & (distance_bucket_start == lower)

            sample_size = int(bucket_mask.sum())
            if sample_size == 0:
                continue

            bucket_losses = {
                model_name: smarterpred.binary_log_loss(y_test[bucket_mask], model_predictions[bucket_mask])
                for model_name, model_predictions in predictions.items()
            }
            best_model_name, best_loss = min(bucket_losses.items(), key=lambda item: item[1])

            row: dict[str, object] = {
                "tte_bucket": minute,
                "distance_bucket": f"{lower}-{upper}c",
                "distance_bucket_start_cents": lower,
                "distance_bucket_end_cents": upper,
                "sample_size": sample_size,
                "best_model": smarterpred.MODEL_NAME_TO_LABEL[best_model_name],
                "best_log_loss": float(best_loss),
            }
            for model_name, loss in bucket_losses.items():
                row[smarterpred.MODEL_NAME_TO_LABEL[model_name]] = float(loss)
            rows.append(row)

    return pd.DataFrame(rows)


def build_expiry_markdown_report(
    df: pd.DataFrame,
    series_ticker: str,
    minute_start: int,
    minute_end: int,
) -> str:
    lines = [
        f"# {series_ticker} Tournament By Expiry Minute",
        "",
        "- Metric: `log_loss` on the held-out test split",
        f"- TTE buckets: rounded `tau_minutes`, from `{minute_start}` down to `{minute_end}`",
        "- Models: same tournament lineup as `smarterpred.py`",
        "",
        "## Winners By Minute",
        "",
        "| TTE (min) | Test Rows | Best Model | Best Log-Loss |",
        "| --- | ---: | --- | ---: |",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"| {int(row['tte_bucket'])} | {int(row['sample_size']):,} | "
            f"{row['best_model']} | {float(row['best_log_loss']):.5f} |"
        )

    lines.extend(
        [
            "",
            "## Full Score Matrix",
            "",
            "| TTE (min) | Test Rows | Standard Gaussian | Student-T df=5 | Student-T df=3 | Momentum-Adjusted Gaussian | Machine Learning (Logistic Regression) | XGBoost | LightGBM | CatBoost | PySR (Symbolic Regression) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, row in df.iterrows():
        values = " | ".join(f"{float(row[column]):.5f}" for column in SCORE_COLUMNS)
        lines.append(f"| {int(row['tte_bucket'])} | {int(row['sample_size']):,} | {values} |")

    lines.append("")
    return "\n".join(lines)


def build_probability_markdown_report(
    df: pd.DataFrame,
    series_ticker: str,
    bucket_size_cents: int,
) -> str:
    lines = [
        f"# {series_ticker} Tournament By Market Probability Bucket",
        "",
        "- Metric: `log_loss` on the held-out test split",
        f"- Probability buckets: implied market probability in `{bucket_size_cents}`-cent bins",
        "- Models: same tournament lineup as `smarterpred.py`",
        "",
        "## Winners By Bucket",
        "",
        "| Probability Bucket | Test Rows | Best Model | Best Log-Loss |",
        "| --- | ---: | --- | ---: |",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"| {row['bucket_label']} | {int(row['sample_size']):,} | "
            f"{row['best_model']} | {float(row['best_log_loss']):.5f} |"
        )

    lines.extend(
        [
            "",
            "## Full Score Matrix",
            "",
            "| Probability Bucket | Test Rows | Standard Gaussian | Student-T df=5 | Student-T df=3 | Momentum-Adjusted Gaussian | Machine Learning (Logistic Regression) | XGBoost | LightGBM | CatBoost | PySR (Symbolic Regression) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, row in df.iterrows():
        values = " | ".join(f"{float(row[column]):.5f}" for column in SCORE_COLUMNS)
        lines.append(f"| {row['bucket_label']} | {int(row['sample_size']):,} | {values} |")

    lines.append("")
    return "\n".join(lines)


def build_market_side_markdown_report(
    df: pd.DataFrame,
    series_ticker: str,
) -> str:
    lines = [
        f"# {series_ticker} Tournament By Market Outcome Side",
        "",
        "- Metric: `log_loss` on the held-out test split",
        "- Market side buckets: `NO-leaning` for implied probability `< 50c`, `YES-leaning` for `>= 50c`",
        "- Models: same tournament lineup as `smarterpred.py`",
        "",
        "## Winners By Side",
        "",
        "| Market Side | Test Rows | Best Model | Best Log-Loss |",
        "| --- | ---: | --- | ---: |",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"| {row['market_side']} | {int(row['sample_size']):,} | "
            f"{row['best_model']} | {float(row['best_log_loss']):.5f} |"
        )

    lines.extend(
        [
            "",
            "## Full Score Matrix",
            "",
            "| Market Side | Test Rows | Standard Gaussian | Student-T df=5 | Student-T df=3 | Momentum-Adjusted Gaussian | Machine Learning (Logistic Regression) | XGBoost | LightGBM | CatBoost | PySR (Symbolic Regression) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, row in df.iterrows():
        values = " | ".join(f"{float(row[column]):.5f}" for column in SCORE_COLUMNS)
        lines.append(f"| {row['market_side']} | {int(row['sample_size']):,} | {values} |")

    lines.append("")
    return "\n".join(lines)


def build_expiry_momentum_markdown_report(
    df: pd.DataFrame,
    series_ticker: str,
    minute_start: int,
    minute_end: int,
    bucket_size_cents: int,
) -> str:
    lines = [
        f"# {series_ticker} Tournament By Expiry Minute And Price Momentum",
        "",
        "- Metric: `log_loss` on the held-out test split",
        f"- TTE buckets: rounded `tau_minutes`, from `{minute_start}` down to `{minute_end}`",
        f"- Momentum buckets: signed `price_momentum` in `{bucket_size_cents}`-cent bins",
        "- Models: same tournament lineup as `smarterpred.py`",
        "",
        "## Winners By Joint Bucket",
        "",
        "| TTE (min) | Price Momentum | Test Rows | Best Model | Best Log-Loss |",
        "| --- | --- | ---: | --- | ---: |",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"| {int(row['tte_bucket'])} | {row['momentum_bucket']} | {int(row['sample_size']):,} | "
            f"{row['best_model']} | {float(row['best_log_loss']):.5f} |"
        )

    lines.extend(
        [
            "",
            "## Full Score Matrix",
            "",
            "| TTE (min) | Price Momentum | Test Rows | Standard Gaussian | Student-T df=5 | Student-T df=3 | Momentum-Adjusted Gaussian | Machine Learning (Logistic Regression) | XGBoost | LightGBM | CatBoost | PySR (Symbolic Regression) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, row in df.iterrows():
        values = " | ".join(f"{float(row[column]):.5f}" for column in SCORE_COLUMNS)
        lines.append(
            f"| {int(row['tte_bucket'])} | {row['momentum_bucket']} | {int(row['sample_size']):,} | {values} |"
        )

    lines.append("")
    return "\n".join(lines)


def build_probability_market_side_markdown_report(
    df: pd.DataFrame,
    series_ticker: str,
    bucket_size_cents: int,
) -> str:
    lines = [
        f"# {series_ticker} Tournament By Market Probability And Market Outcome Side",
        "",
        "- Metric: `log_loss` on the held-out test split",
        f"- Probability buckets: implied market probability in `{bucket_size_cents}`-cent bins",
        "- Market side buckets: `NO-leaning` for implied probability `< 50c`, `YES-leaning` for `>= 50c`",
        "- Models: same tournament lineup as `smarterpred.py`",
        "",
        "## Winners By Joint Bucket",
        "",
        "| Market Side | Probability Bucket | Test Rows | Best Model | Best Log-Loss |",
        "| --- | --- | ---: | --- | ---: |",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"| {row['market_side']} | {row['probability_bucket']} | {int(row['sample_size']):,} | "
            f"{row['best_model']} | {float(row['best_log_loss']):.5f} |"
        )

    lines.extend(
        [
            "",
            "## Full Score Matrix",
            "",
            "| Market Side | Probability Bucket | Test Rows | Standard Gaussian | Student-T df=5 | Student-T df=3 | Momentum-Adjusted Gaussian | Machine Learning (Logistic Regression) | XGBoost | LightGBM | CatBoost | PySR (Symbolic Regression) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, row in df.iterrows():
        values = " | ".join(f"{float(row[column]):.5f}" for column in SCORE_COLUMNS)
        lines.append(
            f"| {row['market_side']} | {row['probability_bucket']} | {int(row['sample_size']):,} | {values} |"
        )

    lines.append("")
    return "\n".join(lines)


def build_expiry_probability_markdown_report(
    df: pd.DataFrame,
    series_ticker: str,
    minute_start: int,
    minute_end: int,
    bucket_size_cents: int,
) -> str:
    lines = [
        f"# {series_ticker} Tournament By Expiry Minute And Market Probability",
        "",
        "- Metric: `log_loss` on the held-out test split",
        f"- TTE buckets: rounded `tau_minutes`, from `{minute_start}` down to `{minute_end}`",
        f"- Probability buckets: implied market probability in `{bucket_size_cents}`-cent bins",
        "- Models: same tournament lineup as `smarterpred.py`",
        "",
        "## Winners By Joint Bucket",
        "",
        "| TTE (min) | Probability Bucket | Test Rows | Best Model | Best Log-Loss |",
        "| --- | --- | ---: | --- | ---: |",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"| {int(row['tte_bucket'])} | {row['probability_bucket']} | {int(row['sample_size']):,} | "
            f"{row['best_model']} | {float(row['best_log_loss']):.5f} |"
        )

    lines.extend(
        [
            "",
            "## Full Score Matrix",
            "",
            "| TTE (min) | Probability Bucket | Test Rows | Standard Gaussian | Student-T df=5 | Student-T df=3 | Momentum-Adjusted Gaussian | Machine Learning (Logistic Regression) | XGBoost | LightGBM | CatBoost | PySR (Symbolic Regression) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, row in df.iterrows():
        values = " | ".join(f"{float(row[column]):.5f}" for column in SCORE_COLUMNS)
        lines.append(
            f"| {int(row['tte_bucket'])} | {row['probability_bucket']} | {int(row['sample_size']):,} | {values} |"
        )

    lines.append("")
    return "\n".join(lines)


def build_expiry_distance_markdown_report(
    df: pd.DataFrame,
    series_ticker: str,
    minute_start: int,
    minute_end: int,
    bucket_size_cents: int,
) -> str:
    lines = [
        f"# {series_ticker} Tournament By Expiry Minute And Distance From Mid",
        "",
        "- Metric: `log_loss` on the held-out test split",
        f"- TTE buckets: rounded `tau_minutes`, from `{minute_start}` down to `{minute_end}`",
        f"- Distance buckets: `abs(market_price_cents - 50)` in `{bucket_size_cents}`-cent bins",
        "- Models: same tournament lineup as `smarterpred.py`",
        "",
        "## Winners By Joint Bucket",
        "",
        "| TTE (min) | Distance From Mid | Test Rows | Best Model | Best Log-Loss |",
        "| --- | --- | ---: | --- | ---: |",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"| {int(row['tte_bucket'])} | {row['distance_bucket']} | {int(row['sample_size']):,} | "
            f"{row['best_model']} | {float(row['best_log_loss']):.5f} |"
        )

    lines.extend(
        [
            "",
            "## Full Score Matrix",
            "",
            "| TTE (min) | Distance From Mid | Test Rows | Standard Gaussian | Student-T df=5 | Student-T df=3 | Momentum-Adjusted Gaussian | Machine Learning (Logistic Regression) | XGBoost | LightGBM | CatBoost | PySR (Symbolic Regression) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, row in df.iterrows():
        values = " | ".join(f"{float(row[column]):.5f}" for column in SCORE_COLUMNS)
        lines.append(
            f"| {int(row['tte_bucket'])} | {row['distance_bucket']} | {int(row['sample_size']):,} | {values} |"
        )

    lines.append("")
    return "\n".join(lines)


def run_evaluation(
    series_ticker: str,
    markets_path: Path,
    trades_path: Path,
    artifacts_dir: Path,
    sample_size: int,
    bucket_mode: str,
    minute_start: int,
    minute_end: int,
    probability_bucket_size_cents: int,
    momentum_bucket_size_cents: int,
) -> None:
    smarterpred = load_smarterpred_module()

    print(f"Series: {series_ticker}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Markets: {markets_path}")
    print(f"Trades: {trades_path}")

    splits = build_dataset(
        smarterpred=smarterpred,
        series_ticker=series_ticker,
        markets_path=markets_path,
        trades_path=trades_path,
        sample_size=sample_size,
    )
    predictions = load_saved_predictions(smarterpred=smarterpred, artifacts_dir=artifacts_dir, splits=splits)

    if bucket_mode == "expiry":
        result_df = evaluate_by_expiry_bucket(
            smarterpred=smarterpred,
            predictions=predictions,
            y_test=splits["y_test"],
            tau_test=splits["tau_test"],
            minute_start=minute_start,
            minute_end=minute_end,
        )
        output_csv = artifacts_dir / "per_expiry_log_loss.csv"
        output_json = artifacts_dir / "per_expiry_log_loss.json"
        output_md = artifacts_dir / "per_expiry_summary.md"
        output_md.write_text(
            build_expiry_markdown_report(
                df=result_df,
                series_ticker=series_ticker,
                minute_start=minute_start,
                minute_end=minute_end,
            ),
            encoding="utf-8",
        )
        print("\nBest model by expiry bucket:")
        print(result_df[["tte_bucket", "sample_size", "best_model", "best_log_loss"]].to_string(index=False))
    elif bucket_mode == "probability":
        result_df = evaluate_by_probability_bucket(
            smarterpred=smarterpred,
            predictions=predictions,
            y_test=splits["y_test"],
            market_prob_test=splits["market_prob_test"],
            bucket_size_cents=probability_bucket_size_cents,
        )
        output_csv = artifacts_dir / f"per_probability_{probability_bucket_size_cents}c_log_loss.csv"
        output_json = artifacts_dir / f"per_probability_{probability_bucket_size_cents}c_log_loss.json"
        output_md = artifacts_dir / f"per_probability_{probability_bucket_size_cents}c_summary.md"
        output_md.write_text(
            build_probability_markdown_report(
                df=result_df,
                series_ticker=series_ticker,
                bucket_size_cents=probability_bucket_size_cents,
            ),
            encoding="utf-8",
        )
        print("\nBest model by probability bucket:")
        print(result_df[["bucket_label", "sample_size", "best_model", "best_log_loss"]].to_string(index=False))
    elif bucket_mode == "market_side":
        result_df = evaluate_by_market_side(
            smarterpred=smarterpred,
            predictions=predictions,
            y_test=splits["y_test"],
            market_prob_test=splits["market_prob_test"],
        )
        output_csv = artifacts_dir / "per_market_side_log_loss.csv"
        output_json = artifacts_dir / "per_market_side_log_loss.json"
        output_md = artifacts_dir / "per_market_side_summary.md"
        output_md.write_text(
            build_market_side_markdown_report(
                df=result_df,
                series_ticker=series_ticker,
            ),
            encoding="utf-8",
        )
        print("\nBest model by market side:")
        print(result_df[["market_side", "sample_size", "best_model", "best_log_loss"]].to_string(index=False))
    elif bucket_mode == "probability_market_side":
        result_df = evaluate_by_probability_market_side_bucket(
            smarterpred=smarterpred,
            predictions=predictions,
            y_test=splits["y_test"],
            market_prob_test=splits["market_prob_test"],
            bucket_size_cents=probability_bucket_size_cents,
        )
        output_csv = artifacts_dir / f"per_probability_market_side_{probability_bucket_size_cents}c_log_loss.csv"
        output_json = artifacts_dir / f"per_probability_market_side_{probability_bucket_size_cents}c_log_loss.json"
        output_md = artifacts_dir / f"per_probability_market_side_{probability_bucket_size_cents}c_summary.md"
        output_md.write_text(
            build_probability_market_side_markdown_report(
                df=result_df,
                series_ticker=series_ticker,
                bucket_size_cents=probability_bucket_size_cents,
            ),
            encoding="utf-8",
        )
        print("\nBest model by probability x market side bucket:")
        print(
            result_df[
                ["market_side", "probability_bucket", "sample_size", "best_model", "best_log_loss"]
            ].to_string(index=False)
        )
    elif bucket_mode == "expiry_probability":
        result_df = evaluate_by_expiry_probability_bucket(
            smarterpred=smarterpred,
            predictions=predictions,
            y_test=splits["y_test"],
            tau_test=splits["tau_test"],
            market_prob_test=splits["market_prob_test"],
            minute_start=minute_start,
            minute_end=minute_end,
            bucket_size_cents=probability_bucket_size_cents,
        )
        output_csv = artifacts_dir / f"per_expiry_probability_{probability_bucket_size_cents}c_log_loss.csv"
        output_json = artifacts_dir / f"per_expiry_probability_{probability_bucket_size_cents}c_log_loss.json"
        output_md = artifacts_dir / f"per_expiry_probability_{probability_bucket_size_cents}c_summary.md"
        output_md.write_text(
            build_expiry_probability_markdown_report(
                df=result_df,
                series_ticker=series_ticker,
                minute_start=minute_start,
                minute_end=minute_end,
                bucket_size_cents=probability_bucket_size_cents,
            ),
            encoding="utf-8",
        )
        print("\nBest model by expiry x probability bucket:")
        print(
            result_df[
                ["tte_bucket", "probability_bucket", "sample_size", "best_model", "best_log_loss"]
            ].to_string(index=False)
        )
    elif bucket_mode == "expiry_momentum":
        result_df = evaluate_by_expiry_momentum_bucket(
            smarterpred=smarterpred,
            predictions=predictions,
            y_test=splits["y_test"],
            tau_test=splits["tau_test"],
            momentum_test=splits["momentum_test"],
            minute_start=minute_start,
            minute_end=minute_end,
            bucket_size_cents=momentum_bucket_size_cents,
        )
        output_csv = artifacts_dir / f"per_expiry_momentum_{momentum_bucket_size_cents}c_log_loss.csv"
        output_json = artifacts_dir / f"per_expiry_momentum_{momentum_bucket_size_cents}c_log_loss.json"
        output_md = artifacts_dir / f"per_expiry_momentum_{momentum_bucket_size_cents}c_summary.md"
        output_md.write_text(
            build_expiry_momentum_markdown_report(
                df=result_df,
                series_ticker=series_ticker,
                minute_start=minute_start,
                minute_end=minute_end,
                bucket_size_cents=momentum_bucket_size_cents,
            ),
            encoding="utf-8",
        )
        print("\nBest model by expiry x momentum bucket:")
        print(
            result_df[
                ["tte_bucket", "momentum_bucket", "sample_size", "best_model", "best_log_loss"]
            ].to_string(index=False)
        )
    elif bucket_mode == "expiry_distance":
        result_df = evaluate_by_expiry_distance_bucket(
            smarterpred=smarterpred,
            predictions=predictions,
            y_test=splits["y_test"],
            tau_test=splits["tau_test"],
            market_prob_test=splits["market_prob_test"],
            minute_start=minute_start,
            minute_end=minute_end,
            bucket_size_cents=probability_bucket_size_cents,
        )
        output_csv = artifacts_dir / f"per_expiry_distance_{probability_bucket_size_cents}c_log_loss.csv"
        output_json = artifacts_dir / f"per_expiry_distance_{probability_bucket_size_cents}c_log_loss.json"
        output_md = artifacts_dir / f"per_expiry_distance_{probability_bucket_size_cents}c_summary.md"
        output_md.write_text(
            build_expiry_distance_markdown_report(
                df=result_df,
                series_ticker=series_ticker,
                minute_start=minute_start,
                minute_end=minute_end,
                bucket_size_cents=probability_bucket_size_cents,
            ),
            encoding="utf-8",
        )
        print("\nBest model by expiry x distance bucket:")
        print(
            result_df[
                ["tte_bucket", "distance_bucket", "sample_size", "best_model", "best_log_loss"]
            ].to_string(index=False)
        )
    else:
        raise ValueError(f"Unsupported bucket mode: {bucket_mode}")

    result_df.to_csv(output_csv, index=False)
    output_json.write_text(result_df.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"\nSaved CSV to {output_csv}")
    print(f"Saved JSON to {output_json}")
    print(f"Saved Markdown to {output_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tournament models by bucket on the held-out test split.")
    parser.add_argument("--series", default=DEFAULT_SERIES, help="Series ticker prefix, e.g. KXBTC15M")
    parser.add_argument("--markets-path", help="Path to the series markets parquet/csv file")
    parser.add_argument("--trades-path", help="Path to a trades directory or trades parquet file")
    parser.add_argument("--artifacts-dir", help="Directory containing saved tournament artifacts")
    parser.add_argument(
        "--bucket-mode",
        choices=[
            "expiry",
            "probability",
            "market_side",
            "probability_market_side",
            "expiry_probability",
            "expiry_momentum",
            "expiry_distance",
        ],
        default="expiry",
        help="Bucket dimension to evaluate on",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional evaluation sample size. Use 0 to evaluate all available rows.",
    )
    parser.add_argument(
        "--minute-start",
        type=int,
        default=14,
        help="Highest rounded expiry minute bucket to include",
    )
    parser.add_argument(
        "--minute-end",
        type=int,
        default=2,
        help="Lowest rounded expiry minute bucket to include",
    )
    parser.add_argument(
        "--probability-bucket-size-cents",
        type=int,
        default=5,
        help="Bucket width for market probability, in cents",
    )
    parser.add_argument(
        "--momentum-bucket-size-cents",
        type=int,
        default=1,
        help="Bucket width for signed price momentum, in cents",
    )
    args = parser.parse_args()

    smarterpred = load_smarterpred_module()
    markets_path, trades_path = smarterpred.resolve_inputs(args.series, args.markets_path, args.trades_path)
    artifacts_dir = resolve_artifacts_dir(args.artifacts_dir, args.series)

    if args.bucket_mode in {"expiry", "expiry_probability", "expiry_momentum", "expiry_distance"} and args.minute_start < args.minute_end:
        raise ValueError("--minute-start must be greater than or equal to --minute-end")
    if args.bucket_mode in {"probability", "probability_market_side", "expiry_probability", "expiry_distance"} and args.probability_bucket_size_cents <= 0:
        raise ValueError("--probability-bucket-size-cents must be positive")
    if args.bucket_mode == "expiry_momentum" and args.momentum_bucket_size_cents <= 0:
        raise ValueError("--momentum-bucket-size-cents must be positive")

    run_evaluation(
        series_ticker=args.series,
        markets_path=markets_path,
        trades_path=trades_path,
        artifacts_dir=artifacts_dir,
        sample_size=args.sample_size,
        bucket_mode=args.bucket_mode,
        minute_start=args.minute_start,
        minute_end=args.minute_end,
        probability_bucket_size_cents=args.probability_bucket_size_cents,
        momentum_bucket_size_cents=args.momentum_bucket_size_cents,
    )


if __name__ == "__main__":
    main()
