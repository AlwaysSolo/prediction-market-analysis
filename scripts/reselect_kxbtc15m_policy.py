from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from lightgbm import Booster
from sklearn.metrics import log_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.kalshi.offline_training import (  # noqa: E402
    MINIMUM_POLICY_TRADES_PER_DAY,
    PolicyConfig,
    PolicyResult,
    apply_calibration_to_predictions,
    bagged_lasso_raw_predictions,
    build_policy_diagnostics,
    build_prediction_export_frame,
    booster_raw_predictions,
    elastic_net_raw_predictions,
    lasso_raw_predictions,
    linear_svm_raw_predictions,
    load_bagged_lasso_artifact,
    load_elastic_net_artifact,
    load_feature_dataset,
    load_lasso_artifact,
    load_linear_svm_artifact,
    minimum_policy_trades_for_frame,
    prediction_export_columns,
    publish_latest_artifacts,
    select_policy_candidate,
    serialize_policy_result,
    write_json,
)

load_dotenv()


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_list(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list at {path}")
    return [item for item in payload if isinstance(item, dict)]


def _manifest_tickers(manifest: dict[str, object], key: str) -> tuple[str, ...]:
    values = manifest.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Split manifest is missing a valid {key} list.")
    return tuple(str(value) for value in values)


def _policy_result_from_payload(payload: dict[str, object]) -> PolicyResult:
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Policy search entry is missing a valid config object.")
    return PolicyResult(
        config=PolicyConfig(**config_payload),
        objective=float(payload["objective"]),
        trades=int(payload["trades"]),
        net_pnl_dollars=float(payload["net_pnl_dollars"]),
        max_drawdown_dollars=float(payload["max_drawdown_dollars"]),
        max_drawdown_pct=float(payload["max_drawdown_pct"]),
        return_pct=float(payload["return_pct"]),
        log_loss=float(payload["log_loss"]),
        skipped_due_open_ticker=int(payload["skipped_due_open_ticker"]),
        skipped_due_price_band=int(payload["skipped_due_price_band"]),
        skipped_due_post_cost_edge=int(payload["skipped_due_post_cost_edge"]),
    )


def _prediction_base_columns() -> list[str]:
    return ["ticker", "trade_id", "created_time", "close_time", "market_prob", "tau_minutes", "actual_outcome"]


def _predictions_have_required_columns(df: pd.DataFrame) -> bool:
    return set(prediction_export_columns()).issubset(df.columns)


def _model_family(summary: dict[str, object]) -> str:
    return str(summary.get("model_family", "lightgbm"))


def _lightgbm_best_iteration(run_dir: Path, summary: dict[str, object]) -> int | None:
    candidate = summary.get("best_iteration")
    if candidate is not None:
        return int(candidate)
    metrics_path = run_dir / "lightgbm" / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = _load_json(metrics_path)
    metric_iteration = metrics.get("best_iteration")
    return int(metric_iteration) if metric_iteration is not None else None


def _score_split_from_saved_model(
    run_dir: Path,
    split_df: pd.DataFrame,
    *,
    summary: dict[str, object],
) -> pd.DataFrame:
    model_family = _model_family(summary)
    if model_family == "lightgbm":
        model_path = run_dir / "lightgbm" / "model.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"Required LightGBM artifact not found: {model_path}")
        booster = Booster(model_file=str(model_path))
        raw = booster_raw_predictions(booster, split_df, best_iteration=_lightgbm_best_iteration(run_dir, summary))
        calibrated = apply_calibration_to_predictions(run_dir, raw, model_family="lightgbm")
    elif model_family == "lasso":
        model_path = run_dir / "lasso" / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Required LASSO artifact not found: {model_path}")
        artifact = load_lasso_artifact(model_path)
        raw = lasso_raw_predictions(artifact, split_df)
        calibrated = apply_calibration_to_predictions(run_dir, raw, model_family="lasso")
    elif model_family == "elastic_net":
        model_path = run_dir / "elastic_net" / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Required Elastic Net artifact not found: {model_path}")
        artifact = load_elastic_net_artifact(model_path)
        raw = elastic_net_raw_predictions(artifact, split_df)
        calibrated = apply_calibration_to_predictions(run_dir, raw, model_family="elastic_net")
    elif model_family == "linear_svm":
        model_path = run_dir / "linear_svm" / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Required Linear SVM artifact not found: {model_path}")
        artifact = load_linear_svm_artifact(model_path)
        raw = linear_svm_raw_predictions(artifact, split_df)
        calibrated = apply_calibration_to_predictions(run_dir, raw, model_family="linear_svm")
    elif model_family == "bagged_lasso":
        model_path = run_dir / "bagged_lasso" / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Required Bagged LASSO artifact not found: {model_path}")
        artifact = load_bagged_lasso_artifact(model_path)
        raw = bagged_lasso_raw_predictions(artifact, split_df)
        calibrated = apply_calibration_to_predictions(run_dir, raw, model_family="bagged_lasso")
    else:
        raise ValueError(f"Unsupported model family for fast reselection: {model_family}")
    return build_prediction_export_frame(split_df, raw, calibrated)


def _load_cached_split_frame(
    dataset_cache_dir: Path,
    tickers: tuple[str, ...],
    *,
    progress_desc: str | None,
) -> pd.DataFrame:
    return load_feature_dataset(dataset_cache_dir, tickers, progress_desc=progress_desc)


def _load_split_frame(
    run_dir: Path,
    split_name: str,
    *,
    summary: dict[str, object],
    dataset_cache_dir: Path | None,
    tickers: tuple[str, ...],
    progress_desc: str | None,
) -> pd.DataFrame:
    predictions_path = run_dir / f"{split_name}_predictions.parquet"
    if not predictions_path.exists():
        if dataset_cache_dir is None:
            raise FileNotFoundError(
                f"Required artifact not found: {predictions_path}. "
                "Pass --dataset-cache-dir once to regenerate missing prediction artifacts from the saved model."
            )
        base_df = _load_cached_split_frame(dataset_cache_dir, tickers, progress_desc=progress_desc)
        frame = _score_split_from_saved_model(run_dir, base_df, summary=summary)
        frame.to_parquet(predictions_path, index=False)
        return frame

    prediction_df = pd.read_parquet(predictions_path)
    if _predictions_have_required_columns(prediction_df):
        frame = prediction_df.loc[:, prediction_export_columns()].copy()
        frame["created_time"] = pd.to_datetime(frame["created_time"], utc=True)
        frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
        return frame.sort_values(["created_time", "ticker", "trade_id"]).reset_index(drop=True)

    if dataset_cache_dir is None:
        raise FileNotFoundError(
            f"{predictions_path} does not contain close_time. "
            "Pass --dataset-cache-dir once to enrich the saved prediction artifacts for future fast reselection."
        )

    base_df = _load_cached_split_frame(dataset_cache_dir, tickers, progress_desc=progress_desc)
    required_prediction_columns = {"ticker", "trade_id", "raw_probability", "calibrated_probability"}
    if not required_prediction_columns.issubset(prediction_df.columns):
        raise ValueError(f"{predictions_path} is missing required probability columns.")
    enriched = base_df.merge(
        prediction_df.loc[:, ["ticker", "trade_id", "raw_probability", "calibrated_probability"]],
        on=["ticker", "trade_id"],
        how="inner",
        validate="one_to_one",
    )
    frame = build_prediction_export_frame(
        enriched,
        enriched["raw_probability"].to_numpy(),
        enriched["calibrated_probability"].to_numpy(),
    )
    frame.to_parquet(predictions_path, index=False)
    return frame


def _refresh_policy_outputs(
    run_dir: Path,
    *,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    selected_policy: PolicyResult,
    policy_selection_mode: str,
    met_validation_requirements: bool,
    minimum_validation_trades: int,
) -> dict[str, object]:
    validation_raw = validation_frame["raw_probability"].to_numpy()
    validation_calibrated = validation_frame["calibrated_probability"].to_numpy()
    validation_log_loss = float(log_loss(validation_frame["actual_outcome"], validation_calibrated, labels=[0, 1]))
    validation_diagnostics, validation_trade_records = build_policy_diagnostics(
        validation_frame.loc[:, _prediction_base_columns()],
        validation_raw,
        validation_calibrated,
        selected_policy.config,
        overall_log_loss=validation_log_loss,
    )
    validation_metrics = dict(validation_diagnostics["policy_metrics"])
    validation_metrics["selection_mode"] = policy_selection_mode
    validation_metrics["met_validation_requirements"] = met_validation_requirements
    validation_metrics["minimum_validation_trades_required"] = minimum_validation_trades

    test_raw = test_frame["raw_probability"].to_numpy()
    test_calibrated = test_frame["calibrated_probability"].to_numpy()
    test_log_loss = float(log_loss(test_frame["actual_outcome"], test_calibrated, labels=[0, 1]))
    test_diagnostics, test_trade_records = build_policy_diagnostics(
        test_frame.loc[:, _prediction_base_columns()],
        test_raw,
        test_calibrated,
        selected_policy.config,
        overall_log_loss=test_log_loss,
    )
    test_metrics = dict(test_diagnostics["policy_metrics"])
    test_metrics["selection_mode"] = policy_selection_mode
    test_metrics["met_validation_requirements"] = met_validation_requirements
    test_metrics["minimum_validation_trades_required"] = minimum_validation_trades

    write_json(
        run_dir / "policy.json",
        serialize_policy_result(
            selected_policy,
            selection_mode=policy_selection_mode,
            met_validation_requirements=met_validation_requirements,
            minimum_trades_required=minimum_validation_trades,
        ),
    )
    write_json(run_dir / "validation_metrics.json", validation_metrics)
    write_json(run_dir / "validation_diagnostics.json", validation_diagnostics)
    validation_trade_records.to_parquet(run_dir / "validation_trade_records.parquet", index=False)
    validation_frame.to_parquet(run_dir / "validation_predictions.parquet", index=False)

    write_json(run_dir / "test_metrics.json", test_metrics)
    write_json(run_dir / "test_diagnostics.json", test_diagnostics)
    test_trade_records.to_parquet(run_dir / "test_trade_records.parquet", index=False)
    test_frame.to_parquet(run_dir / "test_predictions.parquet", index=False)

    return {
        "validation_log_loss": validation_log_loss,
        "test_log_loss": test_log_loss,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast policy-only reselection for an existing KXBTC15M run using saved policy_search.json."
    )
    parser.add_argument("--run-dir", required=True, help="Existing run directory under artifacts/kalshi/")
    parser.add_argument(
        "--dataset-cache-dir",
        help="Optional cache directory only needed for older runs whose prediction parquet files do not include close_time.",
    )
    parser.add_argument(
        "--strict-policy-selection",
        action="store_true",
        help="Require a validation policy with positive PnL and the minimum trade count; otherwise fail.",
    )
    parser.add_argument(
        "--minimum-trades",
        type=int,
        default=None,
        help="Optional explicit minimum validation trades override.",
    )
    parser.add_argument(
        "--minimum-trades-per-day",
        type=float,
        default=MINIMUM_POLICY_TRADES_PER_DAY,
        help="Default strict policy floor expressed as minimum trades per day of validation span.",
    )
    parser.add_argument("--skip-latest-publish", action="store_true", help="Skip publishing the deployable latest/ artifact set.")
    parser.add_argument("--no-progress", action="store_true", help="Disable fallback dataset-load progress bars.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.exists():
        parser.exit(2, f'error: run directory does not exist: "{run_dir}"\n')

    dataset_cache_dir = Path(args.dataset_cache_dir).expanduser() if args.dataset_cache_dir else None
    if dataset_cache_dir is not None and not dataset_cache_dir.exists():
        parser.exit(2, f'error: dataset cache directory does not exist: "{dataset_cache_dir}"\n')

    try:
        split_manifest = _load_json(run_dir / "split_manifest.json")
        summary = _load_json(run_dir / "summary.json")
        policy_search_payload = _load_json_list(run_dir / "policy_search.json")
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    validation_tickers = _manifest_tickers(split_manifest, "validation_tickers")
    test_tickers = _manifest_tickers(split_manifest, "test_tickers")

    print(f"Fast policy reselection for: {run_dir}")
    print(f"Strict policy selection: {args.strict_policy_selection}")
    if args.minimum_trades is not None:
        print(f"Minimum validation trades override: {args.minimum_trades}")
    else:
        print(f"Minimum validation trades per day: {args.minimum_trades_per_day}")

    validation_frame = _load_split_frame(
        run_dir,
        "validation",
        summary=summary,
        dataset_cache_dir=dataset_cache_dir,
        tickers=validation_tickers,
        progress_desc=None if args.no_progress else "Validation dataset fallback",
    )
    test_frame = _load_split_frame(
        run_dir,
        "test",
        summary=summary,
        dataset_cache_dir=dataset_cache_dir,
        tickers=test_tickers,
        progress_desc=None if args.no_progress else "Test dataset fallback",
    )

    minimum_validation_trades = (
        int(args.minimum_trades)
        if args.minimum_trades is not None
        else minimum_policy_trades_for_frame(validation_frame, trades_per_day=float(args.minimum_trades_per_day))
    )
    print(f"Minimum validation trades required: {minimum_validation_trades}")

    policy_results = [_policy_result_from_payload(payload) for payload in policy_search_payload]
    selected_policy, met_validation_requirements = select_policy_candidate(
        policy_results,
        minimum_trades=minimum_validation_trades,
        fallback_to_best_overall=not args.strict_policy_selection,
    )
    policy_selection_mode = "strict" if met_validation_requirements else "fallback_best_overall"
    print(
        "Selected validation policy:",
        f"mode={policy_selection_mode}",
        f"trades={selected_policy.trades}",
        f"net_pnl=${selected_policy.net_pnl_dollars:.4f}",
        f"objective={selected_policy.objective:.6f}",
    )

    refreshed = _refresh_policy_outputs(
        run_dir,
        validation_frame=validation_frame,
        test_frame=test_frame,
        selected_policy=selected_policy,
        policy_selection_mode=policy_selection_mode,
        met_validation_requirements=met_validation_requirements,
        minimum_validation_trades=minimum_validation_trades,
    )

    updated_summary = dict(summary)
    updated_summary["minimum_validation_trades_required"] = minimum_validation_trades
    updated_summary["validation_log_loss_calibrated"] = refreshed["validation_log_loss"]
    updated_summary["test_log_loss_calibrated"] = refreshed["test_log_loss"]
    updated_summary["policy"] = json.loads((run_dir / "policy.json").read_text(encoding="utf-8"))
    updated_summary["validation_metrics"] = refreshed["validation_metrics"]
    updated_summary["test_metrics"] = refreshed["test_metrics"]
    write_json(run_dir / "summary.json", updated_summary)

    if args.skip_latest_publish:
        print("Skipping latest artifact publication.")
    else:
        model_family = str(updated_summary.get("model_family", "lightgbm"))
        latest_dir = publish_latest_artifacts(run_dir, model_family=model_family)
        print(f"Latest artifacts: {latest_dir}")

    print("Fast policy reselection complete.")
    print(json.dumps(updated_summary, indent=2))


if __name__ == "__main__":
    main()
