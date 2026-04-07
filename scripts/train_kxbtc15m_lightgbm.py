from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import log_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.kalshi.offline_training import (  # noqa: E402
    DEFAULT_OPTUNA_SAMPLE_SIZE,
    DEFAULT_OPTUNA_TRIALS,
    HOURLY_CONTEXT_SERIES,
    HourlyContextConfig,
    build_feature_cache,
    build_prediction_export_frame,
    build_split_manifest,
    calibrate_validation_predictions,
    evaluate_policy,
    evaluate_walk_forward,
    infer_feature_names,
    labels,
    load_feature_dataset,
    load_markets_frame,
    minimum_policy_trades_for_frame,
    publish_latest_artifacts,
    raw_predictions,
    resolve_artifacts_dir,
    resolve_inputs,
    run_optuna_search,
    sample_train_subset,
    save_feature_manifest,
    save_lightgbm_artifacts,
    save_split_manifest,
    tune_policy,
    train_lightgbm_model,
    write_json,
)

load_dotenv()


def _load_reference_manifest(reference_run_dir: Path) -> dict[str, object]:
    path = reference_run_dir / "split_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Reference split manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_tickers(manifest: dict[str, object], key: str) -> tuple[str, ...]:
    values = manifest.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Split manifest is missing a valid {key} list.")
    return tuple(str(value) for value in values)


def _resolve_dataset_cache_dir(
    explicit_cache_dir: str | None,
    reference_run_dir: Path | None,
    artifacts_dir: Path,
    *,
    hourly_context_series: str | None = None,
) -> Path:
    if explicit_cache_dir:
        return Path(explicit_cache_dir).expanduser()
    if hourly_context_series:
        return artifacts_dir / "datasets" / f"all_{hourly_context_series.lower()}_atm_hourly_context"
    if reference_run_dir is not None:
        reference_cache_dir = reference_run_dir / "datasets" / "all"
        if reference_cache_dir.exists():
            return reference_cache_dir
    return artifacts_dir / "datasets" / "all"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the KXBTC15M LightGBM model pipeline.")
    parser.add_argument("--series", default="KXBTC15M")
    parser.add_argument("--markets-path", help="Path to the resolved markets parquet/csv")
    parser.add_argument("--trades-path", help="Path to the per-market trades directory or trades parquet")
    parser.add_argument("--dataset-cache-dir", help="Optional existing feature cache directory to reuse")
    parser.add_argument(
        "--reference-run-dir",
        help="Optional existing run directory whose split_manifest.json and cached dataset should be reused.",
    )
    parser.add_argument("--artifacts-root", help="Root directory for run artifacts")
    parser.add_argument("--run-name", help="Optional run directory name under the artifacts root")
    parser.add_argument("--optuna-trials", type=int, default=DEFAULT_OPTUNA_TRIALS)
    parser.add_argument("--optuna-sample-size", type=int, default=DEFAULT_OPTUNA_SAMPLE_SIZE)
    parser.add_argument(
        "--hourly-context-series",
        default=None,
        help=f"Optional hourly Kalshi BTC context series to join, for example {HOURLY_CONTEXT_SERIES}.",
    )
    parser.add_argument("--hourly-context-markets-path", help="Path to the hourly context markets parquet/csv")
    parser.add_argument("--hourly-context-trades-path", help="Path to the hourly context per-market trades directory or parquet")
    parser.add_argument("--hourly-context-max-staleness-seconds", type=float, default=300.0)
    parser.add_argument("--hourly-context-min-recent-trade-count-300s", type=float, default=1.0)
    parser.add_argument("--skip-walk-forward", action="store_true", help="Skip walk-forward evaluation.")
    parser.add_argument("--skip-latest-publish", action="store_true", help="Skip publishing the deployable latest/ artifact set.")
    parser.add_argument("--no-progress", action="store_true", help="Disable dataset loading progress bars.")
    args = parser.parse_args()

    reference_run_dir = Path(args.reference_run_dir).expanduser() if args.reference_run_dir else None
    if reference_run_dir is not None and not reference_run_dir.exists():
        parser.exit(2, f'error: reference run directory does not exist: "{reference_run_dir}"\n')

    artifacts_dir = resolve_artifacts_dir(args.artifacts_root, args.run_name)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    hourly_context_config: HourlyContextConfig | None = None
    context_markets_df: pd.DataFrame | None = None
    context_trades_path: Path | None = None
    if args.hourly_context_series:
        context_series = str(args.hourly_context_series)
        try:
            context_markets_path, context_trades_path = resolve_inputs(
                context_series,
                args.hourly_context_markets_path,
                args.hourly_context_trades_path,
            )
            context_markets_df = load_markets_frame(context_markets_path, context_series)
        except (FileNotFoundError, ValueError) as exc:
            parser.exit(2, f"error: {exc}\n")
        hourly_context_config = HourlyContextConfig(
            series_ticker=context_series,
            max_staleness_seconds=float(args.hourly_context_max_staleness_seconds),
            min_recent_trade_count_300s=float(args.hourly_context_min_recent_trade_count_300s),
        )

    dataset_cache_dir = _resolve_dataset_cache_dir(
        args.dataset_cache_dir,
        reference_run_dir,
        artifacts_dir,
        hourly_context_series=hourly_context_config.series_ticker if hourly_context_config is not None else None,
    )

    markets_df: pd.DataFrame | None = None
    trades_path: Path | None = None
    walk_forward_markets_df: pd.DataFrame | None = None
    manifest_payload: dict[str, object] | None = None
    should_prepare_inputs = reference_run_dir is None or not dataset_cache_dir.exists() or bool(args.markets_path or args.trades_path)
    if should_prepare_inputs:
        try:
            markets_path, trades_path = resolve_inputs(args.series, args.markets_path, args.trades_path)
            markets_df = load_markets_frame(markets_path, args.series)
        except (FileNotFoundError, ValueError) as exc:
            parser.exit(2, f"error: {exc}\n")
    if reference_run_dir is not None:
        try:
            manifest_payload = _load_reference_manifest(reference_run_dir)
        except (FileNotFoundError, ValueError) as exc:
            parser.exit(2, f"error: {exc}\n")
    else:
        assert markets_df is not None
        manifest = build_split_manifest(markets_df, series_ticker=args.series)
        manifest_payload = {
            "series": manifest.series,
            "generated_at": manifest.generated_at,
            "train_fraction": manifest.train_fraction,
            "validation_fraction": manifest.validation_fraction,
            "test_fraction": manifest.test_fraction,
            "train_tickers": list(manifest.train_tickers),
            "validation_tickers": list(manifest.validation_tickers),
            "test_tickers": list(manifest.test_tickers),
        }

    assert manifest_payload is not None
    train_tickers = _manifest_tickers(manifest_payload, "train_tickers")
    validation_tickers = _manifest_tickers(manifest_payload, "validation_tickers")
    test_tickers = _manifest_tickers(manifest_payload, "test_tickers")
    ordered_tickers = train_tickers + validation_tickers + test_tickers
    walk_forward_markets_df = pd.DataFrame({"ticker": list(ordered_tickers)})

    print(f"Series: {args.series}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Dataset cache: {dataset_cache_dir}")
    print(f"Reference run: {reference_run_dir if reference_run_dir is not None else 'none'}")
    print(f"Progress bars: {not args.no_progress}")
    print(
        "Hourly context: "
        f"{hourly_context_config.series_ticker if hourly_context_config is not None else 'disabled'}"
    )

    write_json(artifacts_dir / "split_manifest.json", manifest_payload)

    if not dataset_cache_dir.exists():
        if markets_df is None or trades_path is None:
            parser.exit(
                2,
                "error: dataset cache does not exist and target markets/trades could not be resolved.\n",
            )
        print("Building feature cache...")
        build_feature_cache(
            markets_df,
            trades_path,
            dataset_cache_dir,
            hourly_context=hourly_context_config,
            context_markets_df=context_markets_df,
            context_trades_path=context_trades_path,
        )

    print("Loading datasets...")
    train_df = load_feature_dataset(
        dataset_cache_dir,
        train_tickers,
        progress_desc=None if args.no_progress else "Train dataset",
    )
    validation_df = load_feature_dataset(
        dataset_cache_dir,
        validation_tickers,
        progress_desc=None if args.no_progress else "Validation dataset",
    )
    test_df = load_feature_dataset(
        dataset_cache_dir,
        test_tickers,
        progress_desc=None if args.no_progress else "Test dataset",
    )
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Test rows: {len(test_df)}")
    feature_names = infer_feature_names(train_df)
    save_feature_manifest(
        artifacts_dir / "feature_manifest.json",
        feature_order=feature_names,
        metadata={
            "hourly_context_series": hourly_context_config.series_ticker if hourly_context_config is not None else None,
        },
    )

    train_subset = sample_train_subset(train_df, args.optuna_sample_size)
    best_params, trial_history = run_optuna_search(train_subset, validation_df, n_trials=args.optuna_trials)
    write_json(
        artifacts_dir / "lightgbm" / "optuna_history.json",
        {
            "sample_rows": int(len(train_subset)),
            "trials": trial_history,
            "best_params": best_params,
        },
    )

    model, training_metrics = train_lightgbm_model(train_df, validation_df, best_params)
    save_lightgbm_artifacts(artifacts_dir, model, training_metrics)

    validation_raw = raw_predictions(model, validation_df)
    validation_calibrated = calibrate_validation_predictions(artifacts_dir, validation_df, validation_raw)
    validation_log_loss = float(log_loss(labels(validation_df), validation_calibrated, labels=[0, 1]))
    minimum_validation_trades = minimum_policy_trades_for_frame(validation_df)

    best_policy, grid_results = tune_policy(validation_df, validation_calibrated, validation_log_loss)
    write_json(artifacts_dir / "policy_search.json", grid_results)
    write_json(
        artifacts_dir / "policy.json",
        {
            "config": best_policy.config.__dict__,
            "objective": best_policy.objective,
            "trades": best_policy.trades,
            "net_pnl_dollars": best_policy.net_pnl_dollars,
            "max_drawdown_dollars": best_policy.max_drawdown_dollars,
            "max_drawdown_pct": best_policy.max_drawdown_pct,
            "return_pct": best_policy.return_pct,
            "log_loss": best_policy.log_loss,
            "minimum_trades_required": minimum_validation_trades,
        },
    )

    from src.live.kalshi.offline_training import apply_calibration_to_predictions  # noqa: E402

    test_raw = raw_predictions(model, test_df)
    test_calibrated = apply_calibration_to_predictions(artifacts_dir, test_raw)
    test_log_loss = float(log_loss(labels(test_df), test_calibrated, labels=[0, 1]))
    test_metrics = evaluate_policy(test_df, test_calibrated, best_policy.config, test_log_loss)
    write_json(artifacts_dir / "test_metrics.json", test_metrics)

    test_predictions = build_prediction_export_frame(test_df, test_raw, test_calibrated)
    test_predictions.to_parquet(artifacts_dir / "test_predictions.parquet", index=False)

    walk_forward: list[dict[str, object]] | None = None
    if args.skip_walk_forward:
        print("Skipping walk-forward evaluation.")
    else:
        print("Running walk-forward evaluation...")
        walk_forward = evaluate_walk_forward(dataset_cache_dir, walk_forward_markets_df, best_params)
        write_json(artifacts_dir / "walk_forward.json", walk_forward)

    summary = {
        "series": args.series,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "feature_count": len(feature_names),
        "best_params": best_params,
        "minimum_validation_trades_required": minimum_validation_trades,
        "validation_log_loss_calibrated": validation_log_loss,
        "test_log_loss_calibrated": test_log_loss,
        "policy": json.loads((artifacts_dir / "policy.json").read_text(encoding="utf-8")),
        "test_metrics": test_metrics,
        "hourly_context_series": hourly_context_config.series_ticker if hourly_context_config is not None else None,
        "walk_forward_completed": walk_forward is not None,
    }
    write_json(artifacts_dir / "summary.json", summary)
    latest_dir: Path | None = None
    if args.skip_latest_publish:
        print("Skipping latest artifact publication.")
    else:
        latest_dir = publish_latest_artifacts(artifacts_dir)

    print("Training complete.")
    if latest_dir is not None:
        print(f"Latest artifacts: {latest_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
