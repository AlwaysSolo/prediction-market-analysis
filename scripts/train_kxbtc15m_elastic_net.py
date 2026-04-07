from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import log_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.kalshi.offline_training import (  # noqa: E402
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_ELASTIC_NET_ARTIFACTS_ROOT,
    DEFAULT_ELASTIC_NET_C_VALUES,
    DEFAULT_ELASTIC_NET_L1_RATIOS,
    DEFAULT_ELASTIC_NET_MAX_ITER,
    DEFAULT_ELASTIC_NET_SAMPLE_SIZE,
    DEFAULT_ELASTIC_NET_TOL,
    FEATURE_SCHEMA_CHOICES,
    HOURLY_CONTEXT_SERIES,
    HourlyContextConfig,
    LINEAR_V1_FEATURE_SCHEMA,
    apply_calibration_to_predictions,
    build_feature_cache,
    build_policy_diagnostics,
    build_prediction_export_frame,
    build_split_manifest,
    calibrate_validation_predictions,
    elastic_net_raw_predictions,
    evaluate_walk_forward_elastic_net,
    feature_schema_metadata,
    labels,
    load_elastic_net_artifact,
    load_feature_dataset,
    load_markets_frame,
    minimum_policy_trades_for_frame,
    normalize_feature_schema,
    publish_latest_artifacts,
    infer_feature_names,
    resolve_artifacts_dir,
    resolve_inputs,
    run_elastic_net_search,
    sample_train_subset,
    save_elastic_net_artifacts,
    save_feature_manifest,
    select_policy_candidate,
    serialize_policy_result,
    sweep_policy_grid,
    train_elastic_net_model,
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


def _write_predictions(
    split_name: str,
    artifacts_dir: Path,
    split_df: pd.DataFrame,
    raw_probabilities,
    calibrated_probabilities,
) -> None:
    predictions = build_prediction_export_frame(split_df, raw_probabilities, calibrated_probabilities)
    predictions.to_parquet(artifacts_dir / f"{split_name}_predictions.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the KXBTC15M Elastic Net pipeline (elastic-net logistic regression)."
    )
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
    parser.add_argument("--elastic-net-sample-size", type=int, default=DEFAULT_ELASTIC_NET_SAMPLE_SIZE)
    parser.add_argument("--elastic-net-c-values", type=float, nargs="+", default=list(DEFAULT_ELASTIC_NET_C_VALUES))
    parser.add_argument(
        "--elastic-net-l1-ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_ELASTIC_NET_L1_RATIOS),
        help="Candidate elastic-net l1_ratio values between 0 and 1.",
    )
    parser.add_argument("--elastic-net-max-iter", type=int, default=DEFAULT_ELASTIC_NET_MAX_ITER)
    parser.add_argument("--elastic-net-tol", type=float, default=DEFAULT_ELASTIC_NET_TOL)
    parser.add_argument(
        "--hourly-context-series",
        default=None,
        help=f"Optional hourly Kalshi BTC context series to join, for example {HOURLY_CONTEXT_SERIES}.",
    )
    parser.add_argument("--hourly-context-markets-path", help="Path to the hourly context markets parquet/csv")
    parser.add_argument("--hourly-context-trades-path", help="Path to the hourly context per-market trades directory or parquet")
    parser.add_argument("--hourly-context-max-staleness-seconds", type=float, default=300.0)
    parser.add_argument("--hourly-context-min-recent-trade-count-300s", type=float, default=1.0)
    parser.add_argument(
        "--feature-schema",
        choices=list(FEATURE_SCHEMA_CHOICES),
        default=DEFAULT_FEATURE_SCHEMA,
        help="Feature projection to use for model training.",
    )
    parser.add_argument("--skip-walk-forward", action="store_true", help="Skip walk-forward evaluation.")
    parser.add_argument("--skip-latest-publish", action="store_true", help="Skip publishing the deployable latest/ artifact set.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    reference_run_dir = Path(args.reference_run_dir).expanduser() if args.reference_run_dir else None
    if reference_run_dir is not None and not reference_run_dir.exists():
        parser.exit(2, f'error: reference run directory does not exist: "{reference_run_dir}"\n')
    feature_schema = normalize_feature_schema(args.feature_schema)
    if feature_schema == LINEAR_V1_FEATURE_SCHEMA and not args.hourly_context_series:
        parser.exit(2, "error: --feature-schema linear_v1 requires --hourly-context-series.\n")

    artifacts_dir = resolve_artifacts_dir(
        args.artifacts_root,
        args.run_name,
        default_root=DEFAULT_ELASTIC_NET_ARTIFACTS_ROOT,
    )
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
    print(f"Feature schema: {feature_schema}")
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
        feature_schema=feature_schema,
    )
    validation_df = load_feature_dataset(
        dataset_cache_dir,
        validation_tickers,
        progress_desc=None if args.no_progress else "Validation dataset",
        feature_schema=feature_schema,
    )
    test_df = load_feature_dataset(
        dataset_cache_dir,
        test_tickers,
        progress_desc=None if args.no_progress else "Test dataset",
        feature_schema=feature_schema,
    )
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Test rows: {len(test_df)}")
    feature_names = infer_feature_names(train_df)
    save_feature_manifest(
        artifacts_dir / "feature_manifest.json",
        feature_order=feature_names,
        metadata={
            **feature_schema_metadata(feature_schema),
            "hourly_context_series": hourly_context_config.series_ticker if hourly_context_config is not None else None,
        },
    )

    train_subset = sample_train_subset(train_df, args.elastic_net_sample_size)
    print(
        "Searching Elastic Net over "
        f"{len(args.elastic_net_c_values) * len(args.elastic_net_l1_ratios)} candidates on {len(train_subset)} sampled rows..."
    )
    best_params, search_history = run_elastic_net_search(
        train_subset,
        validation_df,
        c_values=tuple(float(value) for value in args.elastic_net_c_values),
        l1_ratios=tuple(float(value) for value in args.elastic_net_l1_ratios),
        max_iter=args.elastic_net_max_iter,
        tol=args.elastic_net_tol,
        progress_desc=None if args.no_progress else "Elastic Net search",
    )
    write_json(
        artifacts_dir / "elastic_net" / "search_history.json",
        {
            "sample_rows": int(len(train_subset)),
            "trials": search_history,
            "best_params": best_params,
        },
    )

    print("Fitting final Elastic Net model on the full train split...")
    artifact, training_metrics = train_elastic_net_model(train_df, validation_df, best_params)
    save_elastic_net_artifacts(artifacts_dir, artifact, training_metrics)

    print("Calibrating validation probabilities...")
    validation_raw = elastic_net_raw_predictions(artifact, validation_df)
    validation_calibrated = calibrate_validation_predictions(
        artifacts_dir,
        validation_df,
        validation_raw,
        model_family="elastic_net",
    )
    validation_log_loss = float(log_loss(labels(validation_df), validation_calibrated, labels=[0, 1]))
    minimum_validation_trades = minimum_policy_trades_for_frame(validation_df)
    print(f"Minimum validation trades required: {minimum_validation_trades}")

    print("Sweeping validation policy grid...")
    policy_results = sweep_policy_grid(
        validation_df,
        validation_calibrated,
        validation_log_loss,
        progress_desc=None if args.no_progress else "Validation policy sweep",
    )
    best_policy, met_validation_requirements = select_policy_candidate(
        policy_results,
        minimum_trades=minimum_validation_trades,
        fallback_to_best_overall=True,
    )
    policy_selection_mode = "strict" if met_validation_requirements else "fallback_best_overall"
    write_json(artifacts_dir / "policy_search.json", [serialize_policy_result(result) for result in policy_results])
    write_json(
        artifacts_dir / "policy.json",
        serialize_policy_result(
            best_policy,
            selection_mode=policy_selection_mode,
            met_validation_requirements=met_validation_requirements,
            minimum_trades_required=minimum_validation_trades,
        ),
    )

    validation_diagnostics, validation_trade_records = build_policy_diagnostics(
        validation_df,
        validation_raw,
        validation_calibrated,
        best_policy.config,
        overall_log_loss=validation_log_loss,
        progress_desc=None if args.no_progress else "Validation policy diagnostics",
    )
    validation_metrics = dict(validation_diagnostics["policy_metrics"])
    validation_metrics["selection_mode"] = policy_selection_mode
    validation_metrics["met_validation_requirements"] = met_validation_requirements
    validation_metrics["minimum_validation_trades_required"] = minimum_validation_trades
    write_json(artifacts_dir / "validation_metrics.json", validation_metrics)
    write_json(artifacts_dir / "validation_diagnostics.json", validation_diagnostics)
    _write_predictions("validation", artifacts_dir, validation_df, validation_raw, validation_calibrated)
    validation_trade_records.to_parquet(artifacts_dir / "validation_trade_records.parquet", index=False)

    print("Scoring untouched test split...")
    test_raw = elastic_net_raw_predictions(artifact, test_df)
    test_calibrated = apply_calibration_to_predictions(artifacts_dir, test_raw, model_family="elastic_net")
    test_log_loss = float(log_loss(labels(test_df), test_calibrated, labels=[0, 1]))
    test_diagnostics, test_trade_records = build_policy_diagnostics(
        test_df,
        test_raw,
        test_calibrated,
        best_policy.config,
        overall_log_loss=test_log_loss,
        progress_desc=None if args.no_progress else "Test policy diagnostics",
    )
    test_metrics = dict(test_diagnostics["policy_metrics"])
    test_metrics["selection_mode"] = policy_selection_mode
    test_metrics["met_validation_requirements"] = met_validation_requirements
    test_metrics["minimum_validation_trades_required"] = minimum_validation_trades
    write_json(artifacts_dir / "test_metrics.json", test_metrics)
    write_json(artifacts_dir / "test_diagnostics.json", test_diagnostics)
    _write_predictions("test", artifacts_dir, test_df, test_raw, test_calibrated)
    test_trade_records.to_parquet(artifacts_dir / "test_trade_records.parquet", index=False)

    walk_forward: list[dict[str, object]] | None = None
    if args.skip_walk_forward:
        print("Skipping walk-forward evaluation.")
    else:
        print("Running walk-forward evaluation...")
        walk_forward = evaluate_walk_forward_elastic_net(
            dataset_cache_dir,
            walk_forward_markets_df,
            best_params,
            fallback_to_best_overall_policy=True,
            progress_desc=None if args.no_progress else "Walk-forward",
        )
        write_json(artifacts_dir / "walk_forward.json", walk_forward)

    summary = {
        "series": args.series,
        "model_family": "elastic_net",
        "model_type": "elastic_net_logistic_regression",
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "feature_count": len(feature_names),
        "best_params": best_params,
        "best_iteration": None,
        "minimum_validation_trades_required": minimum_validation_trades,
        "validation_log_loss_calibrated": validation_log_loss,
        "test_log_loss_calibrated": test_log_loss,
        "policy": json.loads((artifacts_dir / "policy.json").read_text(encoding="utf-8")),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "walk_forward_completed": walk_forward is not None,
        "hourly_context_series": hourly_context_config.series_ticker if hourly_context_config is not None else None,
    }
    write_json(artifacts_dir / "summary.json", summary)

    if args.skip_latest_publish:
        print("Skipping latest artifact publication.")
    else:
        latest_dir = publish_latest_artifacts(artifacts_dir, model_family="elastic_net")
        print(f"Latest artifacts: {latest_dir}")

    print("Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
