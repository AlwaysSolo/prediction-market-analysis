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

from src.live.kalshi.features import FEATURE_ORDER  # noqa: E402
from src.live.kalshi.offline_training import (  # noqa: E402
    MINIMUM_POLICY_TRADES_PER_DAY,
    apply_calibration_to_predictions,
    build_policy_diagnostics,
    build_prediction_export_frame,
    booster_raw_predictions,
    evaluate_walk_forward,
    labels,
    load_feature_dataset,
    minimum_policy_trades_for_frame,
    publish_latest_artifacts,
    select_policy_candidate,
    serialize_policy_result,
    sweep_policy_grid,
    write_json,
)

load_dotenv()


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_test_predictions(
    split_name: str,
    run_dir: Path,
    split_df: pd.DataFrame,
    raw_probabilities,
    calibrated_probabilities,
) -> None:
    predictions = build_prediction_export_frame(split_df, raw_probabilities, calibrated_probabilities)
    predictions.to_parquet(run_dir / f"{split_name}_predictions.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume the evaluation phase for an existing KXBTC15M LightGBM run.")
    parser.add_argument("--run-dir", required=True, help="Existing run directory under artifacts/kalshi/kxbtc15m_lightgbm/")
    parser.add_argument(
        "--strict-policy-selection",
        action="store_true",
        help="Require a validation policy with positive PnL and the minimum trade count; otherwise fail.",
    )
    parser.add_argument(
        "--minimum-trades",
        type=int,
        default=None,
        help="Optional explicit minimum validation trades override for strict policy selection.",
    )
    parser.add_argument(
        "--minimum-trades-per-day",
        type=float,
        default=MINIMUM_POLICY_TRADES_PER_DAY,
        help="Default strict policy floor expressed as minimum trades per day of validation span.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    parser.add_argument("--skip-walk-forward", action="store_true", help="Skip walk-forward evaluation.")
    parser.add_argument("--skip-latest-publish", action="store_true", help="Skip publishing the deployable latest/ artifact set.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.exists():
        parser.exit(2, f'error: run directory does not exist: "{run_dir}"\n')

    try:
        split_manifest = _load_json(run_dir / "split_manifest.json")
        metrics = _load_json(run_dir / "lightgbm" / "metrics.json")
    except FileNotFoundError as exc:
        parser.exit(2, f"error: {exc}\n")

    calibration_path = run_dir / "lightgbm" / "calibration.json"
    model_path = run_dir / "lightgbm" / "model.txt"
    cache_dir = run_dir / "datasets" / "all"
    if not calibration_path.exists():
        parser.exit(2, f"error: required artifact not found: {calibration_path}\n")
    if not model_path.exists():
        parser.exit(2, f"error: required artifact not found: {model_path}\n")
    if not cache_dir.exists():
        parser.exit(2, f"error: required cached dataset directory not found: {cache_dir}\n")

    train_tickers = tuple(str(ticker) for ticker in split_manifest["train_tickers"])
    validation_tickers = tuple(str(ticker) for ticker in split_manifest["validation_tickers"])
    test_tickers = tuple(str(ticker) for ticker in split_manifest["test_tickers"])
    all_tickers = train_tickers + validation_tickers + test_tickers

    best_iteration = metrics.get("best_iteration")
    best_iteration = int(best_iteration) if best_iteration is not None else None
    best_params = metrics.get("params")
    if not isinstance(best_params, dict):
        optuna_path = run_dir / "lightgbm" / "optuna_history.json"
        if optuna_path.exists():
            optuna_payload = _load_json(optuna_path)
            candidate = optuna_payload.get("best_params")
            if isinstance(candidate, dict):
                best_params = candidate
    if not args.skip_walk_forward and not isinstance(best_params, dict):
        parser.exit(2, "error: best LightGBM params are required to run walk-forward evaluation.\n")

    print(f"Resuming evaluation for: {run_dir}")
    print(f"Validation tickers: {len(validation_tickers)}")
    print(f"Test tickers: {len(test_tickers)}")
    print(f"Best iteration: {best_iteration}")
    print(f"Strict policy selection: {args.strict_policy_selection}")
    if args.minimum_trades is not None:
        print(f"Minimum validation trades override: {args.minimum_trades}")
    else:
        print(f"Minimum validation trades per day: {args.minimum_trades_per_day}")
    print(f"Progress bars: {not args.no_progress}")

    print("Loading validation dataset...")
    validation_df = load_feature_dataset(
        cache_dir,
        validation_tickers,
        progress_desc=None if args.no_progress else "Validation dataset",
    )
    print(f"Validation rows: {len(validation_df)}")

    print("Scoring validation split from saved model...")
    booster = Booster(model_file=str(model_path))
    validation_raw = booster_raw_predictions(booster, validation_df, best_iteration=best_iteration)
    validation_calibrated = apply_calibration_to_predictions(run_dir, validation_raw)
    validation_log_loss = float(log_loss(labels(validation_df), validation_calibrated, labels=[0, 1]))
    minimum_validation_trades = (
        int(args.minimum_trades)
        if args.minimum_trades is not None
        else minimum_policy_trades_for_frame(validation_df, trades_per_day=float(args.minimum_trades_per_day))
    )
    print(f"Validation calibrated log-loss: {validation_log_loss:.6f}")
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
        fallback_to_best_overall=not args.strict_policy_selection,
    )
    policy_selection_mode = "strict" if met_validation_requirements else "fallback_best_overall"
    print(
        "Selected validation policy:",
        f"mode={policy_selection_mode}",
        f"trades={best_policy.trades}",
        f"net_pnl=${best_policy.net_pnl_dollars:.4f}",
        f"objective={best_policy.objective:.6f}",
    )

    write_json(run_dir / "policy_search.json", [serialize_policy_result(result) for result in policy_results])
    write_json(
        run_dir / "policy.json",
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
    write_json(run_dir / "validation_metrics.json", validation_metrics)
    write_json(run_dir / "validation_diagnostics.json", validation_diagnostics)
    _write_test_predictions("validation", run_dir, validation_df, validation_raw, validation_calibrated)
    validation_trade_records.to_parquet(run_dir / "validation_trade_records.parquet", index=False)

    print("Loading test dataset...")
    test_df = load_feature_dataset(
        cache_dir,
        test_tickers,
        progress_desc=None if args.no_progress else "Test dataset",
    )
    print(f"Test rows: {len(test_df)}")

    print("Scoring untouched test split...")
    test_raw = booster_raw_predictions(booster, test_df, best_iteration=best_iteration)
    test_calibrated = apply_calibration_to_predictions(run_dir, test_raw)
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
    write_json(run_dir / "test_metrics.json", test_metrics)
    write_json(run_dir / "test_diagnostics.json", test_diagnostics)
    _write_test_predictions("test", run_dir, test_df, test_raw, test_calibrated)
    test_trade_records.to_parquet(run_dir / "test_trade_records.parquet", index=False)
    print(
        "Test policy result:",
        f"trades={test_metrics['trades']}",
        f"net_pnl=${float(test_metrics['net_pnl_dollars']):.4f}",
        f"objective={float(test_metrics['objective']):.6f}",
    )

    walk_forward: list[dict[str, object]] | None = None
    if args.skip_walk_forward:
        print("Skipping walk-forward evaluation.")
    else:
        print("Running walk-forward evaluation...")
        walk_forward_markets = pd.DataFrame({"ticker": list(all_tickers)})
        walk_forward = evaluate_walk_forward(
            cache_dir,
            walk_forward_markets,
            best_params,
            fallback_to_best_overall_policy=not args.strict_policy_selection,
            progress_desc=None if args.no_progress else "Walk-forward",
        )
        write_json(run_dir / "walk_forward.json", walk_forward)
        print(f"Walk-forward folds completed: {len(walk_forward)}")

    summary = {
        "series": str(split_manifest.get("series", "KXBTC15M")),
        "train_rows": int(metrics["train_rows"]),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "feature_count": len(FEATURE_ORDER),
        "best_params": best_params,
        "best_iteration": best_iteration,
        "minimum_validation_trades_required": minimum_validation_trades,
        "validation_log_loss_calibrated": validation_log_loss,
        "test_log_loss_calibrated": test_log_loss,
        "policy": json.loads((run_dir / "policy.json").read_text(encoding="utf-8")),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "walk_forward_completed": walk_forward is not None,
    }
    write_json(run_dir / "summary.json", summary)

    if args.skip_latest_publish:
        print("Skipping latest artifact publication.")
    else:
        latest_dir = publish_latest_artifacts(run_dir)
        print(f"Latest artifacts: {latest_dir}")

    print("Resume evaluation complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
