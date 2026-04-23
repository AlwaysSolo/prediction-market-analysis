from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import log_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.kalshi.offline_training import (  # noqa: E402
    PolicyConfig,
    build_walk_forward_folds,
    evaluate_policy,
    fit_platt_scaler,
    labels,
    load_feature_dataset,
    raw_predictions,
    train_lightgbm_model,
)


@dataclass(frozen=True)
class RunBundle:
    label: str
    run_dir: Path
    cache_dir: Path
    feature_schema: str
    best_params: dict[str, object]
    policy_config: PolicyConfig
    ordered_tickers: tuple[str, ...]


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _manifest_tickers(manifest: dict[str, object], key: str) -> tuple[str, ...]:
    values = manifest.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Split manifest is missing a valid {key} list.")
    return tuple(str(value) for value in values)


def _ordered_tickers_from_manifest(manifest: dict[str, object]) -> tuple[str, ...]:
    return (
        _manifest_tickers(manifest, "train_tickers")
        + _manifest_tickers(manifest, "validation_tickers")
        + _manifest_tickers(manifest, "test_tickers")
    )


def _resolve_cache_dir(run_dir: Path, explicit_cache_dir: str | None) -> Path:
    if explicit_cache_dir:
        return Path(explicit_cache_dir).expanduser()
    return run_dir / "datasets" / "all"


def _load_run_bundle(run_dir: Path, *, cache_dir: str | None = None, label: str | None = None) -> RunBundle:
    summary = _load_json(run_dir / "summary.json")
    model_family = str(summary.get("model_family", "lightgbm"))
    if model_family != "lightgbm":
        raise ValueError(f"Only LightGBM runs are supported for fixed-policy walk-forward comparison, got {model_family}.")
    best_params = summary.get("best_params")
    if not isinstance(best_params, dict):
        raise ValueError(f"Run is missing best_params in summary.json: {run_dir}")
    feature_schema = str(summary.get("feature_schema", "default"))
    policy_payload = _load_json(run_dir / "policy.json")
    config_payload = policy_payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError(f"Run policy.json is missing a valid config object: {run_dir}")
    manifest = _load_json(run_dir / "split_manifest.json")
    ordered_tickers = _ordered_tickers_from_manifest(manifest)
    resolved_cache_dir = _resolve_cache_dir(run_dir, cache_dir)
    if not resolved_cache_dir.exists():
        raise FileNotFoundError(f"Dataset cache directory does not exist: {resolved_cache_dir}")
    return RunBundle(
        label=label or run_dir.name,
        run_dir=run_dir,
        cache_dir=resolved_cache_dir,
        feature_schema=feature_schema,
        best_params=best_params,
        policy_config=PolicyConfig(**config_payload),
        ordered_tickers=ordered_tickers,
    )


def _score_bundle_fold(bundle: RunBundle, fold: Any) -> dict[str, object]:
    train_df = load_feature_dataset(
        bundle.cache_dir,
        fold.train_tickers,
        progress_desc=f"{bundle.label} fold {fold.fold_index} train",
        feature_schema=bundle.feature_schema,
    )
    validation_df = load_feature_dataset(
        bundle.cache_dir,
        fold.validation_tickers,
        progress_desc=f"{bundle.label} fold {fold.fold_index} validation",
        feature_schema=bundle.feature_schema,
    )
    test_df = load_feature_dataset(
        bundle.cache_dir,
        fold.test_tickers,
        progress_desc=f"{bundle.label} fold {fold.fold_index} test",
        feature_schema=bundle.feature_schema,
    )
    model, _metrics = train_lightgbm_model(train_df, validation_df, bundle.best_params)
    validation_raw = raw_predictions(model, validation_df)
    calibration = fit_platt_scaler(validation_raw, labels(validation_df))
    validation_calibrated = calibration.apply(validation_raw)  # type: ignore[assignment]
    validation_log_loss = float(log_loss(labels(validation_df), validation_calibrated, labels=[0, 1]))
    test_raw = raw_predictions(model, test_df)
    test_calibrated = calibration.apply(test_raw)  # type: ignore[assignment]
    test_log_loss = float(log_loss(labels(test_df), test_calibrated, labels=[0, 1]))
    return {
        "validation_log_loss": validation_log_loss,
        "test_log_loss": test_log_loss,
        "test_df": test_df,
        "test_calibrated": test_calibrated,
    }


def _aggregate_fold_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["model_label"]), str(row["policy_label"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    for (model_label, policy_label), group in grouped.items():
        summaries.append(
            {
                "model_label": model_label,
                "policy_label": policy_label,
                "fold_count": len(group),
                "mean_validation_log_loss": float(sum(float(item["validation_log_loss"]) for item in group) / len(group)),
                "mean_test_log_loss": float(sum(float(item["test_log_loss"]) for item in group) / len(group)),
                "total_test_trades": int(sum(int(item["trades"]) for item in group)),
                "total_test_pnl_dollars": float(sum(float(item["net_pnl_dollars"]) for item in group)),
                "worst_fold_test_pnl_dollars": float(min(float(item["net_pnl_dollars"]) for item in group)),
                "best_fold_test_pnl_dollars": float(max(float(item["net_pnl_dollars"]) for item in group)),
                "worst_fold_drawdown_dollars": float(max(float(item["max_drawdown_dollars"]) for item in group)),
            }
        )
    summaries.sort(key=lambda item: (item["policy_label"], item["model_label"]))
    return summaries


def _pairwise_policy_summary(
    aggregate_rows: list[dict[str, object]],
    *,
    left_label: str,
    right_label: str,
) -> list[dict[str, object]]:
    by_key = {(str(row["model_label"]), str(row["policy_label"])): row for row in aggregate_rows}
    policy_labels = sorted({str(row["policy_label"]) for row in aggregate_rows})
    comparisons: list[dict[str, object]] = []
    for policy_label in policy_labels:
        left_row = by_key.get((left_label, policy_label))
        right_row = by_key.get((right_label, policy_label))
        if left_row is None or right_row is None:
            continue
        comparisons.append(
            {
                "policy_label": policy_label,
                "left_label": left_label,
                "right_label": right_label,
                "left_mean_test_log_loss": left_row["mean_test_log_loss"],
                "right_mean_test_log_loss": right_row["mean_test_log_loss"],
                "left_minus_right_mean_test_log_loss": float(left_row["mean_test_log_loss"]) - float(right_row["mean_test_log_loss"]),
                "left_total_test_pnl_dollars": left_row["total_test_pnl_dollars"],
                "right_total_test_pnl_dollars": right_row["total_test_pnl_dollars"],
                "left_minus_right_total_test_pnl_dollars": float(left_row["total_test_pnl_dollars"]) - float(right_row["total_test_pnl_dollars"]),
            }
        )
    return comparisons


def run_fixed_policy_walkforward_comparison(
    left_bundle: RunBundle,
    right_bundle: RunBundle,
) -> dict[str, object]:
    if left_bundle.ordered_tickers != right_bundle.ordered_tickers:
        raise ValueError("Run bundles must share the same ordered split manifest tickers for walk-forward comparison.")

    folds = build_walk_forward_folds(pd.DataFrame({"ticker": list(left_bundle.ordered_tickers)}))
    fixed_policies = {
        f"{left_bundle.label}_policy": left_bundle.policy_config,
        f"{right_bundle.label}_policy": right_bundle.policy_config,
    }
    fold_results: list[dict[str, object]] = []
    for fold in folds:
        left_scored = _score_bundle_fold(left_bundle, fold)
        right_scored = _score_bundle_fold(right_bundle, fold)
        for bundle, scored in ((left_bundle, left_scored), (right_bundle, right_scored)):
            for policy_label, policy_config in fixed_policies.items():
                metrics = evaluate_policy(
                    scored["test_df"],
                    scored["test_calibrated"],
                    policy_config,
                    float(scored["test_log_loss"]),
                    progress_desc=f"{bundle.label} fold {fold.fold_index} {policy_label}",
                )
                fold_results.append(
                    {
                        "fold_index": int(fold.fold_index),
                        "model_label": bundle.label,
                        "feature_schema": bundle.feature_schema,
                        "policy_label": policy_label,
                        "policy_config": asdict(policy_config),
                        "validation_log_loss": float(scored["validation_log_loss"]),
                        "test_log_loss": float(scored["test_log_loss"]),
                        **metrics,
                    }
                )
    aggregate_rows = _aggregate_fold_results(fold_results)
    return {
        "left_bundle": {
            "label": left_bundle.label,
            "run_dir": str(left_bundle.run_dir),
            "cache_dir": str(left_bundle.cache_dir),
            "feature_schema": left_bundle.feature_schema,
            "policy_config": asdict(left_bundle.policy_config),
        },
        "right_bundle": {
            "label": right_bundle.label,
            "run_dir": str(right_bundle.run_dir),
            "cache_dir": str(right_bundle.cache_dir),
            "feature_schema": right_bundle.feature_schema,
            "policy_config": asdict(right_bundle.policy_config),
        },
        "fold_results": fold_results,
        "aggregate_results": aggregate_rows,
        "pairwise_policy_summary": _pairwise_policy_summary(
            aggregate_rows,
            left_label=left_bundle.label,
            right_label=right_bundle.label,
        ),
    }


def _print_summary(payload: dict[str, object]) -> None:
    print("policy_label | left_minus_right_mean_test_log_loss | left_minus_right_total_test_pnl_dollars")
    print("-------------------------------------------------------------------------------------------")
    for row in payload.get("pairwise_policy_summary", []):
        if not isinstance(row, dict):
            continue
        print(
            f"{row['policy_label']} | "
            f"{float(row['left_minus_right_mean_test_log_loss']):.6f} | "
            f"{float(row['left_minus_right_total_test_pnl_dollars']):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two LightGBM walk-forward runs under fixed policies.")
    parser.add_argument("--left-run-dir", required=True, help="Run directory for the first model.")
    parser.add_argument("--right-run-dir", required=True, help="Run directory for the second model.")
    parser.add_argument("--left-cache-dir", help="Optional dataset cache directory for the first run.")
    parser.add_argument("--right-cache-dir", help="Optional dataset cache directory for the second run.")
    parser.add_argument("--left-label", help="Optional display label for the first run.")
    parser.add_argument("--right-label", help="Optional display label for the second run.")
    parser.add_argument("--output-path", help="Optional JSON output path.")
    args = parser.parse_args()

    left_bundle = _load_run_bundle(
        Path(args.left_run_dir).expanduser(),
        cache_dir=args.left_cache_dir,
        label=args.left_label,
    )
    right_bundle = _load_run_bundle(
        Path(args.right_run_dir).expanduser(),
        cache_dir=args.right_cache_dir,
        label=args.right_label,
    )
    payload = run_fixed_policy_walkforward_comparison(left_bundle, right_bundle)
    output_path = (
        Path(args.output_path).expanduser()
        if args.output_path
        else left_bundle.run_dir.parent / f"{left_bundle.label}__vs__{right_bundle.label}__fixed_policy_walkforward.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_summary(payload)
    print()
    print(f"Wrote comparison to {output_path}")


if __name__ == "__main__":
    main()
