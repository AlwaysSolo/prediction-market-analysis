from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.kalshi.offline_training import PolicyConfig
from src.live.kalshi.policy_mapping_analysis import (
    annotate_policy_gate_status,
    bootstrap_bucket_delta_correlation_by_group,
    build_common_prediction_frame,
    compute_slice_log_loss_matrix,
    summarize_bucket_deltas,
    summarize_threshold_crossings,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_policy(run_dir: Path) -> PolicyConfig:
    policy_payload = _load_json(run_dir / "policy.json")
    config_payload = policy_payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError(f"policy.json missing config object: {run_dir}")
    return PolicyConfig(**config_payload)


def _load_run_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(run_dir / "test_predictions.parquet")
    trade_records_path = run_dir / "test_trade_records.parquet"
    trade_records = pd.read_parquet(trade_records_path) if trade_records_path.exists() else pd.DataFrame(columns=["trade_id"])
    return predictions, trade_records


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def run_analysis(
    *,
    left_run_dir: Path,
    right_run_dir: Path,
    left_label: str,
    right_label: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    near_threshold_band_cents: float,
) -> dict[str, Any]:
    left_predictions, left_trade_records = _load_run_tables(left_run_dir)
    right_predictions, right_trade_records = _load_run_tables(right_run_dir)
    left_policy = _load_policy(left_run_dir)
    right_policy = _load_policy(right_run_dir)

    common = build_common_prediction_frame(
        left_predictions,
        right_predictions,
        left_label=left_label,
        right_label=right_label,
        left_trade_records=left_trade_records,
        right_trade_records=right_trade_records,
    )
    common = annotate_policy_gate_status(
        common,
        probability_column=f"{left_label}_calibrated_probability",
        policy_config=left_policy,
        output_prefix=left_label,
        near_threshold_band_cents=near_threshold_band_cents,
    )
    common = annotate_policy_gate_status(
        common,
        probability_column=f"{right_label}_calibrated_probability",
        policy_config=right_policy,
        output_prefix=right_label,
        near_threshold_band_cents=near_threshold_band_cents,
    )

    payload: dict[str, Any] = {
        "left_run": {"label": left_label, "run_dir": str(left_run_dir), "policy_config": asdict(left_policy)},
        "right_run": {"label": right_label, "run_dir": str(right_run_dir), "policy_config": asdict(right_policy)},
        "row_counts": {
            "left_test_rows": int(len(left_predictions)),
            "right_test_rows": int(len(right_predictions)),
            "common_test_rows": int(len(common)),
            "left_only_rows": int(len(left_predictions) - len(common)),
            "right_only_rows": int(len(right_predictions) - len(common)),
        },
        "slice_log_loss_matrix": compute_slice_log_loss_matrix(common, left_label=left_label, right_label=right_label),
        "hour_of_day_delta_table": summarize_bucket_deltas(
            common,
            group_columns=["hour_of_day_et", "hour_of_day_et_label"],
            left_label=left_label,
            right_label=right_label,
        ),
        "session_block_delta_table": summarize_bucket_deltas(
            common,
            group_columns=["session_block_et"],
            left_label=left_label,
            right_label=right_label,
        ),
        "realized_vol_regime_delta_table": summarize_bucket_deltas(
            common,
            group_columns=["realized_vol_regime_source", "realized_vol_regime_bucket"],
            left_label=left_label,
            right_label=right_label,
        ),
        "threshold_crossings": {
            "hour_of_day": summarize_threshold_crossings(
                common,
                group_columns=["hour_of_day_et", "hour_of_day_et_label"],
                left_label=left_label,
                right_label=right_label,
            ),
            "session_block": summarize_threshold_crossings(
                common,
                group_columns=["session_block_et"],
                left_label=left_label,
                right_label=right_label,
            ),
            "realized_vol_regime": summarize_threshold_crossings(
                common,
                group_columns=["realized_vol_regime_source", "realized_vol_regime_bucket"],
                left_label=left_label,
                right_label=right_label,
            ),
        },
        "bootstrap_hourly_correlation": bootstrap_bucket_delta_correlation_by_group(
            common,
            bucket_column="hour_of_day_et_label",
            left_label=left_label,
            right_label=right_label,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
        "bootstrap_session_correlation": bootstrap_bucket_delta_correlation_by_group(
            common,
            bucket_column="session_block_et",
            left_label=left_label,
            right_label=right_label,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
        "bootstrap_realized_vol_correlation": bootstrap_bucket_delta_correlation_by_group(
            common,
            bucket_column="realized_vol_regime_bucket",
            left_label=left_label,
            right_label=right_label,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze forecast-to-trade mapping between two saved runs.")
    parser.add_argument("--left-run-dir", required=True)
    parser.add_argument("--right-run-dir", required=True)
    parser.add_argument("--left-label", default="default")
    parser.add_argument("--right-label", default="spot_v1")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--near-threshold-band-cents", type=float, default=2.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    left_run_dir = Path(args.left_run_dir).expanduser()
    right_run_dir = Path(args.right_run_dir).expanduser()
    output = (
        Path(args.output).expanduser()
        if args.output
        else left_run_dir.parent / f"{args.left_label}_vs_{args.right_label}_forecast_trade_mapping.json"
    )

    payload = run_analysis(
        left_run_dir=left_run_dir,
        right_run_dir=right_run_dir,
        left_label=args.left_label,
        right_label=args.right_label,
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.bootstrap_seed),
        near_threshold_band_cents=float(args.near_threshold_band_cents),
    )
    output.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
