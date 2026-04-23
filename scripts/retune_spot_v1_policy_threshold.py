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
from src.live.kalshi.policy_mapping_analysis import annotate_policy_gate_status, build_common_prediction_frame
from src.live.kalshi.policy_threshold_retune import (
    build_threshold_map,
    evaluate_standard_policy_with_threshold_overrides,
    search_regime_threshold_offsets,
    split_time_ordered_frame,
    summarize_probability_distribution,
    summarize_tail_calibration,
    summarize_trade_pnl_distribution,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_policy(run_dir: Path) -> PolicyConfig:
    payload = _load_json(run_dir / "policy.json")
    config_payload = payload.get("config")
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


def _parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _log_loss_rows(frame: pd.DataFrame, probability_column: str) -> float | None:
    if frame.empty:
        return None
    from sklearn.metrics import log_loss

    return float(log_loss(frame["actual_outcome"], frame[probability_column], labels=[0, 1]))


def _distribution_shape_rows(frame: pd.DataFrame, *, left_label: str, right_label: str) -> list[dict[str, Any]]:
    rows = summarize_probability_distribution(
        frame,
        group_columns=["realized_vol_regime_bucket"],
        probability_columns={
            left_label: f"{left_label}_calibrated_probability",
            right_label: f"{right_label}_calibrated_probability",
        },
        crossing_columns={
            left_label: f"{left_label}_threshold_crossed",
            right_label: f"{right_label}_threshold_crossed",
        },
    )
    shaped: list[dict[str, Any]] = []
    for row in rows:
        shaped_row = dict(row)
        left_mean = float(row[f"{left_label}_probability_mean"])
        right_mean = float(row[f"{right_label}_probability_mean"])
        left_std = float(row[f"{left_label}_probability_std"])
        right_std = float(row[f"{right_label}_probability_std"])
        left_cross = float(row.get(f"{left_label}_crossing_rate", 0.0))
        right_cross = float(row.get(f"{right_label}_crossing_rate", 0.0))
        shaped_row["mean_shift"] = right_mean - left_mean
        shaped_row["std_ratio"] = (right_std / left_std) if left_std > 0.0 else None
        shaped_row["crossing_rate_delta"] = right_cross - left_cross
        shaped_row["crossing_rate_ratio"] = (right_cross / left_cross) if left_cross > 0.0 else None
        shaped.append(shaped_row)
    return shaped


def _marginal_trade_payload(frame: pd.DataFrame, *, regime_bucket: str, right_label: str, left_label: str) -> dict[str, Any]:
    regime_mask = frame["realized_vol_regime_bucket"].astype(str) == regime_bucket
    spot_only_mask = regime_mask & frame[f"{right_label}_selected"].astype(bool) & ~frame[f"{left_label}_selected"].astype(bool)
    shared_mask = regime_mask & frame[f"{right_label}_selected"].astype(bool) & frame[f"{left_label}_selected"].astype(bool)
    return {
        "regime_bucket": regime_bucket,
        "spot_only_trades": summarize_trade_pnl_distribution(
            frame.loc[spot_only_mask, [f"{right_label}_net_pnl_dollars"]].rename(
                columns={f"{right_label}_net_pnl_dollars": "net_pnl_dollars"}
            ),
        ),
        "shared_trades": summarize_trade_pnl_distribution(
            frame.loc[shared_mask, [f"{right_label}_net_pnl_dollars"]].rename(
                columns={f"{right_label}_net_pnl_dollars": "net_pnl_dollars"}
            ),
        ),
    }


def _regime_trade_pnl(trade_records: pd.DataFrame, regime_bucket: str) -> float:
    if trade_records.empty:
        return 0.0
    mask = trade_records["realized_vol_regime_bucket"].astype(str) == regime_bucket
    if not bool(mask.any()):
        return 0.0
    return float(trade_records.loc[mask, "net_pnl_dollars"].sum())


def _build_slice_log_loss_summary(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    left_label: str,
    right_label: str,
) -> dict[str, float | None]:
    return {
        f"{left_label}_selected_rows": _log_loss_rows(
            frame.loc[frame[f"{left_label}_selected"].astype(bool)],
            probability_column,
        ),
        f"{right_label}_selected_rows": _log_loss_rows(
            frame.loc[frame[f"{right_label}_selected"].astype(bool)],
            probability_column,
        ),
        f"{left_label}_near_threshold": _log_loss_rows(
            frame.loc[frame[f"{left_label}_near_threshold"].astype(bool)],
            probability_column,
        ),
        f"{right_label}_near_threshold": _log_loss_rows(
            frame.loc[frame[f"{right_label}_near_threshold"].astype(bool)],
            probability_column,
        ),
    }


def _serialize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(metrics)
    config = serialized.get("config")
    if isinstance(config, PolicyConfig):
        serialized["config"] = asdict(config)
    return serialized


def run_retune(
    *,
    left_run_dir: Path,
    right_run_dir: Path,
    left_label: str,
    right_label: str,
    fit_fraction: float,
    near_threshold_band_cents: float,
    global_threshold_grid: list[float],
    medium_offsets: list[float],
    high_offsets: list[float],
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

    fit_df, eval_df = split_time_ordered_frame(common, time_column="created_time", fit_fraction=fit_fraction)
    fit_spot_log_loss = _log_loss_rows(fit_df, f"{right_label}_calibrated_probability")
    eval_spot_log_loss = _log_loss_rows(eval_df, f"{right_label}_calibrated_probability")
    eval_default_log_loss = _log_loss_rows(eval_df, f"{left_label}_calibrated_probability")

    current_right_threshold_map = build_threshold_map(float(right_policy.edge_threshold_cents))
    current_left_threshold_map = build_threshold_map(float(left_policy.edge_threshold_cents))

    best_global = search_regime_threshold_offsets(
        fit_df,
        probability_column=f"{right_label}_calibrated_probability",
        config=right_policy,
        overall_log_loss=float(fit_spot_log_loss or 0.0),
        global_threshold_grid=global_threshold_grid,
        medium_offsets=[0.0],
        high_offsets=[0.0],
    )
    best_regime = search_regime_threshold_offsets(
        fit_df,
        probability_column=f"{right_label}_calibrated_probability",
        config=right_policy,
        overall_log_loss=float(fit_spot_log_loss or 0.0),
        global_threshold_grid=global_threshold_grid,
        medium_offsets=medium_offsets,
        high_offsets=high_offsets,
    )

    chosen_search = "regime_threshold_search"
    chosen_candidate = dict(best_regime["best_candidate"])
    if float(best_global["best_candidate"]["fit_score"]) > float(chosen_candidate["fit_score"]):
        chosen_search = "global_threshold_search"
        chosen_candidate = dict(best_global["best_candidate"])

    eval_default_metrics, eval_default_trades = evaluate_standard_policy_with_threshold_overrides(
        eval_df,
        probability_column=f"{left_label}_calibrated_probability",
        config=left_policy,
        overall_log_loss=float(eval_default_log_loss or 0.0),
        threshold_cents_by_bucket=current_left_threshold_map,
    )
    fit_default_metrics, _fit_default_trades = evaluate_standard_policy_with_threshold_overrides(
        fit_df,
        probability_column=f"{left_label}_calibrated_probability",
        config=left_policy,
        overall_log_loss=float(_log_loss_rows(fit_df, f"{left_label}_calibrated_probability") or 0.0),
        threshold_cents_by_bucket=current_left_threshold_map,
    )
    fit_spot_untuned_metrics, _fit_spot_untuned_trades = evaluate_standard_policy_with_threshold_overrides(
        fit_df,
        probability_column=f"{right_label}_calibrated_probability",
        config=right_policy,
        overall_log_loss=float(fit_spot_log_loss or 0.0),
        threshold_cents_by_bucket=current_right_threshold_map,
    )
    fit_global_metrics, _fit_global_trades = evaluate_standard_policy_with_threshold_overrides(
        fit_df,
        probability_column=f"{right_label}_calibrated_probability",
        config=right_policy,
        overall_log_loss=float(fit_spot_log_loss or 0.0),
        threshold_cents_by_bucket=best_global["best_candidate"]["threshold_cents_by_bucket"],
    )
    fit_regime_metrics, _fit_regime_trades = evaluate_standard_policy_with_threshold_overrides(
        fit_df,
        probability_column=f"{right_label}_calibrated_probability",
        config=right_policy,
        overall_log_loss=float(fit_spot_log_loss or 0.0),
        threshold_cents_by_bucket=best_regime["best_candidate"]["threshold_cents_by_bucket"],
    )
    eval_spot_untuned_metrics, eval_spot_untuned_trades = evaluate_standard_policy_with_threshold_overrides(
        eval_df,
        probability_column=f"{right_label}_calibrated_probability",
        config=right_policy,
        overall_log_loss=float(eval_spot_log_loss or 0.0),
        threshold_cents_by_bucket=current_right_threshold_map,
    )
    eval_global_metrics, eval_global_trades = evaluate_standard_policy_with_threshold_overrides(
        eval_df,
        probability_column=f"{right_label}_calibrated_probability",
        config=right_policy,
        overall_log_loss=float(eval_spot_log_loss or 0.0),
        threshold_cents_by_bucket=best_global["best_candidate"]["threshold_cents_by_bucket"],
    )
    eval_regime_metrics, eval_regime_trades = evaluate_standard_policy_with_threshold_overrides(
        eval_df,
        probability_column=f"{right_label}_calibrated_probability",
        config=right_policy,
        overall_log_loss=float(eval_spot_log_loss or 0.0),
        threshold_cents_by_bucket=best_regime["best_candidate"]["threshold_cents_by_bucket"],
    )
    eval_chosen_metrics, eval_chosen_trades = (
        (eval_global_metrics, eval_global_trades)
        if chosen_search == "global_threshold_search"
        else (eval_regime_metrics, eval_regime_trades)
    )

    eval_spot_candidate_frame = annotate_policy_gate_status(
        eval_df,
        probability_column=f"{right_label}_calibrated_probability",
        policy_config=PolicyConfig(
            edge_threshold_cents=float(chosen_candidate["threshold_cents_by_bucket"]["low"]),
            min_tau_minutes=right_policy.min_tau_minutes,
            max_tau_minutes=right_policy.max_tau_minutes,
            price_band_min_cents=right_policy.price_band_min_cents,
            price_band_max_cents=right_policy.price_band_max_cents,
            reserve_cash_pct=right_policy.reserve_cash_pct,
            maintain_edge_cents=right_policy.maintain_edge_cents,
            apply_regime_hard_gate=right_policy.apply_regime_hard_gate,
            starting_cash_dollars=right_policy.starting_cash_dollars,
            contracts_per_order=right_policy.contracts_per_order,
            capital_pct_per_order=right_policy.capital_pct_per_order,
            kelly_fraction_multiplier=right_policy.kelly_fraction_multiplier,
            kelly_fraction_cap_pct=right_policy.kelly_fraction_cap_pct,
            slippage_pct=right_policy.slippage_pct,
            allow_stacking=right_policy.allow_stacking,
            max_entries_per_ticker=right_policy.max_entries_per_ticker,
            require_price_improvement_for_stack=right_policy.require_price_improvement_for_stack,
        ),
        output_prefix="retuned_spot_v1",
        near_threshold_band_cents=near_threshold_band_cents,
    )
    for bucket, threshold in chosen_candidate["threshold_cents_by_bucket"].items():
        bucket_mask = eval_spot_candidate_frame["realized_vol_regime_bucket"].astype(str) == str(bucket)
        edge = eval_spot_candidate_frame.loc[bucket_mask, "retuned_spot_v1_post_cost_edge_cents"]
        eval_spot_candidate_frame.loc[bucket_mask, "retuned_spot_v1_threshold_crossed"] = edge >= float(threshold)
        eval_spot_candidate_frame.loc[bucket_mask, "retuned_spot_v1_pre_path_trade_eligible"] = (
            eval_spot_candidate_frame.loc[bucket_mask, "retuned_spot_v1_in_tau_band"].astype(bool)
            & eval_spot_candidate_frame.loc[bucket_mask, "retuned_spot_v1_in_price_band"].astype(bool)
            & eval_spot_candidate_frame.loc[bucket_mask, "retuned_spot_v1_passes_regime_gate"].astype(bool)
            & eval_spot_candidate_frame.loc[bucket_mask, "retuned_spot_v1_threshold_crossed"].astype(bool)
        )
        eval_spot_candidate_frame.loc[bucket_mask, "retuned_spot_v1_near_threshold"] = (
            edge - float(threshold)
        ).abs() <= near_threshold_band_cents
    candidate_selected_ids = set(eval_chosen_trades["trade_id"].astype(str)) if not eval_chosen_trades.empty else set()
    eval_spot_candidate_frame["retuned_spot_v1_selected"] = eval_spot_candidate_frame["trade_id"].astype(str).isin(candidate_selected_ids)

    untuned_slice_log_loss = _build_slice_log_loss_summary(
        eval_df,
        probability_column=f"{right_label}_calibrated_probability",
        left_label=left_label,
        right_label=right_label,
    )
    retuned_slice_log_loss = {
        f"{left_label}_selected_rows": untuned_slice_log_loss[f"{left_label}_selected_rows"],
        f"{right_label}_selected_rows": _log_loss_rows(
            eval_spot_candidate_frame.loc[eval_spot_candidate_frame["retuned_spot_v1_selected"].astype(bool)],
            f"{right_label}_calibrated_probability",
        ),
        f"{left_label}_near_threshold": untuned_slice_log_loss[f"{left_label}_near_threshold"],
        f"{right_label}_near_threshold": _log_loss_rows(
            eval_spot_candidate_frame.loc[eval_spot_candidate_frame["retuned_spot_v1_near_threshold"].astype(bool)],
            f"{right_label}_calibrated_probability",
        ),
    }

    spot_tail = summarize_tail_calibration(
        eval_df,
        probability_column=f"{right_label}_calibrated_probability",
    )
    default_tail = summarize_tail_calibration(
        eval_df,
        probability_column=f"{left_label}_calibrated_probability",
    )

    candidate_medium_pnl = _regime_trade_pnl(eval_chosen_trades, "medium")
    candidate_high_pnl = _regime_trade_pnl(eval_chosen_trades, "high")
    untuned_medium_pnl = _regime_trade_pnl(eval_spot_untuned_trades, "medium")
    untuned_high_pnl = _regime_trade_pnl(eval_spot_untuned_trades, "high")
    slice_log_loss_deltas = {
        key: (
            None
            if untuned_slice_log_loss[key] is None or retuned_slice_log_loss[key] is None
            else float(retuned_slice_log_loss[key] - untuned_slice_log_loss[key])
        )
        for key in untuned_slice_log_loss
    }
    low_tail_delta = float(
        (spot_tail["low_tail"]["absolute_calibration_error"] or 0.0)
        - (spot_tail["low_tail"]["absolute_calibration_error"] or 0.0)
    )
    high_tail_delta = float(
        (spot_tail["high_tail"]["absolute_calibration_error"] or 0.0)
        - (spot_tail["high_tail"]["absolute_calibration_error"] or 0.0)
    )

    criteria = {
        "better_than_untuned_spot_v1": float(eval_chosen_metrics["net_pnl_dollars"]) > float(
            eval_spot_untuned_metrics["net_pnl_dollars"]
        ),
        "better_than_default": float(eval_chosen_metrics["net_pnl_dollars"]) > float(eval_default_metrics["net_pnl_dollars"]),
        "conditional_log_loss_non_regression": all(
            delta is None or delta <= 0.005 for delta in slice_log_loss_deltas.values()
        ),
        "medium_vol_pnl_improves_by_at_least_one_dollar": (candidate_medium_pnl - untuned_medium_pnl) >= 1.0,
        "high_vol_pnl_not_worse_by_more_than_one_dollar": (candidate_high_pnl - untuned_high_pnl) >= -1.0,
        "tail_calibration_non_regression": low_tail_delta <= 0.02 and high_tail_delta <= 0.02,
    }
    criteria["carry_forward"] = bool(all(criteria.values()))

    return {
        "criteria_path": str(REPO_ROOT / "docs" / "SPOT_V1_POLICY_RETUNE_CRITERIA.md"),
        "left_run": {"label": left_label, "run_dir": str(left_run_dir), "policy_config": asdict(left_policy)},
        "right_run": {"label": right_label, "run_dir": str(right_run_dir), "policy_config": asdict(right_policy)},
        "common_row_counts": {
            "rows": int(len(common)),
            "fit_rows": int(len(fit_df)),
            "eval_rows": int(len(eval_df)),
        },
        "fit_eval_window": {
            "fit_start": fit_df["created_time"].iloc[0],
            "fit_end": fit_df["created_time"].iloc[-1],
            "eval_start": eval_df["created_time"].iloc[0],
            "eval_end": eval_df["created_time"].iloc[-1],
        },
        "distribution_sanity_check": {
            "fit_regime_distribution_rows": _distribution_shape_rows(fit_df, left_label=left_label, right_label=right_label),
            "fit_marginal_trade_pnl": [
                _marginal_trade_payload(fit_df, regime_bucket="medium", right_label=right_label, left_label=left_label),
                _marginal_trade_payload(fit_df, regime_bucket="high", right_label=right_label, left_label=left_label),
            ],
        },
        "search_space": {
            "fit_fraction": float(fit_fraction),
            "near_threshold_band_cents": float(near_threshold_band_cents),
            "global_threshold_grid": global_threshold_grid,
            "medium_offsets": medium_offsets,
            "high_offsets": high_offsets,
        },
        "fit_search_results": {
            "untuned_threshold_map": current_right_threshold_map,
            "fit_baselines": {
                "default_baseline": _serialize_metrics(fit_default_metrics),
                "spot_v1_untuned": _serialize_metrics(fit_spot_untuned_metrics),
                "spot_v1_best_global": _serialize_metrics(fit_global_metrics),
                "spot_v1_best_regime": _serialize_metrics(fit_regime_metrics),
            },
            "best_global_candidate": {
                "search_type": "global_threshold_search",
                **_serialize_metrics(best_global["best_candidate"]),
            },
            "best_regime_candidate": {
                "search_type": "regime_threshold_search",
                **_serialize_metrics(best_regime["best_candidate"]),
            },
            "chosen_candidate_search": chosen_search,
            "chosen_candidate": {
                "search_type": chosen_search,
                **_serialize_metrics(chosen_candidate),
            },
        },
        "eval_results": {
            "default_baseline": _serialize_metrics(eval_default_metrics),
            "spot_v1_untuned": _serialize_metrics(eval_spot_untuned_metrics),
            "spot_v1_best_global": _serialize_metrics(eval_global_metrics),
            "spot_v1_best_regime": _serialize_metrics(eval_regime_metrics),
            "chosen_candidate_search": chosen_search,
            "spot_v1_chosen_candidate": _serialize_metrics(eval_chosen_metrics),
            "eval_regime_pnl": {
                "untuned": {"medium": untuned_medium_pnl, "high": untuned_high_pnl},
                "chosen_candidate": {"medium": candidate_medium_pnl, "high": candidate_high_pnl},
            },
        },
        "conditional_log_loss_confirmation": {
            "untuned_spot_v1_fixed_slices": untuned_slice_log_loss,
            "retuned_spot_v1_candidate_slices": retuned_slice_log_loss,
            "delta_retuned_minus_untuned": slice_log_loss_deltas,
        },
        "tail_calibration_confirmation": {
            "default_model": default_tail,
            "spot_v1_model": spot_tail,
            "retuned_spot_v1_model": spot_tail,
            "absolute_calibration_error_delta_retuned_minus_untuned": {
                "low_tail": low_tail_delta,
                "high_tail": high_tail_delta,
            },
        },
        "criteria_assessment": criteria,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retune spot_v1 threshold policy on a fit/eval split of the common-row test set.")
    parser.add_argument("--left-run-dir", required=True)
    parser.add_argument("--right-run-dir", required=True)
    parser.add_argument("--left-label", default="default")
    parser.add_argument("--right-label", default="spot_v1")
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--near-threshold-band-cents", type=float, default=2.0)
    parser.add_argument("--global-threshold-grid", default="2.5,3.0,3.5,4.0,4.5,5.0,5.5")
    parser.add_argument("--medium-offsets", default="0.0,0.5,1.0,1.5,2.0")
    parser.add_argument("--high-offsets", default="-0.5,0.0,0.5")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    left_run_dir = Path(args.left_run_dir).expanduser()
    right_run_dir = Path(args.right_run_dir).expanduser()
    output = (
        Path(args.output).expanduser()
        if args.output
        else right_run_dir.parent / f"{args.left_label}_vs_{args.right_label}_policy_retune_analysis.json"
    )
    payload = run_retune(
        left_run_dir=left_run_dir,
        right_run_dir=right_run_dir,
        left_label=args.left_label,
        right_label=args.right_label,
        fit_fraction=float(args.fit_fraction),
        near_threshold_band_cents=float(args.near_threshold_band_cents),
        global_threshold_grid=_parse_float_list(args.global_threshold_grid),
        medium_offsets=_parse_float_list(args.medium_offsets),
        high_offsets=_parse_float_list(args.high_offsets),
    )
    output.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
