from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.live.kalshi.offline_training import PolicyConfig
from src.live.kalshi.signal_risk import calculate_cost_metrics, find_max_acceptable_entry_price_cents

_ROW_METADATA_COLUMNS = (
    "ticker",
    "created_time",
    "close_time",
    "actual_outcome",
    "market_prob",
    "tau_minutes",
    "realized_vol_regime_source",
    "realized_vol_regime_bucket",
    "realized_vol_regime_bucket_version",
    "realized_vol_regime_metric",
    "hour_of_day_et",
    "hour_of_day_et_label",
    "session_block_et",
)


def _require_unique_trade_ids(frame: pd.DataFrame, *, name: str) -> None:
    duplicated = frame["trade_id"].duplicated(keep=False)
    if duplicated.any():
        duplicates = frame.loc[duplicated, "trade_id"].astype(str).unique().tolist()[:5]
        raise ValueError(f"{name} contains duplicate trade_id values: {duplicates}")


def _binary_log_loss_contributions(actual_outcome: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    actual = np.asarray(actual_outcome, dtype=np.float64)
    return -(actual * np.log(clipped) + (1.0 - actual) * np.log(1.0 - clipped))


def _reference_price_cents(side: str, market_prob: float) -> int:
    yes_price_cents = int(np.clip(np.rint(float(market_prob) * 100.0), 1, 99))
    return yes_price_cents if side == "YES" else 100 - yes_price_cents


def build_common_prediction_frame(
    left_predictions: pd.DataFrame,
    right_predictions: pd.DataFrame,
    *,
    left_label: str,
    right_label: str,
    left_trade_records: pd.DataFrame | None = None,
    right_trade_records: pd.DataFrame | None = None,
) -> pd.DataFrame:
    _require_unique_trade_ids(left_predictions, name=f"{left_label} predictions")
    _require_unique_trade_ids(right_predictions, name=f"{right_label} predictions")

    left_required = {"trade_id", "calibrated_probability", "actual_outcome", "market_prob", "tau_minutes"}
    right_required = {"trade_id", "calibrated_probability", "actual_outcome", "market_prob", "tau_minutes"}
    missing_left = left_required - set(left_predictions.columns)
    missing_right = right_required - set(right_predictions.columns)
    if missing_left:
        raise ValueError(f"{left_label} predictions missing required columns: {sorted(missing_left)}")
    if missing_right:
        raise ValueError(f"{right_label} predictions missing required columns: {sorted(missing_right)}")

    left = left_predictions.copy()
    right = right_predictions.copy()
    merged = left.merge(
        right,
        on="trade_id",
        how="inner",
        suffixes=(f"__{left_label}", f"__{right_label}"),
    )
    if merged.empty:
        raise ValueError("No common trade_id rows found between the two prediction frames.")

    frame = pd.DataFrame({"trade_id": merged["trade_id"].astype(str)})
    for column in _ROW_METADATA_COLUMNS:
        left_key = f"{column}__{left_label}"
        right_key = f"{column}__{right_label}"
        if left_key not in merged.columns and right_key not in merged.columns:
            continue
        if left_key in merged.columns and right_key in merged.columns:
            left_values = merged[left_key]
            right_values = merged[right_key]
            if pd.api.types.is_numeric_dtype(left_values) or pd.api.types.is_bool_dtype(left_values):
                mismatch = ~np.isclose(
                    left_values.to_numpy(dtype=np.float64, copy=False),
                    right_values.to_numpy(dtype=np.float64, copy=False),
                    equal_nan=True,
                )
                if bool(np.any(mismatch)):
                    raise ValueError(f"Row metadata column {column} differs between prediction frames.")
            else:
                if not left_values.fillna("<NA>").equals(right_values.fillna("<NA>")):
                    raise ValueError(f"Row metadata column {column} differs between prediction frames.")
            frame[column] = left_values.to_numpy(copy=True)
        elif left_key in merged.columns:
            frame[column] = merged[left_key].to_numpy(copy=True)
        else:
            frame[column] = merged[right_key].to_numpy(copy=True)

    for label in (left_label, right_label):
        calibrated_key = f"calibrated_probability__{label}"
        raw_key = f"raw_probability__{label}"
        frame[f"{label}_calibrated_probability"] = merged[calibrated_key].to_numpy(dtype=np.float64, copy=True)
        if raw_key in merged.columns:
            frame[f"{label}_raw_probability"] = merged[raw_key].to_numpy(dtype=np.float64, copy=True)

    actual_outcome = frame["actual_outcome"].to_numpy(dtype=np.int8, copy=False)
    for label in (left_label, right_label):
        probability_col = f"{label}_calibrated_probability"
        contributions = _binary_log_loss_contributions(actual_outcome, frame[probability_col].to_numpy(dtype=np.float64, copy=False))
        frame[f"{label}_logloss_contribution"] = contributions

    frame["logloss_delta_contribution"] = (
        frame[f"{left_label}_logloss_contribution"] - frame[f"{right_label}_logloss_contribution"]
    )

    for label, trade_records in ((left_label, left_trade_records), (right_label, right_trade_records)):
        selected_col = f"{label}_selected"
        pnl_col = f"{label}_net_pnl_dollars"
        if trade_records is None or trade_records.empty:
            frame[selected_col] = False
            frame[pnl_col] = 0.0
            continue
        _require_unique_trade_ids(trade_records, name=f"{label} trade records")
        trade_payload = trade_records.loc[:, ["trade_id", "net_pnl_dollars"]].copy()
        trade_payload["trade_id"] = trade_payload["trade_id"].astype(str)
        trade_payload = trade_payload.rename(columns={"net_pnl_dollars": pnl_col})
        frame = frame.merge(trade_payload, on="trade_id", how="left")
        frame[selected_col] = frame[pnl_col].notna()
        frame[pnl_col] = frame[pnl_col].fillna(0.0).astype(np.float64, copy=False)

    frame["pnl_delta_dollars"] = frame[f"{right_label}_net_pnl_dollars"] - frame[f"{left_label}_net_pnl_dollars"]
    sort_columns = [column for column in ("created_time", "ticker", "trade_id") if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns).reset_index(drop=True)
    return frame


def annotate_policy_gate_status(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    policy_config: PolicyConfig,
    output_prefix: str,
    near_threshold_band_cents: float = 2.0,
) -> pd.DataFrame:
    if probability_column not in frame.columns:
        raise ValueError(f"Missing probability column: {probability_column}")
    if "market_prob" not in frame.columns or "tau_minutes" not in frame.columns:
        raise ValueError("Frame must include market_prob and tau_minutes.")

    signal_config = policy_config.signal_config
    annotated = frame.copy()
    probabilities = annotated[probability_column].to_numpy(dtype=np.float64, copy=False)
    market_prob = annotated["market_prob"].to_numpy(dtype=np.float64, copy=False)
    tau_minutes = annotated["tau_minutes"].to_numpy(dtype=np.float64, copy=False)

    side_is_yes = probabilities > market_prob
    side = np.where(side_is_yes, "YES", "NO")
    reference_price_cents = np.where(
        side_is_yes,
        np.clip(np.rint(market_prob * 100.0), 1, 99),
        100 - np.clip(np.rint(market_prob * 100.0), 1, 99),
    ).astype(np.int16, copy=False)
    in_tau_band = (tau_minutes >= signal_config.min_tau_minutes) & (tau_minutes <= signal_config.max_tau_minutes)
    in_price_band = (reference_price_cents >= signal_config.price_band_min_cents) & (
        reference_price_cents <= signal_config.price_band_max_cents
    )

    regime_column = "regime_label"
    passes_regime_gate = np.ones(len(annotated), dtype=bool)
    if signal_config.apply_regime_hard_gate and regime_column in annotated.columns:
        regime_values = annotated[regime_column].astype(str).to_numpy(copy=False)
        passes_regime_gate = ~(np.logical_and(side_is_yes, regime_values == "downtrend"))

    max_acceptable_cents: list[int | None] = []
    post_cost_edge_cents: list[float] = []
    threshold_crossed: list[bool] = []
    near_threshold: list[bool] = []
    contracts = max(1, int(policy_config.contracts_per_order))

    for row_side, probability, price_cents, tau_ok, price_ok, regime_ok in zip(
        side,
        probabilities,
        reference_price_cents,
        in_tau_band,
        in_price_band,
        passes_regime_gate,
        strict=False,
    ):
        if not (tau_ok and price_ok and regime_ok):
            max_acceptable_cents.append(None)
            post_cost_edge_cents.append(np.nan)
            threshold_crossed.append(False)
            near_threshold.append(False)
            continue
        max_acceptable = find_max_acceptable_entry_price_cents(
            side=str(row_side),
            predicted_yes_probability=float(probability),
            config=signal_config,
            contracts=contracts,
        )
        max_acceptable_cents.append(max_acceptable)
        post_cost_edge, _entry_cost, _fees, _cash_required = calculate_cost_metrics(
            side=str(row_side),
            predicted_yes_probability=float(probability),
            displayed_entry_price_cents=int(price_cents),
            contracts=contracts,
            slippage=signal_config.slippage,
        )
        post_cost_edge_cents.append(post_cost_edge * 100.0)
        threshold_crossed.append(bool(max_acceptable is not None and int(price_cents) <= max_acceptable and post_cost_edge + 1e-12 >= signal_config.edge_threshold))
        near_threshold.append(abs(post_cost_edge * 100.0 - float(policy_config.edge_threshold_cents)) <= near_threshold_band_cents)

    annotated[f"{output_prefix}_side"] = side
    annotated[f"{output_prefix}_reference_price_cents"] = reference_price_cents
    annotated[f"{output_prefix}_in_tau_band"] = in_tau_band
    annotated[f"{output_prefix}_in_price_band"] = in_price_band
    annotated[f"{output_prefix}_passes_regime_gate"] = passes_regime_gate
    annotated[f"{output_prefix}_max_acceptable_entry_price_cents"] = max_acceptable_cents
    annotated[f"{output_prefix}_post_cost_edge_cents"] = post_cost_edge_cents
    annotated[f"{output_prefix}_threshold_crossed"] = threshold_crossed
    annotated[f"{output_prefix}_pre_path_trade_eligible"] = (
        annotated[f"{output_prefix}_in_tau_band"]
        & annotated[f"{output_prefix}_in_price_band"]
        & annotated[f"{output_prefix}_passes_regime_gate"]
        & annotated[f"{output_prefix}_threshold_crossed"]
    )
    annotated[f"{output_prefix}_near_threshold"] = near_threshold
    return annotated


def _slice_log_loss(actual_outcome: pd.Series, probabilities: pd.Series) -> float | None:
    if actual_outcome.empty:
        return None
    return float(log_loss(actual_outcome, probabilities, labels=[0, 1]))


def compute_slice_log_loss_matrix(frame: pd.DataFrame, *, left_label: str, right_label: str) -> list[dict[str, object]]:
    actual_outcome = frame["actual_outcome"]
    slices = {
        "all_rows": pd.Series(True, index=frame.index),
        f"{left_label}_selected_rows": frame[f"{left_label}_selected"].astype(bool),
        f"{right_label}_selected_rows": frame[f"{right_label}_selected"].astype(bool),
        f"{left_label}_pre_path_trade_eligible": frame[f"{left_label}_pre_path_trade_eligible"].astype(bool),
        f"{right_label}_pre_path_trade_eligible": frame[f"{right_label}_pre_path_trade_eligible"].astype(bool),
        f"{left_label}_near_threshold": frame[f"{left_label}_near_threshold"].astype(bool),
        f"{right_label}_near_threshold": frame[f"{right_label}_near_threshold"].astype(bool),
    }
    rows: list[dict[str, object]] = []
    for slice_name, mask in slices.items():
        subset = frame.loc[mask]
        left_log_loss = _slice_log_loss(actual_outcome.loc[mask], subset[f"{left_label}_calibrated_probability"])
        right_log_loss = _slice_log_loss(actual_outcome.loc[mask], subset[f"{right_label}_calibrated_probability"])
        rows.append(
            {
                "slice_name": slice_name,
                "rows": int(mask.sum()),
                "left_label": left_label,
                "right_label": right_label,
                "left_log_loss": left_log_loss,
                "right_log_loss": right_log_loss,
                "left_minus_right_log_loss": (
                    (left_log_loss - right_log_loss) if left_log_loss is not None and right_log_loss is not None else None
                ),
            }
        )
    return rows


def summarize_threshold_crossings(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    left_label: str,
    right_label: str,
) -> list[dict[str, object]]:
    if not group_columns:
        raise ValueError("group_columns must be non-empty.")
    summaries: list[dict[str, object]] = []
    grouped = frame.groupby(list(group_columns), dropna=False, sort=False, observed=False)
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        payload: dict[str, object] = {
            column: ("unknown" if pd.isna(value) else value)
            for column, value in zip(group_columns, key_tuple, strict=False)
        }
        payload["rows"] = int(len(group))
        for label in (left_label, right_label):
            payload[f"{label}_threshold_crossings"] = int(group[f"{label}_threshold_crossed"].sum())
            payload[f"{label}_pre_path_trade_eligible_rows"] = int(group[f"{label}_pre_path_trade_eligible"].sum())
            selected_column = f"{label}_selected"
            payload[f"{label}_selected_rows"] = int(group[selected_column].sum()) if selected_column in group.columns else 0
        payload["crossing_delta"] = int(payload[f"{right_label}_threshold_crossings"]) - int(
            payload[f"{left_label}_threshold_crossings"]
        )
        payload["selected_delta"] = int(payload[f"{right_label}_selected_rows"]) - int(
            payload[f"{left_label}_selected_rows"]
        )
        summaries.append(payload)
    return summaries


def summarize_bucket_deltas(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    left_label: str,
    right_label: str,
) -> list[dict[str, object]]:
    if not group_columns:
        raise ValueError("group_columns must be non-empty.")
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(list(group_columns), dropna=False, sort=False, observed=False)
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        payload: dict[str, object] = {
            column: ("unknown" if pd.isna(value) else value)
            for column, value in zip(group_columns, key_tuple, strict=False)
        }
        payload["rows"] = int(len(group))
        payload["left_log_loss"] = _slice_log_loss(group["actual_outcome"], group[f"{left_label}_calibrated_probability"])
        payload["right_log_loss"] = _slice_log_loss(group["actual_outcome"], group[f"{right_label}_calibrated_probability"])
        payload["logloss_delta"] = float(group["logloss_delta_contribution"].mean())
        payload[f"{left_label}_net_pnl_dollars"] = float(group[f"{left_label}_net_pnl_dollars"].sum())
        payload[f"{right_label}_net_pnl_dollars"] = float(group[f"{right_label}_net_pnl_dollars"].sum())
        payload["pnl_delta_dollars"] = float(group["pnl_delta_dollars"].sum())
        payload[f"{left_label}_selected_rows"] = int(group[f"{left_label}_selected"].sum())
        payload[f"{right_label}_selected_rows"] = int(group[f"{right_label}_selected"].sum())
        rows.append(payload)
    return rows


def _safe_pearson(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) < 2 or len(values_y) < 2:
        return None
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_bucket_delta_correlation(
    frame: pd.DataFrame,
    *,
    bucket_column: str,
    logloss_delta_column: str,
    pnl_delta_column: str,
    iterations: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    if bucket_column not in frame.columns:
        raise ValueError(f"Missing bucket column: {bucket_column}")
    bucket_arrays: dict[object, np.ndarray] = {}
    for bucket_value, group in frame.groupby(bucket_column, dropna=False, sort=False, observed=False):
        bucket_arrays[bucket_value] = group.loc[:, [logloss_delta_column, pnl_delta_column]].to_numpy(dtype=np.float64, copy=True)
    point_logloss = [float(group[:, 0].mean()) for group in bucket_arrays.values()]
    point_pnl = [float(group[:, 1].sum()) for group in bucket_arrays.values()]
    point_estimate = _safe_pearson(point_logloss, point_pnl)

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled_logloss: list[float] = []
        sampled_pnl: list[float] = []
        for values in bucket_arrays.values():
            n_rows = len(values)
            if n_rows == 0:
                continue
            sampled_indices = rng.integers(0, n_rows, size=n_rows)
            sampled = values[sampled_indices]
            sampled_logloss.append(float(sampled[:, 0].mean()))
            sampled_pnl.append(float(sampled[:, 1].sum()))
        sample_corr = _safe_pearson(sampled_logloss, sampled_pnl)
        if sample_corr is not None:
            samples.append(sample_corr)
    if not samples:
        bootstrap_p05 = None
        bootstrap_p95 = None
        bootstrap_median = None
    else:
        bootstrap_p05, bootstrap_median, bootstrap_p95 = (
            float(np.percentile(samples, 5)),
            float(np.percentile(samples, 50)),
            float(np.percentile(samples, 95)),
        )
    return {
        "bucket_column": bucket_column,
        "bucket_count": len(bucket_arrays),
        "iterations": iterations,
        "seed": seed,
        "point_estimate": point_estimate,
        "bootstrap_p05": bootstrap_p05,
        "bootstrap_p50": bootstrap_median,
        "bootstrap_p95": bootstrap_p95,
    }


def bootstrap_bucket_delta_correlation_by_group(
    frame: pd.DataFrame,
    *,
    bucket_column: str,
    left_label: str,
    right_label: str,
    iterations: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    working = frame.copy()
    working["logloss_delta_contribution"] = (
        working[f"{left_label}_logloss_contribution"] - working[f"{right_label}_logloss_contribution"]
    )
    working["pnl_delta_dollars"] = (
        working[f"{right_label}_net_pnl_dollars"] - working[f"{left_label}_net_pnl_dollars"]
    )
    return bootstrap_bucket_delta_correlation(
        working,
        bucket_column=bucket_column,
        logloss_delta_column="logloss_delta_contribution",
        pnl_delta_column="pnl_delta_dollars",
        iterations=iterations,
        seed=seed,
    )


__all__ = [
    "annotate_policy_gate_status",
    "bootstrap_bucket_delta_correlation",
    "bootstrap_bucket_delta_correlation_by_group",
    "build_common_prediction_frame",
    "compute_slice_log_loss_matrix",
    "summarize_bucket_deltas",
    "summarize_threshold_crossings",
]
