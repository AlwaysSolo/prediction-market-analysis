from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.live.kalshi.offline_training import (
    PolicyConfig,
    _is_standard_policy_fast_path_compatible,
    _prepare_standard_policy_inputs,
)


@dataclass(frozen=True)
class _OpenPosition:
    ticker_code: int
    entry_cost: float
    payout: float
    fees: float


def split_time_ordered_frame(
    frame: pd.DataFrame,
    *,
    time_column: str = "created_time",
    fit_fraction: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < fit_fraction < 1.0):
        raise ValueError("fit_fraction must be strictly between 0 and 1.")
    if time_column not in frame.columns:
        raise ValueError(f"Missing time column: {time_column}")
    ordered = frame.copy()
    ordered[time_column] = pd.to_datetime(ordered[time_column], utc=True)
    sort_columns = [time_column] + [column for column in ("ticker", "trade_id") if column in ordered.columns]
    ordered = ordered.sort_values(sort_columns).reset_index(drop=True)
    split_index = int(np.floor(len(ordered) * fit_fraction))
    split_index = min(max(split_index, 1), len(ordered) - 1)
    return ordered.iloc[:split_index].reset_index(drop=True), ordered.iloc[split_index:].reset_index(drop=True)


def summarize_probability_distribution(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    probability_columns: dict[str, str],
    crossing_columns: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(group_columns, dropna=False, sort=False, observed=False)
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        payload: dict[str, object] = {
            column: ("unknown" if pd.isna(value) else value)
            for column, value in zip(group_columns, key_tuple, strict=False)
        }
        payload["rows"] = int(len(group))
        for label, column in probability_columns.items():
            values = group[column].to_numpy(dtype=np.float64, copy=False)
            payload[f"{label}_probability_mean"] = float(np.mean(values))
            payload[f"{label}_probability_std"] = float(np.std(values))
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
                payload[f"{label}_probability_q{int(quantile * 100):02d}"] = float(np.quantile(values, quantile))
            if crossing_columns is not None and label in crossing_columns:
                crossing_values = group[crossing_columns[label]].astype(bool)
                payload[f"{label}_crossing_rate"] = float(crossing_values.mean()) if len(crossing_values) else 0.0
        rows.append(payload)
    return rows


def build_threshold_map(
    global_threshold: float,
    *,
    medium_offset: float = 0.0,
    high_offset: float = 0.0,
) -> dict[str, float]:
    return {
        "low": float(global_threshold),
        "medium": float(global_threshold + medium_offset),
        "high": float(global_threshold + high_offset),
        "extreme": float(global_threshold + high_offset),
        "unknown": float(global_threshold),
    }


def summarize_tail_calibration(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    actual_column: str = "actual_outcome",
    market_probability_column: str = "market_prob",
    low_cutoff: float = 0.10,
    high_cutoff: float = 0.90,
) -> dict[str, dict[str, float | int | None]]:
    def _tail_payload(mask: pd.Series) -> dict[str, float | int | None]:
        subset = frame.loc[mask]
        if subset.empty:
            return {
                "rows": 0,
                "row_pct": 0.0,
                "mean_predicted_yes_probability": None,
                "empirical_yes_rate": None,
                "mean_market_probability": None,
                "calibration_error": None,
                "absolute_calibration_error": None,
            }
        mean_probability = float(subset[probability_column].mean())
        empirical_yes_rate = float(subset[actual_column].mean())
        return {
            "rows": int(len(subset)),
            "row_pct": float(len(subset) / len(frame) * 100.0) if len(frame) else 0.0,
            "mean_predicted_yes_probability": mean_probability,
            "empirical_yes_rate": empirical_yes_rate,
            "mean_market_probability": float(subset[market_probability_column].mean()),
            "calibration_error": mean_probability - empirical_yes_rate,
            "absolute_calibration_error": abs(mean_probability - empirical_yes_rate),
        }

    probabilities = frame[probability_column].astype(float)
    return {
        "low_tail": _tail_payload(probabilities < low_cutoff),
        "high_tail": _tail_payload(probabilities > high_cutoff),
        "cutoffs": {"low": float(low_cutoff), "high": float(high_cutoff)},
    }


def summarize_trade_pnl_distribution(
    frame: pd.DataFrame,
    *,
    pnl_column: str = "net_pnl_dollars",
) -> dict[str, float | int | None]:
    if frame.empty:
        return {
            "rows": 0,
            "wins": 0,
            "win_rate": None,
            "net_pnl_dollars": 0.0,
            "mean_net_pnl_dollars": None,
            "median_net_pnl_dollars": None,
            "p05_net_pnl_dollars": None,
            "p25_net_pnl_dollars": None,
            "p75_net_pnl_dollars": None,
            "p95_net_pnl_dollars": None,
        }
    pnl = frame[pnl_column].astype(float)
    return {
        "rows": int(len(frame)),
        "wins": int((pnl > 0.0).sum()),
        "win_rate": float((pnl > 0.0).mean()),
        "net_pnl_dollars": float(pnl.sum()),
        "mean_net_pnl_dollars": float(pnl.mean()),
        "median_net_pnl_dollars": float(pnl.median()),
        "p05_net_pnl_dollars": float(pnl.quantile(0.05)),
        "p25_net_pnl_dollars": float(pnl.quantile(0.25)),
        "p75_net_pnl_dollars": float(pnl.quantile(0.75)),
        "p95_net_pnl_dollars": float(pnl.quantile(0.95)),
    }


def _sorted_frame_and_prepared(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    config: PolicyConfig,
) -> tuple[pd.DataFrame, Any]:
    if not _is_standard_policy_fast_path_compatible(config):
        raise ValueError("Threshold retune helpers currently support only standard fast-path policy configs.")
    working = frame.loc[
        :,
        [
            "ticker",
            "trade_id",
            "created_time",
            "close_time",
            "tau_minutes",
            "market_prob",
            "actual_outcome",
        ],
    ].copy()
    probabilities = frame[probability_column].to_numpy(dtype=np.float64, copy=True)
    prepared = _prepare_standard_policy_inputs(working, probabilities, slippage=config.signal_config.slippage)
    sorted_frame = frame.sort_values(["created_time", "ticker", "trade_id"]).reset_index(drop=True)
    if len(sorted_frame) != len(prepared.created_time_ns):
        raise RuntimeError("Prepared inputs and sorted frame length mismatch.")
    return sorted_frame, prepared


def evaluate_standard_policy_with_threshold_overrides(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    config: PolicyConfig,
    overall_log_loss: float,
    threshold_cents_by_bucket: dict[str, float],
    bucket_column: str = "realized_vol_regime_bucket",
) -> tuple[dict[str, object], pd.DataFrame]:
    sorted_frame, prepared = _sorted_frame_and_prepared(frame, probability_column=probability_column, config=config)
    signal_config = config.signal_config

    available_cash = signal_config.starting_cash_dollars
    open_cost_basis = 0.0
    peak_equity = signal_config.starting_cash_dollars
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    total_pnl = 0.0
    trades = 0
    skipped_due_open_ticker = 0
    skipped_due_price_band = 0
    skipped_due_post_cost_edge = 0
    sequence = 0
    reserve_cash_ratio = signal_config.reserve_cash_pct / 100.0
    open_positions: list[tuple[int, int, _OpenPosition]] = []
    active_tickers: set[int] = set()
    trade_records: list[dict[str, object]] = []

    bucket_values = sorted_frame[bucket_column].astype(str).fillna("unknown").to_numpy(copy=False)

    def update_drawdown() -> None:
        nonlocal peak_equity, max_drawdown, max_drawdown_pct
        equity = available_cash + open_cost_basis
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        drawdown_pct = 0.0 if peak_equity <= 0 else drawdown / peak_equity
        max_drawdown = max(max_drawdown, drawdown)
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct * 100.0)

    def settle_positions(cutoff_time_ns: int) -> None:
        nonlocal available_cash, open_cost_basis, total_pnl
        while open_positions and open_positions[0][0] <= cutoff_time_ns:
            _close_time_ns, _seq, position = heapq.heappop(open_positions)
            open_cost_basis -= position.entry_cost
            available_cash += position.payout
            total_pnl += (position.payout - position.entry_cost) - position.fees
            active_tickers.discard(position.ticker_code)
            update_drawdown()

    for row_index in range(len(prepared.created_time_ns)):
        settle_positions(int(prepared.created_time_ns[row_index]))
        tau_minutes = float(prepared.tau_minutes[row_index])
        if tau_minutes < signal_config.min_tau_minutes or tau_minutes > signal_config.max_tau_minutes:
            continue

        reference_price_cents = int(prepared.reference_price_cents[row_index])
        if reference_price_cents < signal_config.price_band_min_cents or reference_price_cents > signal_config.price_band_max_cents:
            skipped_due_price_band += 1
            continue

        ticker_code = int(prepared.ticker_codes[row_index])
        if ticker_code in active_tickers:
            skipped_due_open_ticker += 1
            continue

        bucket_value = bucket_values[row_index]
        edge_threshold_cents = float(threshold_cents_by_bucket.get(bucket_value, config.edge_threshold_cents))
        edge_threshold = edge_threshold_cents / 100.0
        post_cost_edge = float(prepared.post_cost_edge[row_index])
        if post_cost_edge + 1e-12 < edge_threshold:
            skipped_due_post_cost_edge += 1
            continue

        cash_required = float(prepared.cash_required_dollars[row_index])
        reserve_cash = reserve_cash_ratio * (available_cash + open_cost_basis)
        if available_cash - cash_required < reserve_cash - 1e-12:
            continue

        entry_cost = float(prepared.entry_cost_dollars[row_index])
        payout_dollars = float(prepared.payout_dollars[row_index])
        fees_dollars = float(prepared.fees_dollars[row_index])

        available_cash -= cash_required
        open_cost_basis += entry_cost
        position = _OpenPosition(
            ticker_code=ticker_code,
            entry_cost=entry_cost,
            payout=payout_dollars,
            fees=fees_dollars,
        )
        heapq.heappush(open_positions, (int(prepared.close_time_ns[row_index]), sequence, position))
        active_tickers.add(ticker_code)
        trade_records.append(
            {
                "trade_id": str(sorted_frame.iloc[row_index]["trade_id"]),
                "ticker": str(sorted_frame.iloc[row_index]["ticker"]),
                "created_time": sorted_frame.iloc[row_index]["created_time"],
                "close_time": sorted_frame.iloc[row_index]["close_time"],
                "actual_outcome": int(sorted_frame.iloc[row_index]["actual_outcome"]),
                "market_prob": float(sorted_frame.iloc[row_index]["market_prob"]),
                "tau_minutes": float(sorted_frame.iloc[row_index]["tau_minutes"]),
                "predicted_yes_probability": float(sorted_frame.iloc[row_index][probability_column]),
                "realized_vol_regime_bucket": bucket_value,
                "session_block_et": sorted_frame.iloc[row_index].get("session_block_et", "unknown"),
                "hour_of_day_et_label": sorted_frame.iloc[row_index].get("hour_of_day_et_label", "unknown"),
                "reference_price_cents": reference_price_cents,
                "post_cost_edge_cents": post_cost_edge * 100.0,
                "net_pnl_dollars": (payout_dollars - entry_cost) - fees_dollars,
            }
        )
        sequence += 1
        trades += 1
        update_drawdown()

    settle_positions(prepared.max_close_time_ns)
    ending_equity = available_cash + open_cost_basis
    return_pct = 0.0 if signal_config.starting_cash_dollars <= 0 else (
        (ending_equity / signal_config.starting_cash_dollars) - 1.0
    ) * 100.0
    objective = total_pnl / max(1.0, max_drawdown)
    metrics = {
        "config": config,
        "threshold_cents_by_bucket": dict(threshold_cents_by_bucket),
        "objective": float(objective),
        "trades": int(trades),
        "net_pnl_dollars": float(total_pnl),
        "max_drawdown_dollars": float(max_drawdown),
        "max_drawdown_pct": float(max_drawdown_pct),
        "return_pct": float(return_pct),
        "log_loss": float(overall_log_loss),
        "skipped_due_open_ticker": int(skipped_due_open_ticker),
        "skipped_due_price_band": int(skipped_due_price_band),
        "skipped_due_post_cost_edge": int(skipped_due_post_cost_edge),
    }
    return metrics, pd.DataFrame(trade_records)


def search_regime_threshold_offsets(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    config: PolicyConfig,
    overall_log_loss: float,
    global_threshold_grid: list[float],
    medium_offsets: list[float],
    high_offsets: list[float],
    bucket_column: str = "realized_vol_regime_bucket",
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for global_threshold in global_threshold_grid:
        for medium_offset in medium_offsets:
            for high_offset in high_offsets:
                threshold_map = {
                    **build_threshold_map(
                        float(global_threshold),
                        medium_offset=float(medium_offset),
                        high_offset=float(high_offset),
                    )
                }
                candidate_config = PolicyConfig(
                    edge_threshold_cents=float(global_threshold),
                    min_tau_minutes=config.min_tau_minutes,
                    max_tau_minutes=config.max_tau_minutes,
                    price_band_min_cents=config.price_band_min_cents,
                    price_band_max_cents=config.price_band_max_cents,
                    reserve_cash_pct=config.reserve_cash_pct,
                    maintain_edge_cents=config.maintain_edge_cents,
                    apply_regime_hard_gate=config.apply_regime_hard_gate,
                    starting_cash_dollars=config.starting_cash_dollars,
                    contracts_per_order=config.contracts_per_order,
                    capital_pct_per_order=config.capital_pct_per_order,
                    kelly_fraction_multiplier=config.kelly_fraction_multiplier,
                    kelly_fraction_cap_pct=config.kelly_fraction_cap_pct,
                    slippage_pct=config.slippage_pct,
                    allow_stacking=config.allow_stacking,
                    max_entries_per_ticker=config.max_entries_per_ticker,
                    require_price_improvement_for_stack=config.require_price_improvement_for_stack,
                )
                metrics, trade_records = evaluate_standard_policy_with_threshold_overrides(
                    frame,
                    probability_column=probability_column,
                    config=candidate_config,
                    overall_log_loss=overall_log_loss,
                    threshold_cents_by_bucket=threshold_map,
                    bucket_column=bucket_column,
                )
                metrics = dict(metrics)
                metrics["offsets"] = {"medium": float(medium_offset), "high": float(high_offset)}
                metrics["fit_score"] = float(metrics["net_pnl_dollars"])
                candidates.append(metrics)

    def candidate_sort_key(candidate: dict[str, object]) -> tuple[float, float, float]:
        offsets = candidate["offsets"]
        total_abs_offset = abs(float(offsets["medium"])) + abs(float(offsets["high"]))
        return (float(candidate["fit_score"]), float(candidate["objective"]), -total_abs_offset)

    candidates.sort(key=candidate_sort_key, reverse=True)
    best_candidate = candidates[0]
    return {
        "best_candidate": best_candidate,
        "candidate_count": len(candidates),
        "all_candidates": candidates,
    }


__all__ = [
    "build_threshold_map",
    "evaluate_standard_policy_with_threshold_overrides",
    "search_regime_threshold_offsets",
    "split_time_ordered_frame",
    "summarize_probability_distribution",
    "summarize_tail_calibration",
    "summarize_trade_pnl_distribution",
]
