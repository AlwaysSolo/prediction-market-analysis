from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path("output") / "live" / "kalshi_all_models_20260330_hourlyfix"
DEFAULT_ARTIFACTS_ROOT = Path("artifacts") / "kalshi"

EDGE_BUCKETS: tuple[tuple[float, float | None], ...] = (
    (0.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
    (20.0, 40.0),
    (40.0, 60.0),
    (60.0, None),
)
EDGE_BUCKET_ORDER = tuple(
    (f"{int(lower)}-{int(upper)}" if upper is not None else f"{int(lower)}+")
    for lower, upper in EDGE_BUCKETS
)
PRICE_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 10),
    (10, 20),
    (20, 30),
    (30, 40),
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 80),
    (80, 90),
    (90, 100),
)
PRICE_BUCKET_ORDER = tuple(f"{lower}-{upper}" for lower, upper in PRICE_BUCKETS)


@dataclass(frozen=True)
class SettledTrade:
    model: str
    decision_id: str
    ticker: str
    approved_at: str
    settled_at: str
    trade_day: str
    side: str
    settlement_result: str
    is_win: bool
    pnl_dollars: float
    cumulative_pnl_dollars: float
    contracts: int
    cash_required_dollars: float
    feature_basis_market_prob: float | None
    predicted_yes_probability: float | None
    raw_model_edge: float | None
    yes_post_cost_edge: float | None
    no_post_cost_edge: float | None
    yes_post_cost_edge_cents: float | None
    no_post_cost_edge_cents: float | None
    chosen_edge_cents: float | None
    chosen_edge_bucket: str
    reference_price_cents: int | None
    reference_price_bucket: str
    max_acceptable_entry_price_cents: int | None
    buy_yes_price_cents: int | None
    buy_no_price_cents: int | None
    quote_mid_prob: float | None
    quote_spread_cents: int | None
    quote_age_seconds: float | None
    tau_minutes: float | None
    offline_rule_side: str | None
    same_as_offline_rule: bool | None
    one_sided_quote: bool


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _discover_models(run_dir: Path) -> list[str]:
    signal_root = run_dir / "signal"
    execution_root = run_dir / "execution"
    signal_models = {path.name for path in signal_root.iterdir() if path.is_dir()} if signal_root.exists() else set()
    execution_models = {path.name for path in execution_root.iterdir() if path.is_dir()} if execution_root.exists() else set()
    return sorted(signal_models & execution_models)


def _edge_bucket_label(edge_cents: float | None) -> str:
    if edge_cents is None:
        return "unknown"
    if edge_cents < 0.0:
        return "<0"
    for lower, upper in EDGE_BUCKETS:
        if upper is None and edge_cents >= lower:
            return f"{int(lower)}+"
        if upper is not None and lower <= edge_cents < upper:
            return f"{int(lower)}-{int(upper)}"
    return "unknown"


def _price_bucket_label(price_cents: int | None) -> str:
    if price_cents is None:
        return "unknown"
    for lower, upper in PRICE_BUCKETS:
        if lower <= price_cents < upper:
            return f"{lower}-{upper}"
    if price_cents == 100:
        return "90-100"
    return "unknown"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _load_model_trades(run_dir: Path, model: str) -> list[SettledTrade]:
    approvals: dict[str, tuple[str, dict[str, Any]]] = {}
    signal_dir = run_dir / "signal" / model / "demo"
    for path in sorted(signal_dir.rglob("events.jsonl")):
        for event in _iter_jsonl(path):
            if event.get("event_type") != "signal_decision":
                continue
            payload = event.get("payload", {})
            if not payload.get("approved"):
                continue
            decision_id = payload.get("decision_id")
            if not isinstance(decision_id, str):
                continue
            approvals[decision_id] = (str(event.get("logged_at")), payload)

    trades: list[SettledTrade] = []
    execution_dir = run_dir / "execution" / model / "demo"
    for path in sorted(execution_dir.rglob("events.jsonl")):
        for event in _iter_jsonl(path):
            if event.get("event_type") != "simulated_position_settled":
                continue
            payload = event.get("payload", {})
            decision_id = payload.get("decision_id")
            if not isinstance(decision_id, str):
                continue
            approval = approvals.get(decision_id)
            if approval is None:
                continue
            approved_at, approved_payload = approval

            side = str(payload.get("side"))
            yes_edge = _safe_float(approved_payload.get("yes_post_cost_edge"))
            no_edge = _safe_float(approved_payload.get("no_post_cost_edge"))
            chosen_edge = yes_edge if side == "YES" else no_edge
            predicted_yes_probability = _safe_float(approved_payload.get("predicted_yes_probability"))
            feature_basis_market_prob = _safe_float(approved_payload.get("feature_basis_market_prob"))
            offline_rule_side: str | None = None
            same_as_offline_rule: bool | None = None
            if predicted_yes_probability is not None and feature_basis_market_prob is not None:
                offline_rule_side = "YES" if predicted_yes_probability > feature_basis_market_prob else "NO"
                same_as_offline_rule = offline_rule_side == side

            yes_bid_cents = _safe_int(approved_payload.get("yes_bid_cents"))
            yes_ask_cents = _safe_int(approved_payload.get("yes_ask_cents"))
            one_sided_quote = bool((yes_bid_cents == 0) or (yes_ask_cents == 100))

            trade = SettledTrade(
                model=model,
                decision_id=decision_id,
                ticker=str(payload.get("ticker")),
                approved_at=approved_at,
                settled_at=str(event.get("logged_at")),
                trade_day=str(event.get("logged_at", ""))[:10],
                side=side,
                settlement_result=str(payload.get("settlement_result")),
                is_win=float(payload.get("realized_pnl_dollars", 0.0)) > 0.0,
                pnl_dollars=float(payload.get("realized_pnl_dollars", 0.0)),
                cumulative_pnl_dollars=float(payload.get("cumulative_realized_pnl_dollars", 0.0)),
                contracts=int(payload.get("contracts", 0)),
                cash_required_dollars=float(payload.get("cash_required_dollars", 0.0)),
                feature_basis_market_prob=feature_basis_market_prob,
                predicted_yes_probability=predicted_yes_probability,
                raw_model_edge=_safe_float(approved_payload.get("raw_model_edge")),
                yes_post_cost_edge=yes_edge,
                no_post_cost_edge=no_edge,
                yes_post_cost_edge_cents=(None if yes_edge is None else yes_edge * 100.0),
                no_post_cost_edge_cents=(None if no_edge is None else no_edge * 100.0),
                chosen_edge_cents=(None if chosen_edge is None else chosen_edge * 100.0),
                chosen_edge_bucket=_edge_bucket_label(None if chosen_edge is None else chosen_edge * 100.0),
                reference_price_cents=_safe_int(approved_payload.get("reference_price_cents")),
                reference_price_bucket=_price_bucket_label(_safe_int(approved_payload.get("reference_price_cents"))),
                max_acceptable_entry_price_cents=_safe_int(approved_payload.get("max_acceptable_entry_price_cents")),
                buy_yes_price_cents=_safe_int(approved_payload.get("buy_yes_price_cents")),
                buy_no_price_cents=_safe_int(approved_payload.get("buy_no_price_cents")),
                quote_mid_prob=_safe_float(approved_payload.get("quote_mid_prob")),
                quote_spread_cents=_safe_int(approved_payload.get("quote_spread_cents")),
                quote_age_seconds=_safe_float(approved_payload.get("quote_age_seconds")),
                tau_minutes=_safe_float(approved_payload.get("tau_minutes")),
                offline_rule_side=offline_rule_side,
                same_as_offline_rule=same_as_offline_rule,
                one_sided_quote=one_sided_quote,
            )
            trades.append(trade)

    trades.sort(key=lambda item: (item.settled_at, item.model, item.ticker, item.decision_id))
    return trades


def _summary_rows(
    trades: list[SettledTrade],
    *,
    scope: str,
    model: str | None = None,
    taken_side: str | None = None,
) -> list[dict[str, Any]]:
    return _summary_rows_by_bucket(
        trades,
        scope=scope,
        bucket_attr="chosen_edge_bucket",
        bucket_order=EDGE_BUCKET_ORDER,
        bucket_key="edge_bucket",
        model=model,
        taken_side=taken_side,
    )


def _price_summary_rows(
    trades: list[SettledTrade],
    *,
    scope: str,
    model: str | None = None,
    taken_side: str | None = None,
) -> list[dict[str, Any]]:
    return _summary_rows_by_bucket(
        trades,
        scope=scope,
        bucket_attr="reference_price_bucket",
        bucket_order=PRICE_BUCKET_ORDER,
        bucket_key="price_bucket",
        model=model,
        taken_side=taken_side,
    )


def _summary_rows_by_bucket(
    trades: list[SettledTrade],
    *,
    scope: str,
    bucket_attr: str,
    bucket_order: tuple[str, ...],
    bucket_key: str,
    model: str | None = None,
    taken_side: str | None = None,
) -> list[dict[str, Any]]:
    filtered = [
        trade
        for trade in trades
        if (model is None or trade.model == model) and (taken_side is None or trade.side == taken_side)
    ]
    grouped: dict[str, list[SettledTrade]] = defaultdict(list)
    for trade in filtered:
        grouped[str(getattr(trade, bucket_attr))].append(trade)

    rows: list[dict[str, Any]] = []
    for bucket in bucket_order:
        bucket_trades = grouped.get(bucket, [])
        if not bucket_trades:
            continue
        model_pnls: dict[str, float] = defaultdict(float)
        for trade in bucket_trades:
            model_pnls[trade.model] += trade.pnl_dollars
        profitable_models = sorted(model_name for model_name, pnl in model_pnls.items() if pnl > 0.0)
        rows.append(
            {
                "scope": scope,
                "model": model or "ALL_MODELS",
                "taken_side": taken_side or "ALL",
                bucket_key: bucket,
                "trades": len(bucket_trades),
                "wins": sum(trade.is_win for trade in bucket_trades),
                "losses": sum(not trade.is_win for trade in bucket_trades),
                "win_rate": sum(trade.is_win for trade in bucket_trades) / len(bucket_trades),
                "net_pnl_dollars": sum(trade.pnl_dollars for trade in bucket_trades),
                "avg_pnl_dollars": sum(trade.pnl_dollars for trade in bucket_trades) / len(bucket_trades),
                "avg_edge_cents": sum(
                    trade.chosen_edge_cents or 0.0 for trade in bucket_trades
                )
                / len(bucket_trades),
                "avg_contracts": sum(trade.contracts for trade in bucket_trades) / len(bucket_trades),
                "avg_cash_required_dollars": sum(trade.cash_required_dollars for trade in bucket_trades)
                / len(bucket_trades),
                "yes_trades": sum(trade.side == "YES" for trade in bucket_trades),
                "no_trades": sum(trade.side == "NO" for trade in bucket_trades),
                "models_with_trades": len(model_pnls),
                "profitable_models": len(profitable_models),
                "profitable_model_names": ",".join(profitable_models),
            }
        )
    return rows


def _model_total_rows(trades: list[SettledTrade]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in sorted({trade.model for trade in trades}):
        model_trades = [trade for trade in trades if trade.model == model]
        rows.append(
            {
                "model": model,
                "trades": len(model_trades),
                "wins": sum(trade.is_win for trade in model_trades),
                "losses": sum(not trade.is_win for trade in model_trades),
                "win_rate": sum(trade.is_win for trade in model_trades) / len(model_trades),
                "net_pnl_dollars": sum(trade.pnl_dollars for trade in model_trades),
                "avg_pnl_dollars": sum(trade.pnl_dollars for trade in model_trades) / len(model_trades),
                "yes_trades": sum(trade.side == "YES" for trade in model_trades),
                "no_trades": sum(trade.side == "NO" for trade in model_trades),
            }
        )
    return rows


def _recommendation_status(row: dict[str, Any]) -> tuple[str, str]:
    trades = int(row["trades"])
    win_rate = float(row["win_rate"])
    pnl = float(row["net_pnl_dollars"])
    profitable_models = int(row["profitable_models"])
    models_with_trades = int(row["models_with_trades"])

    if trades == 0:
        return "No Sample", "No settled trades in this bucket."
    if pnl > 0.0 and win_rate >= 0.45 and profitable_models >= max(1, models_with_trades // 2):
        return "Allow", "Positive aggregate pnl with decent hit rate and at least mixed model support."
    if pnl < 0.0 and (win_rate < 0.40 or profitable_models <= 1):
        return "Ban", "Negative aggregate pnl with weak hit rate or almost no model support."
    return "Watch", "Mixed signal. Keep under review or gate behind tighter filters."


def _recommendation_rows(trades: list[SettledTrade]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side in ("YES", "NO"):
        for summary in _summary_rows(trades, scope=f"aggregate_{side.lower()}", taken_side=side):
            status, note = _recommendation_status(summary)
            row = dict(summary)
            row["status"] = status
            row["note"] = note
            rows.append(row)
    return rows


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return table


def _render_summary_table(summary_rows: list[dict[str, Any]]) -> list[list[str]]:
    return _render_bucket_summary_table(summary_rows, bucket_key="edge_bucket")


def _render_price_summary_table(summary_rows: list[dict[str, Any]]) -> list[list[str]]:
    return _render_bucket_summary_table(summary_rows, bucket_key="price_bucket")


def _render_bucket_summary_table(summary_rows: list[dict[str, Any]], *, bucket_key: str) -> list[list[str]]:
    return [
        [
            row[bucket_key],
            str(row["trades"]),
            str(row["wins"]),
            str(row["losses"]),
            _format_pct(float(row["win_rate"])),
            _format_money(float(row["net_pnl_dollars"])),
            _format_money(float(row["avg_pnl_dollars"])),
            str(row["yes_trades"]),
            str(row["no_trades"]),
        ]
        for row in summary_rows
    ]


def _render_recommendation_table(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            str(row["taken_side"]),
            str(row["edge_bucket"]),
            str(row["trades"]),
            _format_pct(float(row["win_rate"])),
            _format_money(float(row["net_pnl_dollars"])),
            f'{row["profitable_models"]}/{row["models_with_trades"]}',
            str(row["status"]),
            str(row["note"]),
        ]
        for row in rows
    ]


def _write_price_markdown_report(
    path: Path,
    *,
    run_dir: Path,
    trades: list[SettledTrade],
    model_totals: list[dict[str, Any]],
    aggregate_all: list[dict[str, Any]],
    aggregate_yes: list[dict[str, Any]],
    aggregate_no: list[dict[str, Any]],
    per_model_all: dict[str, list[dict[str, Any]]],
) -> None:
    lines: list[str] = []
    generated_at = datetime.now(UTC).isoformat()
    total_wins = sum(trade.is_win for trade in trades)
    total_pnl = sum(trade.pnl_dollars for trade in trades)

    lines.append("# Live Paper Price-Bucket Analysis")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Generated at: `{generated_at}`")
    lines.append(f"- Settled trades analyzed: `{len(trades)}`")
    lines.append(f"- Total wins: `{total_wins}`")
    lines.append(f"- Aggregate win rate: `{_format_pct(total_wins / len(trades)) if trades else '0.0%'}`")
    lines.append(f"- Aggregate net pnl: `{_format_money(total_pnl)}`")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Each settled paper trade was joined to its approved signal by `decision_id`.")
    lines.append("- Price buckets use the chosen trade's `reference_price_cents`, which is the contract price the bot actually tried to buy.")
    lines.append("- Buckets are in 10-cent ranges: `0-10`, `10-20`, `20-30`, `30-40`, `40-50`, `50-60`, `60-70`, `70-80`, `80-90`, and `90-100`.")
    lines.append("- Aggregate YES and aggregate NO sections split the settled trades by the side the bot actually took.")
    lines.append("")
    lines.append("## Model Totals")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["Model", "Trades", "Wins", "Losses", "Win Rate", "Net PnL", "Avg PnL", "YES", "NO"],
            [
                [
                    row["model"],
                    str(row["trades"]),
                    str(row["wins"]),
                    str(row["losses"]),
                    _format_pct(float(row["win_rate"])),
                    _format_money(float(row["net_pnl_dollars"])),
                    _format_money(float(row["avg_pnl_dollars"])),
                    str(row["yes_trades"]),
                    str(row["no_trades"]),
                ]
                for row in model_totals
            ],
        )
    )
    lines.append("")
    lines.append("## Aggregate Price Buckets")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["Price Bucket", "Trades", "Wins", "Losses", "Win Rate", "Net PnL", "Avg PnL", "YES", "NO"],
            _render_price_summary_table(aggregate_all),
        )
    )
    lines.append("")
    lines.append("## Aggregate YES Trades By Price")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["Price Bucket", "Trades", "Wins", "Losses", "Win Rate", "Net PnL", "Avg PnL", "YES", "NO"],
            _render_price_summary_table(aggregate_yes),
        )
    )
    lines.append("")
    lines.append("## Aggregate NO Trades By Price")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["Price Bucket", "Trades", "Wins", "Losses", "Win Rate", "Net PnL", "Avg PnL", "YES", "NO"],
            _render_price_summary_table(aggregate_no),
        )
    )
    lines.append("")
    best_bucket = max(aggregate_all, key=lambda row: float(row["net_pnl_dollars"])) if aggregate_all else None
    worst_bucket = min(aggregate_all, key=lambda row: float(row["net_pnl_dollars"])) if aggregate_all else None
    lines.append("## Extra Findings")
    lines.append("")
    if best_bucket is not None:
        lines.append(
            f"- Best aggregate price bucket: `{best_bucket['price_bucket']}` with `{best_bucket['trades']}` trades and `{_format_money(float(best_bucket['net_pnl_dollars']))}` net pnl."
        )
    if worst_bucket is not None:
        lines.append(
            f"- Worst aggregate price bucket: `{worst_bucket['price_bucket']}` with `{worst_bucket['trades']}` trades and `{_format_money(float(worst_bucket['net_pnl_dollars']))}` net pnl."
        )
    lines.append("")

    for model in sorted(per_model_all):
        lines.append(f"## {model}")
        lines.append("")
        lines.extend(
            _markdown_table(
                ["Price Bucket", "Trades", "Wins", "Losses", "Win Rate", "Net PnL", "Avg PnL", "YES", "NO"],
                _render_price_summary_table(per_model_all[model]),
            )
        )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(
    path: Path,
    *,
    run_dir: Path,
    trades: list[SettledTrade],
    model_totals: list[dict[str, Any]],
    aggregate_all: list[dict[str, Any]],
    aggregate_yes: list[dict[str, Any]],
    aggregate_no: list[dict[str, Any]],
    per_model_all: dict[str, list[dict[str, Any]]],
    per_model_yes: dict[str, list[dict[str, Any]]],
    per_model_no: dict[str, list[dict[str, Any]]],
    recommendations: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    generated_at = datetime.now(UTC).isoformat()
    total_wins = sum(trade.is_win for trade in trades)
    total_pnl = sum(trade.pnl_dollars for trade in trades)

    lines.append("# Live Paper Edge-Bucket Analysis")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Generated at: `{generated_at}`")
    lines.append(f"- Settled trades analyzed: `{len(trades)}`")
    lines.append(f"- Total wins: `{total_wins}`")
    lines.append(f"- Aggregate win rate: `{_format_pct(total_wins / len(trades)) if trades else '0.0%'}`")
    lines.append(f"- Aggregate net pnl: `{_format_money(total_pnl)}`")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Each settled paper trade was joined to its approved signal by `decision_id`.")
    lines.append("- `YES` trades are bucketed by `yes_post_cost_edge`; `NO` trades are bucketed by `no_post_cost_edge`.")
    lines.append("- Edge buckets are measured in post-cost cents: `0-5`, `5-10`, `10-20`, `20-40`, `40-60`, and `60+`.")
    lines.append("- Side-specific tables below split the chosen trades into `YES` and `NO` buckets separately.")
    lines.append("")
    lines.append("## Model Totals")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["Model", "Trades", "Wins", "Losses", "Win Rate", "Net PnL", "Avg PnL", "YES", "NO"],
            [
                [
                    row["model"],
                    str(row["trades"]),
                    str(row["wins"]),
                    str(row["losses"]),
                    _format_pct(float(row["win_rate"])),
                    _format_money(float(row["net_pnl_dollars"])),
                    _format_money(float(row["avg_pnl_dollars"])),
                    str(row["yes_trades"]),
                    str(row["no_trades"]),
                ]
                for row in model_totals
            ],
        )
    )
    lines.append("")
    lines.append("## Aggregate Chosen-Side Buckets")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["Bucket", "Trades", "Wins", "Losses", "Win Rate", "Net PnL", "Avg PnL", "YES", "NO"],
            _render_summary_table(aggregate_all),
        )
    )
    lines.append("")
    lines.append("## Aggregate YES Trades")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["YES Edge Bucket", "Trades", "Wins", "Losses", "Win Rate", "Net PnL", "Avg PnL", "YES", "NO"],
            _render_summary_table(aggregate_yes),
        )
    )
    lines.append("")
    lines.append("## Aggregate NO Trades")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["NO Edge Bucket", "Trades", "Wins", "Losses", "Win Rate", "Net PnL", "Avg PnL", "YES", "NO"],
            _render_summary_table(aggregate_no),
        )
    )
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["Side", "Bucket", "Trades", "Win Rate", "Net PnL", "Positive Models", "Status", "Note"],
            _render_recommendation_table(recommendations),
        )
    )
    lines.append("")
    lines.append("## Extra Findings")
    lines.append("")
    worst_bucket = min(aggregate_all, key=lambda row: float(row["net_pnl_dollars"])) if aggregate_all else None
    best_bucket = max(aggregate_all, key=lambda row: float(row["net_pnl_dollars"])) if aggregate_all else None
    if worst_bucket is not None:
        lines.append(
            f"- Worst chosen-side edge bucket: `{worst_bucket['edge_bucket']}` with `{worst_bucket['trades']}` trades and `{_format_money(float(worst_bucket['net_pnl_dollars']))}` net pnl."
        )
    if best_bucket is not None:
        lines.append(
            f"- Best chosen-side edge bucket: `{best_bucket['edge_bucket']}` with `{best_bucket['trades']}` trades and `{_format_money(float(best_bucket['net_pnl_dollars']))}` net pnl."
        )
    lines.append(
        f"- One-sided quotes (`yes_bid=0` or `yes_ask=100`) appeared in `{sum(trade.one_sided_quote for trade in trades)}` settled trades."
    )
    lines.append(
        f"- Trades that disagreed with the offline side rule accounted for `{sum(not trade.same_as_offline_rule for trade in trades if trade.same_as_offline_rule is not None)}` settled trades."
    )
    lines.append("")

    for model in sorted(per_model_all):
        lines.append(f"## {model}")
        lines.append("")
        lines.append("### Chosen-Side Buckets")
        lines.append("")
        lines.extend(
            _markdown_table(
                ["Bucket", "Trades", "Wins", "Losses", "Win Rate", "Net PnL", "Avg PnL", "YES", "NO"],
                _render_summary_table(per_model_all[model]),
            )
        )
        lines.append("")
        lines.append("### YES Trades")
        lines.append("")
        lines.extend(
            _markdown_table(
                ["YES Edge Bucket", "Trades", "Wins", "Losses", "Win Rate", "Net PnL", "Avg PnL", "YES", "NO"],
                _render_summary_table(per_model_yes[model]),
            )
        )
        lines.append("")
        lines.append("### NO Trades")
        lines.append("")
        lines.extend(
            _markdown_table(
                ["NO Edge Bucket", "Trades", "Wins", "Losses", "Win Rate", "Net PnL", "Avg PnL", "YES", "NO"],
                _render_summary_table(per_model_no[model]),
            )
        )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report live Kalshi paper-trade win/loss by YES/NO edge bucket.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Live run directory containing signal/ and execution/ subdirectories.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory where markdown and CSV artifacts should be written.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional artifact filename prefix. Defaults to the run directory name.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    prefix = args.prefix or run_dir.name
    models = _discover_models(run_dir)
    if not models:
        raise RuntimeError(f"No shared signal/execution model directories found under {run_dir}")

    trades: list[SettledTrade] = []
    for model in models:
        trades.extend(_load_model_trades(run_dir, model))

    model_totals = _model_total_rows(trades)
    aggregate_all = _summary_rows(trades, scope="aggregate_all")
    aggregate_yes = _summary_rows(trades, scope="aggregate_yes", taken_side="YES")
    aggregate_no = _summary_rows(trades, scope="aggregate_no", taken_side="NO")
    per_model_all = {model: _summary_rows(trades, scope="model_all", model=model) for model in models}
    per_model_yes = {
        model: _summary_rows(trades, scope="model_yes", model=model, taken_side="YES") for model in models
    }
    per_model_no = {
        model: _summary_rows(trades, scope="model_no", model=model, taken_side="NO") for model in models
    }
    recommendations = _recommendation_rows(trades)
    price_aggregate_all = _price_summary_rows(trades, scope="aggregate_all_price")
    price_aggregate_yes = _price_summary_rows(trades, scope="aggregate_yes_price", taken_side="YES")
    price_aggregate_no = _price_summary_rows(trades, scope="aggregate_no_price", taken_side="NO")
    price_per_model_all = {
        model: _price_summary_rows(trades, scope="model_all_price", model=model) for model in models
    }

    summary_rows = (
        aggregate_all
        + aggregate_yes
        + aggregate_no
        + [row for model in models for row in per_model_all[model]]
        + [row for model in models for row in per_model_yes[model]]
        + [row for model in models for row in per_model_no[model]]
    )
    price_summary_rows = (
        price_aggregate_all
        + price_aggregate_yes
        + price_aggregate_no
        + [row for model in models for row in price_per_model_all[model]]
    )

    trades_csv = args.artifacts_root / f"{prefix}_edge_bucket_trades.csv"
    summary_csv = args.artifacts_root / f"{prefix}_edge_bucket_summary.csv"
    totals_csv = args.artifacts_root / f"{prefix}_edge_bucket_model_totals.csv"
    recommendations_csv = args.artifacts_root / f"{prefix}_edge_bucket_recommendations.csv"
    report_md = args.artifacts_root / f"{prefix}_edge_bucket_report.md"
    price_summary_csv = args.artifacts_root / f"{prefix}_price_bucket_summary.csv"
    price_report_md = args.artifacts_root / f"{prefix}_price_bucket_report.md"

    _write_csv(trades_csv, [trade.__dict__ for trade in trades])
    _write_csv(summary_csv, summary_rows)
    _write_csv(totals_csv, model_totals)
    _write_csv(recommendations_csv, recommendations)
    _write_csv(price_summary_csv, price_summary_rows)
    _write_markdown_report(
        report_md,
        run_dir=run_dir,
        trades=trades,
        model_totals=model_totals,
        aggregate_all=aggregate_all,
        aggregate_yes=aggregate_yes,
        aggregate_no=aggregate_no,
        per_model_all=per_model_all,
        per_model_yes=per_model_yes,
        per_model_no=per_model_no,
        recommendations=recommendations,
    )
    _write_price_markdown_report(
        price_report_md,
        run_dir=run_dir,
        trades=trades,
        model_totals=model_totals,
        aggregate_all=price_aggregate_all,
        aggregate_yes=price_aggregate_yes,
        aggregate_no=price_aggregate_no,
        per_model_all=price_per_model_all,
    )

    print(f"Wrote report: {report_md}")
    print(f"Wrote trade CSV: {trades_csv}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote model totals CSV: {totals_csv}")
    print(f"Wrote recommendations CSV: {recommendations_csv}")
    print(f"Wrote price summary CSV: {price_summary_csv}")
    print(f"Wrote price report: {price_report_md}")


if __name__ == "__main__":
    main()
