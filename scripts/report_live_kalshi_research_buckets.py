from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path("output") / "live_research" / "kalshi"
DEFAULT_ARTIFACTS_ROOT = Path("artifacts") / "kalshi"

TAU_BUCKET_ORDER = ("2-4", "4-6", "6-8", "8-10", "10-12", "12-14")
PRICE_BUCKET_ORDER = tuple(f"{value}-{value + 10}" for value in range(0, 100, 10))
PROBABILITY_BUCKET_ORDER = tuple(f"{value}-{value + 10}" for value in range(0, 100, 10))
EDGE_BUCKET_ORDER = ("<0", "0-5", "5-10", "10-20", "20-40", "40-60", "60+")


@dataclass(frozen=True)
class ResearchSampleRow:
    model: str
    sample_id: str
    ticker: str
    status: str
    recorded_at: str
    settled_at: str | None
    side: str
    settlement_result: str | None
    is_win: bool | None
    realized_pnl_dollars: float | None
    cumulative_realized_pnl_dollars: float | None
    predicted_yes_probability: float
    predicted_no_probability: float
    chosen_side_probability: float
    feature_basis_market_prob: float
    raw_model_edge: float
    yes_post_cost_edge: float | None
    no_post_cost_edge: float | None
    chosen_post_cost_edge: float
    chosen_edge_cents: float
    reference_price_cents: int
    tau_minutes: float
    tau_bucket: str
    price_bucket: str
    chosen_side_probability_bucket: str
    chosen_side_edge_bucket: str
    quote_spread_cents: int | None
    quote_age_seconds: float | None
    estimated_cash_required_dollars: float


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
    research_root = run_dir / "research"
    if not research_root.exists():
        return []
    return sorted(path.name for path in research_root.iterdir() if path.is_dir())


def _load_model_samples(run_dir: Path, model: str) -> list[ResearchSampleRow]:
    recorded: dict[str, tuple[str, dict[str, Any]]] = {}
    settled: dict[str, tuple[str, dict[str, Any]]] = {}
    events_root = run_dir / "research" / model / "demo"
    for path in sorted(events_root.rglob("events.jsonl")):
        for event in _iter_jsonl(path):
            event_type = event.get("event_type")
            payload = event.get("payload", {})
            sample_id = payload.get("sample_id")
            if not isinstance(sample_id, str):
                continue
            if event_type == "research_sample_recorded":
                recorded[sample_id] = (str(event.get("logged_at")), payload)
            elif event_type == "research_sample_settled":
                settled[sample_id] = (str(event.get("logged_at")), payload)

    rows: list[ResearchSampleRow] = []
    for sample_id, (recorded_at, payload) in sorted(recorded.items()):
        settlement = settled.get(sample_id)
        settled_at = None if settlement is None else settlement[0]
        settled_payload = None if settlement is None else settlement[1]
        rows.append(
            ResearchSampleRow(
                model=model,
                sample_id=sample_id,
                ticker=str(payload["ticker"]),
                status="settled" if settled_payload is not None else "open",
                recorded_at=recorded_at,
                settled_at=settled_at,
                side=str(payload["side"]),
                settlement_result=None if settled_payload is None else str(settled_payload["settlement_result"]),
                is_win=None if settled_payload is None else bool(settled_payload["is_win"]),
                realized_pnl_dollars=None if settled_payload is None else float(settled_payload["realized_pnl_dollars"]),
                cumulative_realized_pnl_dollars=(
                    None if settled_payload is None else float(settled_payload["cumulative_realized_pnl_dollars"])
                ),
                predicted_yes_probability=float(payload["predicted_yes_probability"]),
                predicted_no_probability=float(payload["predicted_no_probability"]),
                chosen_side_probability=float(payload["chosen_side_probability"]),
                feature_basis_market_prob=float(payload["feature_basis_market_prob"]),
                raw_model_edge=float(payload["raw_model_edge"]),
                yes_post_cost_edge=(
                    None if payload.get("yes_post_cost_edge") is None else float(payload["yes_post_cost_edge"])
                ),
                no_post_cost_edge=(
                    None if payload.get("no_post_cost_edge") is None else float(payload["no_post_cost_edge"])
                ),
                chosen_post_cost_edge=float(payload["chosen_post_cost_edge"]),
                chosen_edge_cents=float(payload["chosen_post_cost_edge"]) * 100.0,
                reference_price_cents=int(payload["reference_price_cents"]),
                tau_minutes=float(payload["tau_minutes"]),
                tau_bucket=str(payload["tau_bucket"]),
                price_bucket=str(payload["price_bucket"]),
                chosen_side_probability_bucket=str(payload["chosen_side_probability_bucket"]),
                chosen_side_edge_bucket=str(payload["chosen_side_edge_bucket"]),
                quote_spread_cents=None if payload.get("quote_spread_cents") is None else int(payload["quote_spread_cents"]),
                quote_age_seconds=None if payload.get("quote_age_seconds") is None else float(payload["quote_age_seconds"]),
                estimated_cash_required_dollars=float(payload["estimated_cash_required_dollars"]),
            )
        )
    return rows


def _summary_rows(
    rows: list[ResearchSampleRow],
    *,
    dimension: str,
    bucket_attr: str,
    bucket_order: tuple[str, ...],
    model: str | None = None,
) -> list[dict[str, Any]]:
    settled_rows = [row for row in rows if row.status == "settled" and (model is None or row.model == model)]
    grouped: dict[str, list[ResearchSampleRow]] = defaultdict(list)
    for row in settled_rows:
        grouped[str(getattr(row, bucket_attr))].append(row)

    summary: list[dict[str, Any]] = []
    for bucket in bucket_order:
        bucket_rows = grouped.get(bucket, [])
        wins = sum(1 for row in bucket_rows if row.is_win)
        settled_count = len(bucket_rows)
        net_pnl = sum(float(row.realized_pnl_dollars or 0.0) for row in bucket_rows)
        summary.append(
            {
                "model": model or "all_models",
                "dimension": dimension,
                "bucket": bucket,
                "settled_count": settled_count,
                "wins": wins,
                "losses": settled_count - wins,
                "win_rate": None if settled_count == 0 else wins / settled_count,
                "net_pnl_dollars": net_pnl,
                "avg_pnl_dollars": None if settled_count == 0 else net_pnl / settled_count,
            }
        )
    return summary


def _model_totals(rows: list[ResearchSampleRow]) -> list[dict[str, Any]]:
    totals: list[dict[str, Any]] = []
    for model in sorted({row.model for row in rows}):
        model_rows = [row for row in rows if row.model == model]
        settled_rows = [row for row in model_rows if row.status == "settled"]
        wins = sum(1 for row in settled_rows if row.is_win)
        net_pnl = sum(float(row.realized_pnl_dollars or 0.0) for row in settled_rows)
        totals.append(
            {
                "model": model,
                "recorded_count": len(model_rows),
                "settled_count": len(settled_rows),
                "open_count": len(model_rows) - len(settled_rows),
                "wins": wins,
                "losses": len(settled_rows) - wins,
                "win_rate": None if not settled_rows else wins / len(settled_rows),
                "net_pnl_dollars": net_pnl,
                "avg_pnl_dollars": None if not settled_rows else net_pnl / len(settled_rows),
            }
        )
    return totals


def _combo_rows(rows: list[ResearchSampleRow]) -> list[dict[str, Any]]:
    settled_rows = [row for row in rows if row.status == "settled"]
    grouped: dict[tuple[str, str, str, str], list[ResearchSampleRow]] = defaultdict(list)
    for row in settled_rows:
        grouped[
            (
                row.tau_bucket,
                row.price_bucket,
                row.chosen_side_probability_bucket,
                row.chosen_side_edge_bucket,
            )
        ].append(row)

    combos: list[dict[str, Any]] = []
    for combo, combo_rows in grouped.items():
        if len(combo_rows) < 3:
            continue
        wins = sum(1 for row in combo_rows if row.is_win)
        net_pnl = sum(float(row.realized_pnl_dollars or 0.0) for row in combo_rows)
        combos.append(
            {
                "tau_bucket": combo[0],
                "price_bucket": combo[1],
                "probability_bucket": combo[2],
                "edge_bucket": combo[3],
                "settled_count": len(combo_rows),
                "wins": wins,
                "win_rate": wins / len(combo_rows),
                "net_pnl_dollars": net_pnl,
                "avg_pnl_dollars": net_pnl / len(combo_rows),
            }
        )
    return combos


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _format_money(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${value:+.2f}"


def _markdown_table(rows: list[dict[str, Any]], headers: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for label, _key in headers) + " |",
        "| " + " | ".join("---" for _label, _key in headers) + " |",
    ]
    for row in rows:
        rendered: list[str] = []
        for _label, key in headers:
            value = row.get(key)
            if key.endswith("win_rate"):
                rendered.append(_format_pct(value))
            elif key.endswith("_pnl_dollars"):
                rendered.append(_format_money(value))
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _build_markdown(
    *,
    run_name: str,
    rows: list[ResearchSampleRow],
    totals: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    combos: list[dict[str, Any]],
) -> str:
    settled_count = sum(1 for row in rows if row.status == "settled")
    lines = [
        f"# {run_name} Research Bucket Report",
        "",
        f"Recorded samples: `{len(rows)}`",
        f"Settled samples: `{settled_count}`",
        "",
        "## Model Totals",
        _markdown_table(
            totals,
            [
                ("Model", "model"),
                ("Recorded", "recorded_count"),
                ("Settled", "settled_count"),
                ("Open", "open_count"),
                ("Wins", "wins"),
                ("Losses", "losses"),
                ("Win Rate", "win_rate"),
                ("Net PnL", "net_pnl_dollars"),
                ("Avg PnL", "avg_pnl_dollars"),
            ],
        ),
    ]

    for dimension, order in (
        ("tau", TAU_BUCKET_ORDER),
        ("price", PRICE_BUCKET_ORDER),
        ("probability", PROBABILITY_BUCKET_ORDER),
        ("edge", EDGE_BUCKET_ORDER),
    ):
        subset = [row for row in aggregate_rows if row["dimension"] == dimension]
        lines.extend(
            [
                "",
                f"## Aggregate {dimension.title()} Buckets",
                _markdown_table(
                    subset,
                    [
                        ("Bucket", "bucket"),
                        ("Settled", "settled_count"),
                        ("Wins", "wins"),
                        ("Losses", "losses"),
                        ("Win Rate", "win_rate"),
                        ("Net PnL", "net_pnl_dollars"),
                        ("Avg PnL", "avg_pnl_dollars"),
                    ],
                ),
            ]
        )

    lines.extend(["", "## Per-Model Bucket Winners"])
    winners: list[dict[str, Any]] = []
    losers: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in model_rows}):
        for dimension in ("tau", "price", "probability", "edge"):
            subset = [row for row in model_rows if row["model"] == model and row["dimension"] == dimension and row["settled_count"] > 0]
            if not subset:
                continue
            winners.append(max(subset, key=lambda row: (float(row["avg_pnl_dollars"] or -1e9), int(row["settled_count"]))))
            losers.append(min(subset, key=lambda row: (float(row["avg_pnl_dollars"] or 1e9), -int(row["settled_count"]))))
    lines.append(
        _markdown_table(
            winners,
            [
                ("Model", "model"),
                ("Dimension", "dimension"),
                ("Bucket", "bucket"),
                ("Settled", "settled_count"),
                ("Win Rate", "win_rate"),
                ("Net PnL", "net_pnl_dollars"),
                ("Avg PnL", "avg_pnl_dollars"),
            ],
        )
    )

    if combos:
        best_combos = sorted(combos, key=lambda row: (row["avg_pnl_dollars"], row["settled_count"]), reverse=True)[:10]
        worst_combos = sorted(combos, key=lambda row: (row["avg_pnl_dollars"], -row["settled_count"]))[:10]
        combo_headers = [
            ("Tau", "tau_bucket"),
            ("Price", "price_bucket"),
            ("Prob", "probability_bucket"),
            ("Edge", "edge_bucket"),
            ("Settled", "settled_count"),
            ("Win Rate", "win_rate"),
            ("Net PnL", "net_pnl_dollars"),
            ("Avg PnL", "avg_pnl_dollars"),
        ]
        lines.extend(
            [
                "",
                "## Top Bucket Combinations",
                "_Only combinations with at least 3 settled samples are shown._",
                _markdown_table(best_combos, combo_headers),
                "",
                "## Worst Bucket Combinations",
                "_Only combinations with at least 3 settled samples are shown._",
                _markdown_table(worst_combos, combo_headers),
            ]
        )

    return "\n".join(lines) + "\n"


def generate_research_bucket_report(
    run_dir: Path,
    *,
    artifacts_root: Path,
) -> dict[str, Path]:
    models = _discover_models(run_dir)
    rows: list[ResearchSampleRow] = []
    for model in models:
        rows.extend(_load_model_samples(run_dir, model))

    aggregate_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for dimension, attr, order in (
        ("tau", "tau_bucket", TAU_BUCKET_ORDER),
        ("price", "price_bucket", PRICE_BUCKET_ORDER),
        ("probability", "chosen_side_probability_bucket", PROBABILITY_BUCKET_ORDER),
        ("edge", "chosen_side_edge_bucket", EDGE_BUCKET_ORDER),
    ):
        aggregate_rows.extend(_summary_rows(rows, dimension=dimension, bucket_attr=attr, bucket_order=order))
        for model in models:
            model_rows.extend(_summary_rows(rows, dimension=dimension, bucket_attr=attr, bucket_order=order, model=model))

    totals = _model_totals(rows)
    combos = _combo_rows(rows)

    prefix = run_dir.name
    artifacts_root.mkdir(parents=True, exist_ok=True)
    samples_csv = artifacts_root / f"{prefix}_research_samples.csv"
    aggregate_csv = artifacts_root / f"{prefix}_research_bucket_summary.csv"
    per_model_csv = artifacts_root / f"{prefix}_research_model_bucket_summary.csv"
    report_md = artifacts_root / f"{prefix}_research_bucket_report.md"

    _write_csv(samples_csv, [asdict(row) for row in rows])
    _write_csv(aggregate_csv, aggregate_rows)
    _write_csv(per_model_csv, model_rows)
    report_md.write_text(
        _build_markdown(
            run_name=prefix,
            rows=rows,
            totals=totals,
            aggregate_rows=aggregate_rows,
            model_rows=model_rows,
            combos=combos,
        ),
        encoding="utf-8",
    )
    return {
        "samples_csv": samples_csv,
        "aggregate_csv": aggregate_csv,
        "per_model_csv": per_model_csv,
        "report_md": report_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bucket reports from live research sampler logs.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT))
    args = parser.parse_args()
    paths = generate_research_bucket_report(
        Path(args.run_dir),
        artifacts_root=Path(args.artifacts_root),
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
