from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_summary(path_or_dir: str) -> tuple[Path, dict[str, object]]:
    path = Path(path_or_dir).expanduser()
    summary_path = path / "summary.json" if path.is_dir() else path
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary_path, payload


def _extract_metrics(summary_path: Path, summary: dict[str, object]) -> dict[str, object]:
    validation_metrics = summary.get("validation_metrics")
    if not isinstance(validation_metrics, dict):
        validation_metrics = summary.get("policy", {})
    test_metrics = summary.get("test_metrics")
    if not isinstance(test_metrics, dict):
        test_metrics = {}
    return {
        "summary_path": str(summary_path),
        "run_dir": str(summary_path.parent),
        "model_family": str(summary.get("model_family", summary_path.parent.parent.name)),
        "validation_log_loss": summary.get("validation_log_loss_calibrated"),
        "test_log_loss": summary.get("test_log_loss_calibrated"),
        "validation_net_pnl": validation_metrics.get("net_pnl_dollars"),
        "validation_trades": validation_metrics.get("trades"),
        "validation_objective": validation_metrics.get("objective"),
        "test_net_pnl": test_metrics.get("net_pnl_dollars"),
        "test_trades": test_metrics.get("trades"),
        "test_objective": test_metrics.get("objective"),
        "walk_forward_completed": summary.get("walk_forward_completed"),
    }


def _sort_key(row: dict[str, object]) -> tuple[float, float]:
    test_pnl = row.get("test_net_pnl")
    test_log_loss = row.get("test_log_loss")
    pnl_value = float(test_pnl) if isinstance(test_pnl, (float, int)) else float("-inf")
    log_loss_value = float(test_log_loss) if isinstance(test_log_loss, (float, int)) else float("inf")
    return (pnl_value, -log_loss_value)


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare KXBTC15M model summary artifacts.")
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run directory containing summary.json. Pass this flag multiple times to compare several runs.",
    )
    args = parser.parse_args()

    rows = []
    for run_dir in args.run_dir:
        summary_path, summary = _load_summary(run_dir)
        rows.append(_extract_metrics(summary_path, summary))

    rows.sort(key=_sort_key, reverse=True)
    columns = (
        "model_family",
        "validation_log_loss",
        "test_log_loss",
        "validation_net_pnl",
        "test_net_pnl",
        "validation_trades",
        "test_trades",
        "test_objective",
        "walk_forward_completed",
        "run_dir",
    )

    header = " | ".join(columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(_format_value(row[column]) for column in columns))

    print()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
