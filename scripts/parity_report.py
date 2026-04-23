from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.kalshi.parity_analysis import (
    DEFAULT_ECONOMIC_FLOOR_PER_TRADE,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_SNAPSHOT_LAG_MS,
    DEFAULT_REPORT_TIMEZONE,
    DEFAULT_ROLLING_WINDOW_DAYS,
    build_parity_artifacts,
    render_parity_report_markdown,
    write_parity_artifacts,
)

DEFAULT_LIVE_ROOT = REPO_ROOT / "output" / "live"
DEFAULT_RESEARCH_ROOT = REPO_ROOT / "output" / "live_research"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "kalshi" / "parity_reports"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the nightly Kalshi parity report.")
    parser.add_argument("--live-root", type=Path, default=DEFAULT_LIVE_ROOT)
    parser.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--as-of-date", type=date.fromisoformat, required=True)
    parser.add_argument("--report-timezone", default=DEFAULT_REPORT_TIMEZONE)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--rolling-window-days", type=int, default=DEFAULT_ROLLING_WINDOW_DAYS)
    parser.add_argument("--models", nargs="*", default=())
    parser.add_argument("--live-run", type=Path, action="append", default=[])
    parser.add_argument("--research-run", type=Path, action="append", default=[])
    parser.add_argument("--shadow-run", type=Path, action="append", default=[])
    parser.add_argument("--max-snapshot-lag-ms", type=int, default=DEFAULT_MAX_SNAPSHOT_LAG_MS)
    parser.add_argument("--economic-floor-per-trade", type=float, default=DEFAULT_ECONOMIC_FLOOR_PER_TRADE)
    parser.add_argument("--allow-replayed-in-verdict", action="store_true")
    parser.add_argument("--fail-on-low-coverage", action="store_true")
    parser.add_argument("--fail-on-parity-breach", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifacts = build_parity_artifacts(
        live_root=args.live_root,
        research_root=args.research_root,
        as_of_date=args.as_of_date,
        report_timezone=args.report_timezone,
        lookback_days=args.lookback_days,
        rolling_window_days=args.rolling_window_days,
        models=tuple(args.models),
        live_runs=tuple(path.resolve() for path in args.live_run),
        research_runs=tuple(path.resolve() for path in args.research_run),
        shadow_runs=tuple(path.resolve() for path in args.shadow_run),
        max_snapshot_lag_ms=args.max_snapshot_lag_ms,
        economic_floor_per_trade=args.economic_floor_per_trade,
        allow_replayed_in_verdict=args.allow_replayed_in_verdict,
    )
    markdown_text = render_parity_report_markdown(artifacts)
    output_dir = write_parity_artifacts(
        artifacts,
        output_root=args.output_root,
        as_of_date=args.as_of_date,
        markdown_text=markdown_text,
        economic_floor_per_trade=args.economic_floor_per_trade,
    )

    summary = artifacts.summary
    latest_14d = summary.get("latest_14d", {})
    latest_rolling = summary.get("latest_rolling", {})
    print(f"Parity report written to {output_dir}")
    print(
        "14d research/shadow/live net PnL:",
        f"{latest_14d.get('research_theoretical_pnl_dollars', 0.0):.2f} / "
        f"{latest_14d.get('shadow_simulated_pnl_dollars', 0.0):.2f} / "
        f"{latest_14d.get('live_realized_pnl_dollars', 0.0):.2f}",
    )
    print(
        "14d shadow-research gap:",
        f"total={latest_14d.get('shadow_minus_research_total_dollars')}, "
        f"mean={latest_14d.get('shadow_minus_research_mean_per_trade_dollars')}",
    )
    print(
        "14d live-shadow gap:",
        f"total={latest_14d.get('live_minus_shadow_total_dollars')}, "
        f"mean={latest_14d.get('live_minus_shadow_mean_per_trade_dollars')}",
    )
    print(
        "Rolling leak label:",
        latest_rolling.get("leak_label", "none"),
        "| parity breach:",
        bool(summary.get("parity_breach", False)),
    )

    if args.fail_on_low_coverage and bool(summary.get("low_coverage_warning")):
        return 2
    if args.fail_on_parity_breach and bool(summary.get("parity_breach")):
        return 3
    if not output_dir.exists():
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
