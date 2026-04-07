from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.kalshi.performance_analysis import generate_kalshi_performance_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Kalshi offline artifacts, live execution runs, or live research runs."
    )
    parser.add_argument("input_path", help="Run directory, root directory, or known repo-native performance file.")
    parser.add_argument(
        "--mode",
        choices=("auto", "offline_artifacts", "live_execution", "live_research"),
        default="auto",
        help="Force a source type instead of using auto-detection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to artifacts/kalshi/performance_reports/<detected_name>/",
    )
    parser.add_argument(
        "--environment",
        default=None,
        help="Optional environment filter for live/live_research runs, for example demo or prod.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Optional model filter. Repeat to include multiple models.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Optional rolling lookback window for canonical rows.",
    )
    parser.add_argument(
        "--min-combo-count",
        type=int,
        default=5,
        help="Minimum settled rows required before a combo bucket is included.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    outputs = generate_kalshi_performance_reports(
        args.input_path,
        mode=args.mode,
        output_dir=args.output_dir,
        environment=args.environment,
        model_filters=tuple(args.model),
        lookback_days=args.lookback_days,
        min_combo_count=args.min_combo_count,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
