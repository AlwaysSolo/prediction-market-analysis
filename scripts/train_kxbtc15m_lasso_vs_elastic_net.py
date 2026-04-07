from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _timestamp_run_name() -> str:
    return datetime.now(UTC).strftime("kxbtc15m_compare_%Y%m%dT%H%M%SZ")


def _append_optional_arg(command: list[str], flag: str, value: str | None) -> None:
    if value:
        command.extend([flag, value])


def _append_optional_bool(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def _run_command(command: list[str]) -> None:
    print()
    print("Running:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed LASSO (l1_ratio=1.0) and fixed Elastic Net (l1_ratio=0.5) "
            "KXBTC15M pipelines back-to-back, then compare their summary artifacts."
        )
    )
    parser.add_argument("--series", default="KXBTC15M")
    parser.add_argument("--markets-path")
    parser.add_argument("--trades-path")
    parser.add_argument("--dataset-cache-dir")
    parser.add_argument("--reference-run-dir")
    parser.add_argument("--base-run-name", help="Base run name prefix. Defaults to a UTC timestamp.")
    parser.add_argument("--lasso-artifacts-root", help="Optional artifacts root override for the LASSO run.")
    parser.add_argument("--elastic-net-artifacts-root", help="Optional artifacts root override for the Elastic Net run.")
    parser.add_argument("--lasso-sample-size", type=int)
    parser.add_argument("--lasso-max-iter", type=int)
    parser.add_argument("--lasso-tol", type=float)
    parser.add_argument("--elastic-net-sample-size", type=int)
    parser.add_argument("--elastic-net-max-iter", type=int)
    parser.add_argument("--elastic-net-tol", type=float)
    parser.add_argument("--skip-walk-forward", action="store_true")
    parser.add_argument("--skip-latest-publish", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    base_run_name = args.base_run_name or _timestamp_run_name()
    lasso_run_name = f"{base_run_name}_lasso_l1r100"
    elastic_net_run_name = f"{base_run_name}_elastic_net_l1r050"

    common_args = ["--series", args.series]
    for flag, value in (
        ("--markets-path", args.markets_path),
        ("--trades-path", args.trades_path),
        ("--dataset-cache-dir", args.dataset_cache_dir),
        ("--reference-run-dir", args.reference_run_dir),
    ):
        if value:
            common_args.extend([flag, value])

    common_flags = []
    for flag, enabled in (
        ("--skip-walk-forward", args.skip_walk_forward),
        ("--skip-latest-publish", args.skip_latest_publish),
        ("--no-progress", args.no_progress),
    ):
        if enabled:
            common_flags.append(flag)

    lasso_command = [sys.executable, str(REPO_ROOT / "scripts" / "train_kxbtc15m_lasso.py"), *common_args]
    _append_optional_arg(lasso_command, "--artifacts-root", args.lasso_artifacts_root)
    _append_optional_arg(lasso_command, "--run-name", lasso_run_name)
    if args.lasso_sample_size is not None:
        lasso_command.extend(["--lasso-sample-size", str(args.lasso_sample_size)])
    if args.lasso_max_iter is not None:
        lasso_command.extend(["--lasso-max-iter", str(args.lasso_max_iter)])
    if args.lasso_tol is not None:
        lasso_command.extend(["--lasso-tol", str(args.lasso_tol)])
    lasso_command.extend(common_flags)

    elastic_net_command = [sys.executable, str(REPO_ROOT / "scripts" / "train_kxbtc15m_elastic_net.py"), *common_args]
    _append_optional_arg(elastic_net_command, "--artifacts-root", args.elastic_net_artifacts_root)
    _append_optional_arg(elastic_net_command, "--run-name", elastic_net_run_name)
    elastic_net_command.extend(["--elastic-net-l1-ratios", "0.5"])
    if args.elastic_net_sample_size is not None:
        elastic_net_command.extend(["--elastic-net-sample-size", str(args.elastic_net_sample_size)])
    if args.elastic_net_max_iter is not None:
        elastic_net_command.extend(["--elastic-net-max-iter", str(args.elastic_net_max_iter)])
    if args.elastic_net_tol is not None:
        elastic_net_command.extend(["--elastic-net-tol", str(args.elastic_net_tol)])
    elastic_net_command.extend(common_flags)

    _run_command(lasso_command)
    _run_command(elastic_net_command)

    lasso_run_dir = (
        Path(args.lasso_artifacts_root).expanduser() / lasso_run_name
        if args.lasso_artifacts_root
        else REPO_ROOT / "artifacts" / "kalshi" / "kxbtc15m_lasso" / lasso_run_name
    )
    elastic_net_run_dir = (
        Path(args.elastic_net_artifacts_root).expanduser() / elastic_net_run_name
        if args.elastic_net_artifacts_root
        else REPO_ROOT / "artifacts" / "kalshi" / "kxbtc15m_elastic_net" / elastic_net_run_name
    )

    compare_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "compare_kxbtc15m_models.py"),
        "--run-dir",
        str(lasso_run_dir),
        "--run-dir",
        str(elastic_net_run_dir),
    ]
    _run_command(compare_command)


if __name__ == "__main__":
    main()
