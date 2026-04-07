from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.kalshi import (  # noqa: E402
    DEFAULT_BAGGED_LASSO_MODEL_FILE,
    DEFAULT_ELASTIC_NET_MODEL_FILE,
    DEFAULT_LASSO_MODEL_FILE,
    DEFAULT_LINEAR_SVM_MODEL_FILE,
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiFeatureEngineConfig,
    KalshiFeatureStateEngine,
    KalshiLinearSVMScorer,
    KalshiLinearSVMScorerConfig,
    KalshiMarketDataCollector,
    KalshiRegularizedLogisticScorer,
    KalshiRegularizedLogisticScorerConfig,
    KalshiResearchLedger,
    KalshiResearchSampleUpdate,
    KalshiResearchSampler,
    KalshiResearchSamplerConfig,
    KalshiResearchSettlementUpdate,
)
from src.live.kalshi.research_archive import (  # noqa: E402
    KalshiResearchArchiveConfig,
    KalshiResearchArchiveManager,
    KalshiResearchArchiveRuntime,
)


@dataclass(frozen=True)
class ResearchRuntime:
    label: str
    family: str
    model_file: Path
    scorer: object
    sampler: KalshiResearchSampler
    ledger: KalshiResearchLedger
    sample_queue: asyncio.Queue
    settlement_queue: asyncio.Queue


@dataclass(frozen=True)
class RuntimeSpec:
    label: str
    family: str
    run_dir: Path


def _parse_family_path_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f'Expected FAMILY=PATH override, got "{value}"')
        family, raw_path = value.split("=", 1)
        family_name = family.strip()
        if family_name == "":
            raise argparse.ArgumentTypeError(f'Invalid empty family in override "{value}"')
        overrides[family_name] = Path(raw_path).expanduser()
    return overrides


def _parse_runtime_specs(values: list[str]) -> list[RuntimeSpec]:
    specs: list[RuntimeSpec] = []
    for value in values:
        parts = value.split("=", 2)
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(f'Expected LABEL=FAMILY=PATH runtime override, got "{value}"')
        label, family, raw_path = (part.strip() for part in parts)
        if not label:
            raise argparse.ArgumentTypeError(f'Invalid empty runtime label in "{value}"')
        if family not in {"lasso", "elastic_net", "bagged_lasso", "linear_svm"}:
            raise argparse.ArgumentTypeError(f'Unsupported runtime family "{family}" in "{value}"')
        specs.append(RuntimeSpec(label=label, family=family, run_dir=Path(raw_path).expanduser()))
    return specs


def _default_model_file_for_family(model_family: str) -> Path:
    if model_family == "elastic_net":
        return DEFAULT_ELASTIC_NET_MODEL_FILE
    if model_family == "bagged_lasso":
        return DEFAULT_BAGGED_LASSO_MODEL_FILE
    if model_family == "linear_svm":
        return DEFAULT_LINEAR_SVM_MODEL_FILE
    return DEFAULT_LASSO_MODEL_FILE


def _resolve_model_file(
    family: str,
    *,
    run_dir_overrides: dict[str, Path],
    model_file_overrides: dict[str, Path],
) -> Path:
    override_run_dir = run_dir_overrides.get(family)
    if override_run_dir is not None:
        if family == "linear_svm":
            return override_run_dir / "linear_svm" / "model.joblib"
        return override_run_dir / family / "model.joblib"
    return model_file_overrides.get(family, _default_model_file_for_family(family))


def _runtime_requires_hourly_context(model_file: Path) -> bool:
    metrics_file = model_file.with_name("metrics.json")
    if not metrics_file.exists():
        return False
    try:
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    feature_names = tuple(str(name) for name in metrics.get("feature_names", []))
    return any(
        feature_name.startswith("kxbtcd_atm_")
        or feature_name.startswith("k15_minus_k1h_")
        or feature_name == "k15_k1h_atm_direction_agreement"
        for feature_name in feature_names
    )


def _apply_research_config_overrides(
    config: KalshiResearchSamplerConfig,
    args: argparse.Namespace,
) -> KalshiResearchSamplerConfig:
    overrides: dict[str, float | int] = {}
    for key in (
        "min_edge_cents",
        "min_tau_minutes",
        "max_tau_minutes",
        "price_band_min_cents",
        "price_band_max_cents",
        "quote_max_age_seconds",
        "contracts_per_sample",
        "slippage_pct",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return replace(config, **overrides) if overrides else config


def _fmt_cents_field(name: str, value: int | None) -> str:
    return f"{name}={value}c" if value is not None else f"{name}=?"


def _fmt_quote_age_field(value: float | None) -> str:
    return f"quote_age={value:.2f}s" if value is not None else "quote_age=?"


async def _consume_research_updates(label: str, queue: asyncio.Queue) -> None:
    while True:
        update: KalshiResearchSampleUpdate = await queue.get()
        parts = [
            f"[{label}] RESEARCH",
            update.ticker,
            f"status={update.status}",
            f"reason={update.reason}" if update.reason else "",
            f"side={update.side}" if update.side else "",
            f"tau={update.tau_minutes:.2f}",
            (
                f"chosen_edge={update.chosen_post_cost_edge:+.6f}"
                if update.chosen_post_cost_edge is not None
                else "chosen_edge=?"
            ),
            f"price={update.reference_price_cents}c" if update.reference_price_cents is not None else "price=?",
        ]
        if update.status == "skipped":
            parts.extend([
                _fmt_cents_field("yes_bid", update.yes_bid_cents),
                _fmt_cents_field("yes_ask", update.yes_ask_cents),
                _fmt_cents_field("buy_yes", update.buy_yes_price_cents),
                _fmt_cents_field("buy_no", update.buy_no_price_cents),
                _fmt_quote_age_field(update.quote_age_seconds),
            ])
        print(*parts)


async def _consume_settlement_updates(label: str, queue: asyncio.Queue) -> None:
    while True:
        update: KalshiResearchSettlementUpdate = await queue.get()
        print(
            f"[{label}] SETTLED",
            update.ticker,
            f"side={update.side}",
            f"result={update.settlement_result}",
            f"win={update.is_win}",
            f"pnl=${update.realized_pnl_dollars:+.2f}",
            f"cum_pnl=${update.cumulative_realized_pnl_dollars:+.2f}",
        )


async def _run(args: argparse.Namespace) -> None:
    environment = KalshiEnvironment(args.environment)
    log_root = Path(args.log_root)
    run_dir_overrides = _parse_family_path_overrides(args.run_dir)
    model_file_overrides = _parse_family_path_overrides(args.model_file)
    runtime_specs = _parse_runtime_specs(args.runtime_run_dir)

    resolved_runtime_specs: list[tuple[str, str, Path]] = []
    for family in args.model_family:
        resolved_runtime_specs.append(
            (family, family, _resolve_model_file(family, run_dir_overrides=run_dir_overrides, model_file_overrides=model_file_overrides))
        )
    for spec in runtime_specs:
        resolved_runtime_specs.append(
            (spec.label, spec.family, _resolve_model_file(spec.family, run_dir_overrides={spec.family: spec.run_dir}, model_file_overrides={}))
        )

    requires_hourly_context = any(_runtime_requires_hourly_context(model_file) for _label, _family, model_file in resolved_runtime_specs)
    collector_series = tuple(
        dict.fromkeys([*args.series, *([args.hourly_context_series] if requires_hourly_context and args.hourly_context_series else [])])
    )
    research_config = _apply_research_config_overrides(KalshiResearchSamplerConfig.from_env(environment), args)

    credentials = KalshiCredentials.from_env(environment)
    collector_config = KalshiCollectorConfig(
        environment=environment,
        credentials=credentials,
        log_dir=log_root / "raw",
        series_tickers=collector_series,
        market_tickers=tuple(args.ticker),
        metadata_refresh_interval_seconds=args.metadata_refresh_interval_seconds,
    )
    collector = KalshiMarketDataCollector(collector_config)
    feature_engine = KalshiFeatureStateEngine(
        collector,
        config=KalshiFeatureEngineConfig(
            publish_series_tickers=tuple(args.series),
            hourly_context_target_series_ticker=args.series[0] if requires_hourly_context and args.series else None,
            hourly_context_series_ticker=args.hourly_context_series if requires_hourly_context else None,
        ),
    )

    runtimes: list[ResearchRuntime] = []
    for label, family, model_file in resolved_runtime_specs:
        if family == "linear_svm":
            scorer = KalshiLinearSVMScorer(
                feature_engine,
                KalshiLinearSVMScorerConfig(model_file=model_file),
            )
        else:
            scorer = KalshiRegularizedLogisticScorer(
                feature_engine,
                KalshiRegularizedLogisticScorerConfig(model_file=model_file),
            )
        research_log_dir = log_root / "research" / label
        sampler = KalshiResearchSampler(
            scorer,
            research_config,
            log_dir=research_log_dir,
        )
        ledger = KalshiResearchLedger(
            collector,
            sampler,
            log_dir=research_log_dir,
        )
        sample_queue = sampler.subscribe_queue()
        settlement_queue = ledger.subscribe_queue()
        runtimes.append(
            ResearchRuntime(
                label=label,
                family=family,
                model_file=model_file,
                scorer=scorer,
                sampler=sampler,
                ledger=ledger,
                sample_queue=sample_queue,
                settlement_queue=settlement_queue,
            )
        )
        print(f"Configured research runtime: {label}")
        print(f"  family: {family}")
        print(f"  model file: {model_file}")
        print(f"  research log dir: {research_log_dir}")

    archive_manager: KalshiResearchArchiveManager | None = None
    archive_mode = getattr(args, "archive", "off")
    if archive_mode == "full":
        archive_root_arg = getattr(args, "archive_root", None)
        archive_root = Path(archive_root_arg) if archive_root_arg else (log_root / "archive")
        archive_manager = KalshiResearchArchiveManager(
            collector,
            feature_engine,
            [
                KalshiResearchArchiveRuntime(
                    label=runtime.label,
                    family=runtime.family,
                    model_file=runtime.model_file,
                    scorer=runtime.scorer,
                    sampler=runtime.sampler,
                    ledger=runtime.ledger,
                    calibration_enabled=bool(getattr(runtime.scorer.config, "apply_calibration", False)),
                )
                for runtime in runtimes
            ],
            KalshiResearchArchiveConfig(
                archive_root=archive_root,
                run_name=log_root.name,
                environment=environment.value,
                compact_on_shutdown=getattr(args, "archive_compact_on_shutdown", True),
            ),
        )
        await archive_manager.start()
        print(f"Archive enabled: {archive_root}")

    print("Starting shared collector and feature engine...")
    await collector.start()
    print("Collector ready")
    await feature_engine.start()
    print("Feature engine ready")
    if requires_hourly_context:
        print(f"Hourly context enabled: target={args.series[0]} context={args.hourly_context_series}")

    print("Starting research stacks...")
    for runtime in runtimes:
        await runtime.scorer.start()
        await runtime.ledger.start()
        await runtime.sampler.start()
        print(
            f"[{runtime.label}] started",
            f"family={runtime.family}",
            f"edge={research_config.min_edge_cents:.1f}c",
            f"tau={research_config.min_tau_minutes:.1f}-{research_config.max_tau_minutes:.1f}",
            f"price_band={research_config.price_band_min_cents}-{research_config.price_band_max_cents}c",
            f"quote_max_age={research_config.quote_max_age_seconds:.1f}s",
            f"contracts={research_config.contracts_per_sample}",
        )

    consumer_tasks: list[asyncio.Task] = []
    for runtime in runtimes:
        consumer_tasks.append(
            asyncio.create_task(
                _consume_research_updates(runtime.label, runtime.sample_queue),
                name=f"{runtime.label}-research-printer",
            )
        )
        consumer_tasks.append(
            asyncio.create_task(
                _consume_settlement_updates(runtime.label, runtime.settlement_queue),
                name=f"{runtime.label}-settlement-printer",
            )
        )

    print("Shared multi-model research stack started.")
    print("Research root:", log_root)

    try:
        if args.max_runtime_seconds is None:
            await asyncio.Event().wait()
        else:
            await asyncio.sleep(args.max_runtime_seconds)
    finally:
        for task in consumer_tasks:
            task.cancel()
        for task in consumer_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        for runtime in reversed(runtimes):
            await runtime.sampler.stop()
            await runtime.ledger.stop()
            await runtime.scorer.stop()
        await feature_engine.stop()
        await collector.stop()
        if archive_manager is not None:
            await archive_manager.stop()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run multiple Kalshi research-only model stacks on one shared websocket stream."
    )
    parser.add_argument("--environment", choices=["demo", "production"], default="demo")
    parser.add_argument(
        "--model-family",
        action="append",
        choices=["lasso", "elastic_net", "bagged_lasso", "linear_svm"],
        default=[],
        help="Model families to run. Defaults to lasso, elastic_net, bagged_lasso, and linear_svm together.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        metavar="FAMILY=PATH",
        help="Optional exact run dir override per family.",
    )
    parser.add_argument(
        "--runtime-run-dir",
        action="append",
        default=[],
        metavar="LABEL=FAMILY=PATH",
        help="Add a named runtime from an exact run dir.",
    )
    parser.add_argument(
        "--model-file",
        action="append",
        default=[],
        metavar="FAMILY=PATH",
        help="Optional explicit model file override per family.",
    )
    parser.add_argument("--series", action="append", default=["KXBTC15M"], help="Series ticker prefix filter")
    parser.add_argument("--hourly-context-series", default="KXBTCD")
    parser.add_argument("--ticker", action="append", default=[], help="Explicit market ticker filter")
    parser.add_argument("--log-root", default="output/live_research/kalshi", help="Root folder for research logs")
    parser.add_argument("--metadata-refresh-interval-seconds", type=float, default=300.0)
    parser.add_argument("--min-edge-cents", type=float, default=None)
    parser.add_argument("--min-tau-minutes", type=float, default=None)
    parser.add_argument("--max-tau-minutes", type=float, default=None)
    parser.add_argument("--price-band-min-cents", type=int, default=None)
    parser.add_argument("--price-band-max-cents", type=int, default=None)
    parser.add_argument("--quote-max-age-seconds", type=float, default=None)
    parser.add_argument("--contracts-per-sample", type=int, default=None)
    parser.add_argument("--slippage-pct", type=float, default=None)
    parser.add_argument("--max-runtime-seconds", type=float, default=None)
    parser.add_argument("--archive", choices=["off", "full"], default="off")
    parser.add_argument("--archive-root", default=None)
    parser.add_argument(
        "--archive-compact-on-shutdown",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if not args.model_family and not args.runtime_run_dir:
        args.model_family = ["lasso", "elastic_net", "bagged_lasso", "linear_svm"]
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
