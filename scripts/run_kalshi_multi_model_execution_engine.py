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
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiExecutionConfig,
    KalshiExecutionEngine,
    KalshiExecutionMode,
    KalshiFeatureEngineConfig,
    KalshiFeatureStateEngine,
    KalshiLinearSVMScorer,
    KalshiLinearSVMScorerConfig,
    KalshiMarketDataCollector,
    KalshiPathDependentBinaryLayeringEngine,
    KalshiRegularizedLogisticScorer,
    KalshiRegularizedLogisticScorerConfig,
    KalshiSignalRiskEngine,
)
from scripts.run_kalshi_regularized_execution_engine import (  # noqa: E402
    apply_signal_config_overrides,
    default_model_file_for_family,
    resolve_policy_file,
    signal_config_from_env_and_policy,
)


@dataclass(frozen=True)
class ModelRuntime:
    label: str
    family: str
    model_file: Path
    policy_file: Path | None
    scorer: object
    signal_engine: KalshiSignalRiskEngine
    execution_engine: KalshiExecutionEngine
    layering_engine: KalshiPathDependentBinaryLayeringEngine | None
    signal_queue: asyncio.Queue
    execution_queue: asyncio.Queue
    layering_queue: asyncio.Queue | None


@dataclass(frozen=True)
class RuntimeSpec:
    label: str
    family: str
    run_dir: Path


def _parse_family_path_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f'Expected FAMILY=PATH override, got "{value}"'
            )
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
            raise argparse.ArgumentTypeError(
                f'Expected LABEL=FAMILY=PATH runtime override, got "{value}"'
            )
        label, family, raw_path = (part.strip() for part in parts)
        if not label:
            raise argparse.ArgumentTypeError(f'Invalid empty runtime label in "{value}"')
        if family not in {"lasso", "elastic_net", "bagged_lasso", "linear_svm"}:
            raise argparse.ArgumentTypeError(f'Unsupported runtime family "{family}" in "{value}"')
        specs.append(RuntimeSpec(label=label, family=family, run_dir=Path(raw_path).expanduser()))
    return specs


def _resolve_model_and_policy_files(
    family: str,
    *,
    run_dir_overrides: dict[str, Path],
    model_file_overrides: dict[str, Path],
    policy_file_overrides: dict[str, Path],
) -> tuple[Path, Path | None]:
    override_run_dir = run_dir_overrides.get(family)
    if override_run_dir is not None:
        if family == "linear_svm":
            model_file = override_run_dir / "linear_svm" / "model.joblib"
        else:
            model_file = override_run_dir / family / "model.joblib"
        policy_file = override_run_dir / "policy.json"
        return model_file, (policy_file if policy_file.exists() else None)

    model_file = model_file_overrides.get(family, default_model_file_for_family(family))
    policy_file = resolve_policy_file(
        str(policy_file_overrides[family]) if family in policy_file_overrides else None,
        model_file,
    )
    return model_file, policy_file


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
        or feature_name.startswith("k15_k1h_")
        for feature_name in feature_names
    )


async def _consume_signal_updates(model_family: str, queue: asyncio.Queue) -> None:
    while True:
        update = await queue.get()
        print(
            f"[{model_family}] SIGNAL",
            update.ticker,
            f"tau={update.tau_minutes:.2f}",
            f"side={update.side}",
            f"raw_edge={update.raw_model_edge:+.6f}" if update.raw_model_edge is not None else "raw_edge=?",
            (
                f"post_cost_edge={update.post_cost_edge:+.6f}"
                if update.post_cost_edge is not None
                else "post_cost_edge=?"
            ),
            (
                f"yes_edge={update.yes_post_cost_edge:+.6f}"
                if update.yes_post_cost_edge is not None
                else "yes_edge=?"
            ),
            (
                f"no_edge={update.no_post_cost_edge:+.6f}"
                if update.no_post_cost_edge is not None
                else "no_edge=?"
            ),
            (
                f"bid_ask={update.yes_bid_cents}/{update.yes_ask_cents}"
                if update.yes_bid_cents is not None and update.yes_ask_cents is not None
                else "bid_ask=?"
            ),
            f"price={update.reference_price_cents}c" if update.reference_price_cents is not None else "price=?",
            (
                f"max_price={update.max_acceptable_entry_price_cents}c"
                if update.max_acceptable_entry_price_cents is not None
                else "max_price=?"
            ),
            "APPROVED" if update.approved else f"BLOCKED:{update.block_reason}",
        )


async def _consume_execution_updates(
    model_family: str,
    queue: asyncio.Queue,
    execution_engine: KalshiExecutionEngine,
) -> None:
    while True:
        update = await queue.get()
        portfolio_snapshot = execution_engine.get_portfolio_snapshot()
        portfolio_cash = (
            f"${portfolio_snapshot.available_cash_dollars:.2f}"
            if portfolio_snapshot is not None
            else (f"${update.available_cash_dollars:.2f}" if update.available_cash_dollars is not None else "n/a")
        )


async def _consume_layering_updates(model_family: str, queue: asyncio.Queue) -> None:
    while True:
        update = await queue.get()
        print(
            f"[{model_family}] LAYERING",
            update.ticker,
            f"action={update.action}",
            f"status={update.status}",
            f"window={update.decision_window}",
            f"side={update.side}",
            f"tranche={update.tranche_index}",
            f"contracts={update.contracts}",
            (
                f"ev=${update.expected_value_dollars:+.2f}"
                if update.expected_value_dollars is not None
                else "ev=?"
            ),
            (
                f"worst=${update.worst_case_loss_dollars:.2f}"
                if update.worst_case_loss_dollars is not None
                else "worst=?"
            ),
            f"msg={update.message}" if update.message else "",
        )
        print(
            f"[{model_family}] EXECUTION",
            update.ticker,
            f"decision={update.decision_id}",
            f"side={update.side}",
            f"contracts={update.contracts}",
            f"limit={update.limit_price_cents}c",
            f"status={update.status}",
            f"filled={update.filled_contracts}/{update.contracts}",
            f"fill_price={update.fill_price_cents}c" if update.fill_price_cents is not None else "fill_price=?",
            f"cash={portfolio_cash}",
            (
                f"pnl=${update.realized_pnl_dollars:+.2f}"
                if update.realized_pnl_dollars is not None
                else ""
            ),
            (
                f"cum_pnl=${update.cumulative_realized_pnl_dollars:+.2f}"
                if update.cumulative_realized_pnl_dollars is not None
                else ""
            ),
            (
                f"result={update.settlement_result}"
                if update.settlement_result is not None
                else ""
            ),
            f"msg={update.message}" if update.message else "",
        )


async def _run(args: argparse.Namespace) -> None:
    environment = KalshiEnvironment(args.environment)
    log_root = Path(args.log_root)
    run_dir_overrides = _parse_family_path_overrides(args.run_dir)
    model_file_overrides = _parse_family_path_overrides(args.model_file)
    policy_file_overrides = _parse_family_path_overrides(args.policy_file)
    runtime_specs = _parse_runtime_specs(args.runtime_run_dir)
    layering_enabled = bool(args.enable_layering)

    if layering_enabled:
        if any(series != "KXBTC15M" for series in args.series):
            raise RuntimeError("Path-dependent layering v1 currently supports only the KXBTC15M series.")
        if any(not ticker.startswith("KXBTC15M") for ticker in args.ticker):
            raise RuntimeError("Path-dependent layering v1 currently supports only KXBTC15M tickers.")

    resolved_runtime_specs: list[tuple[str, str, Path, Path | None]] = []
    for family in args.model_family:
        model_file, policy_file = _resolve_model_and_policy_files(
            family,
            run_dir_overrides=run_dir_overrides,
            model_file_overrides=model_file_overrides,
            policy_file_overrides=policy_file_overrides,
        )
        resolved_runtime_specs.append((family, family, model_file, policy_file))
    for spec in runtime_specs:
        model_file, policy_file = _resolve_model_and_policy_files(
            spec.family,
            run_dir_overrides={spec.family: spec.run_dir},
            model_file_overrides={},
            policy_file_overrides={},
        )
        resolved_runtime_specs.append((spec.label, spec.family, model_file, policy_file))

    requires_hourly_context = any(_runtime_requires_hourly_context(model_file) for _label, _family, model_file, _policy_file in resolved_runtime_specs)
    collector_series = tuple(dict.fromkeys([*args.series, *( [args.hourly_context_series] if requires_hourly_context and args.hourly_context_series else [] )]))

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

    model_runtimes: list[ModelRuntime] = []
    for label, family, model_file, policy_file in resolved_runtime_specs:
        signal_config, loaded_policy_file = signal_config_from_env_and_policy(environment, policy_file)
        signal_config = apply_signal_config_overrides(signal_config, args)
        if layering_enabled:
            signal_config = replace(signal_config, auto_reserve_trade_intents=False)

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

        signal_engine = KalshiSignalRiskEngine(
            scorer,
            signal_config,
            log_dir=log_root / "signal" / label,
        )
        execution_config = replace(
            KalshiExecutionConfig.from_env(environment),
            log_dir=log_root / "execution" / label,
        )
        if args.execution_mode is not None:
            execution_config = replace(execution_config, mode=KalshiExecutionMode(args.execution_mode))
        if execution_config.mode is not KalshiExecutionMode.PAPER:
            raise RuntimeError(
                "Current multi-model regularized artifacts are legacy LTP-trained models and are restricted to paper "
                "mode until quote-aware retraining and offline evaluation are complete."
            )
        layering_engine = None
        if layering_enabled:
            layering_engine = KalshiPathDependentBinaryLayeringEngine(
                signal_engine,
                log_dir=log_root / "layering" / label,
            )
        execution_engine = KalshiExecutionEngine(
            signal_engine,
            execution_config,
            trade_intent_source=layering_engine,
        )
        if layering_engine is not None:
            layering_engine.bind_execution_engine(execution_engine)
        signal_queue = signal_engine.subscribe_queue()
        execution_queue = execution_engine.subscribe_queue()
        layering_queue = None if layering_engine is None else layering_engine.subscribe_queue()

        print(f"Configured runtime: {label}")
        print(f"  family: {family}")
        print(f"  model file: {model_file}")
        if loaded_policy_file is not None:
            print(f"  policy file: {loaded_policy_file}")
        else:
            print("  policy file: env/default signal config")
        print(f"  signal log dir: {log_root / 'signal' / label}")
        print(f"  execution log dir: {log_root / 'execution' / label}")
        if layering_engine is not None:
            print(f"  layering log dir: {log_root / 'layering' / label}")

        model_runtimes.append(
            ModelRuntime(
                label=label,
                family=family,
                model_file=model_file,
                policy_file=loaded_policy_file,
                scorer=scorer,
                signal_engine=signal_engine,
                execution_engine=execution_engine,
                layering_engine=layering_engine,
                signal_queue=signal_queue,
                execution_queue=execution_queue,
                layering_queue=layering_queue,
            )
        )

    print("Starting shared collector and feature engine...")
    await collector.start()
    print("Collector ready")
    await feature_engine.start()
    print("Feature engine ready")
    if requires_hourly_context:
        print(f"Hourly context enabled: target={args.series[0]} context={args.hourly_context_series}")

    print("Starting model stacks...")
    for runtime in model_runtimes:
        await runtime.scorer.start()
        await runtime.signal_engine.start()
        await runtime.execution_engine.start()
        if runtime.layering_engine is not None:
            await runtime.layering_engine.start()
        print(
            f"[{runtime.label}] started",
            f"family={runtime.family}",
            f"mode={runtime.execution_engine.config.mode.value}",
            f"simulate_fills={runtime.execution_engine.config.simulate_immediate_fills}",
            f"edge={runtime.signal_engine.config.edge_threshold_cents:.1f}c",
            f"tau={runtime.signal_engine.config.min_tau_minutes:.1f}-{runtime.signal_engine.config.max_tau_minutes:.1f}",
            f"price_band={runtime.signal_engine.config.price_band_min_cents}-{runtime.signal_engine.config.price_band_max_cents}c",
            f"quote_max_age={runtime.signal_engine.config.quote_max_age_seconds:.1f}s",
            f"contracts={runtime.signal_engine.config.contracts_per_order}",
            f"capital_pct={runtime.signal_engine.config.capital_pct_per_order}",
            f"kelly_mult={runtime.signal_engine.config.kelly_fraction_multiplier}",
            f"kelly_cap={runtime.signal_engine.config.kelly_fraction_cap_pct}",
            f"stacking={runtime.signal_engine.config.allow_stacking}",
            f"layering={runtime.layering_engine is not None}",
        )

    consumer_tasks: list[asyncio.Task] = []
    for runtime in model_runtimes:
        consumer_tasks.append(
            asyncio.create_task(
                _consume_signal_updates(runtime.label, runtime.signal_queue),
                name=f"{runtime.label}-signal-printer",
            )
        )
        consumer_tasks.append(
            asyncio.create_task(
                _consume_execution_updates(runtime.label, runtime.execution_queue, runtime.execution_engine),
                name=f"{runtime.label}-execution-printer",
            )
        )
        if runtime.layering_queue is not None:
            consumer_tasks.append(
                asyncio.create_task(
                    _consume_layering_updates(runtime.label, runtime.layering_queue),
                    name=f"{runtime.label}-layering-printer",
                )
            )

    print("Shared multi-model paper stack started.")
    print("Dashboard root:", log_root)
    print("Open: http://localhost:8765/tools/live_trading_dashboard.html")
    print("Choose folder:", log_root)

    try:
        await asyncio.Event().wait()
    finally:
        for task in consumer_tasks:
            task.cancel()
        for task in consumer_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        for runtime in reversed(model_runtimes):
            if runtime.layering_engine is not None:
                await runtime.layering_engine.stop()
            await runtime.execution_engine.stop()
            await runtime.signal_engine.stop()
            await runtime.scorer.stop()
        await feature_engine.stop()
        await collector.stop()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run multiple Kalshi paper-trading model stacks on one shared websocket stream."
    )
    parser.add_argument("--environment", choices=["demo", "production"], default="demo")
    parser.add_argument("--execution-mode", choices=["paper", "live"], default="paper")
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
        help="Optional exact run dir override per family. Example: lasso=artifacts\\kalshi\\kxbtc15m_lasso\\my_run",
    )
    parser.add_argument(
        "--runtime-run-dir",
        action="append",
        default=[],
        metavar="LABEL=FAMILY=PATH",
        help="Add a named runtime from an exact run dir. Use this to run multiple artifacts from the same family side by side.",
    )
    parser.add_argument(
        "--model-file",
        action="append",
        default=[],
        metavar="FAMILY=PATH",
        help="Optional explicit model file override per family.",
    )
    parser.add_argument(
        "--policy-file",
        action="append",
        default=[],
        metavar="FAMILY=PATH",
        help="Optional explicit policy file override per family.",
    )
    parser.add_argument("--contracts-per-order", type=int, default=None)
    parser.add_argument("--capital-pct-per-order", type=float, default=None)
    parser.add_argument("--kelly-fraction-multiplier", type=float, default=None)
    parser.add_argument("--kelly-fraction-cap-pct", type=float, default=None)
    parser.add_argument("--series", action="append", default=["KXBTC15M"], help="Series ticker prefix filter")
    parser.add_argument("--hourly-context-series", default="KXBTCD", help="Context series to subscribe to when a runtime expects hourly context features.")
    parser.add_argument("--ticker", action="append", default=[], help="Explicit market ticker filter")
    parser.add_argument("--log-root", default="output/live/kalshi", help="Root folder for raw, signal, and execution logs")
    parser.add_argument("--metadata-refresh-interval-seconds", type=float, default=300.0)
    parser.add_argument(
        "--enable-layering",
        action="store_true",
        help="Enable the path-dependent binary layering engine for KXBTC15M runtimes.",
    )
    args = parser.parse_args()
    if not args.model_family and not args.runtime_run_dir:
        args.model_family = ["lasso", "elastic_net", "bagged_lasso", "linear_svm"]
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
