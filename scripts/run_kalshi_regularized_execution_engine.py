from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
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
    KalshiExecutionConfig,
    KalshiExecutionEngine,
    KalshiExecutionMode,
    KalshiFeatureStateEngine,
    KalshiLinearSVMScorer,
    KalshiLinearSVMScorerConfig,
    KalshiMarketDataCollector,
    KalshiRegularizedLogisticScorer,
    KalshiRegularizedLogisticScorerConfig,
    KalshiSignalRiskConfig,
    KalshiSignalRiskEngine,
)


def default_model_file_for_family(model_family: str) -> Path:
    if model_family == "elastic_net":
        return DEFAULT_ELASTIC_NET_MODEL_FILE
    if model_family == "bagged_lasso":
        return DEFAULT_BAGGED_LASSO_MODEL_FILE
    if model_family == "linear_svm":
        return DEFAULT_LINEAR_SVM_MODEL_FILE
    return DEFAULT_LASSO_MODEL_FILE


def resolve_policy_file(policy_file: str | None, model_file: Path) -> Path | None:
    if policy_file is not None:
        resolved = Path(policy_file)
        return resolved if resolved.exists() else None

    candidate = model_file.parents[1] / "policy.json"
    return candidate if candidate.exists() else None


def signal_config_from_env_and_policy(
    environment: KalshiEnvironment,
    policy_file: Path | None,
) -> tuple[KalshiSignalRiskConfig, Path | None]:
    config = KalshiSignalRiskConfig.from_env(environment)
    if policy_file is None:
        return config, None

    payload = json.loads(policy_file.read_text(encoding="utf-8"))
    policy_config = payload.get("config", payload)
    overrides = {
        key: value
        for key, value in policy_config.items()
        if key in KalshiSignalRiskConfig.__dataclass_fields__ and value is not None
    }
    if not overrides:
        return config, policy_file
    return replace(config, **overrides), policy_file


def apply_signal_config_overrides(
    config: KalshiSignalRiskConfig,
    args: argparse.Namespace,
) -> KalshiSignalRiskConfig:
    overrides = {}
    if getattr(args, "contracts_per_order", None) is not None:
        overrides["contracts_per_order"] = int(args.contracts_per_order)
    if getattr(args, "capital_pct_per_order", None) is not None:
        overrides["capital_pct_per_order"] = float(args.capital_pct_per_order)
    if getattr(args, "kelly_fraction_multiplier", None) is not None:
        overrides["kelly_fraction_multiplier"] = float(args.kelly_fraction_multiplier)
    if getattr(args, "kelly_fraction_cap_pct", None) is not None:
        overrides["kelly_fraction_cap_pct"] = float(args.kelly_fraction_cap_pct)
    return replace(config, **overrides) if overrides else config


async def _run(args: argparse.Namespace) -> None:
    environment = KalshiEnvironment(args.environment)
    model_file = Path(args.model_file) if args.model_file is not None else default_model_file_for_family(args.model_family)
    policy_file = resolve_policy_file(args.policy_file, model_file)
    signal_config, loaded_policy_file = signal_config_from_env_and_policy(environment, policy_file)
    signal_config = apply_signal_config_overrides(signal_config, args)

    print(f"Initializing regularized-model live stack for {environment.value}...")
    print(f"Model family: {args.model_family}")
    print(f"Model file: {model_file}")
    if loaded_policy_file is not None:
        print(f"Policy file: {loaded_policy_file}")
    else:
        print("Policy file: not found, using env/default signal config")

    credentials = KalshiCredentials.from_env(environment)
    collector_config = KalshiCollectorConfig(
        environment=environment,
        credentials=credentials,
        log_dir=Path(args.log_dir),
        series_tickers=tuple(args.series),
        market_tickers=tuple(args.ticker),
        metadata_refresh_interval_seconds=args.metadata_refresh_interval_seconds,
    )
    collector = KalshiMarketDataCollector(collector_config)
    feature_engine = KalshiFeatureStateEngine(collector)
    if args.model_family == "linear_svm":
        scorer = KalshiLinearSVMScorer(
            feature_engine,
            KalshiLinearSVMScorerConfig(model_file=model_file),
        )
    else:
        scorer = KalshiRegularizedLogisticScorer(
            feature_engine,
            KalshiRegularizedLogisticScorerConfig(model_file=model_file),
        )
    signal_engine = KalshiSignalRiskEngine(scorer, signal_config)
    execution_config = KalshiExecutionConfig.from_env(environment)
    if args.execution_mode is not None:
        execution_config = replace(execution_config, mode=KalshiExecutionMode(args.execution_mode))
    if execution_config.mode is not KalshiExecutionMode.PAPER:
        raise RuntimeError(
            "Current regularized-model artifacts are legacy LTP-trained models and are restricted to paper mode "
            "until quote-aware retraining and offline evaluation are complete."
        )
    execution_engine = KalshiExecutionEngine(signal_engine, execution_config)
    signal_queue = signal_engine.subscribe_queue()
    queue = execution_engine.subscribe_queue()

    print("Starting collector and loading market metadata...")
    await collector.start()
    print("Collector ready")
    await feature_engine.start()
    print("Feature engine ready")
    await scorer.start()
    print("Scorer ready")
    await signal_engine.start()
    print("Signal engine ready")
    await execution_engine.start()
    print("Execution engine ready")

    print(f"Full live stack started in {environment.value} mode")
    print(
        "Execution config:",
        f"mode={execution_engine.config.mode.value}",
        f"live_enabled={execution_engine.config.enable_live_trading}",
        f"simulate_fills={execution_engine.config.simulate_immediate_fills}",
        f"invert_signal={signal_engine.config.invert_model_signal}",
        f"subaccount={execution_engine.config.subaccount}",
        f"reconcile={execution_engine.config.reconcile_interval_seconds:.1f}s",
    )
    print(
        "Signal config:",
        f"edge={signal_engine.config.edge_threshold_cents:.1f}c",
        f"tau={signal_engine.config.min_tau_minutes:.1f}-{signal_engine.config.max_tau_minutes:.1f}",
        f"price_band={signal_engine.config.price_band_min_cents}-{signal_engine.config.price_band_max_cents}c",
        f"quote_max_age={signal_engine.config.quote_max_age_seconds:.1f}s",
        f"contracts={signal_engine.config.contracts_per_order}",
        f"capital_pct={signal_engine.config.capital_pct_per_order}",
        f"kelly_mult={signal_engine.config.kelly_fraction_multiplier}",
        f"kelly_cap={signal_engine.config.kelly_fraction_cap_pct}",
        f"stacking={signal_engine.config.allow_stacking}",
        f"reserve={signal_engine.config.reserve_cash_pct:.1f}%",
        f"slippage={signal_engine.config.slippage_pct:.1f}%",
    )

    try:
        while True:
            signal_task = asyncio.create_task(signal_queue.get())
            execution_task = asyncio.create_task(queue.get())
            done, pending = await asyncio.wait(
                {signal_task, execution_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

            for completed in done:
                update = completed.result()

                if completed is signal_task:
                    print(
                        "SIGNAL",
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
                    continue

                portfolio_snapshot = execution_engine.get_portfolio_snapshot()
                portfolio_cash = (
                    f"${portfolio_snapshot.available_cash_dollars:.2f}"
                    if portfolio_snapshot is not None
                    else (f"${update.available_cash_dollars:.2f}" if update.available_cash_dollars is not None else "n/a")
                )
                print(
                    "EXECUTION",
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
    finally:
        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()
        await collector.stop()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run the full live Kalshi stack in paper/demo mode with a regularized model."
    )
    parser.add_argument("--environment", choices=["demo", "production"], default="demo")
    parser.add_argument("--execution-mode", choices=["paper", "live"], default=None)
    parser.add_argument("--model-family", choices=["lasso", "elastic_net", "bagged_lasso", "linear_svm"], default="lasso")
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--policy-file", default=None)
    parser.add_argument("--contracts-per-order", type=int, default=None)
    parser.add_argument("--capital-pct-per-order", type=float, default=None)
    parser.add_argument("--kelly-fraction-multiplier", type=float, default=None)
    parser.add_argument("--kelly-fraction-cap-pct", type=float, default=None)
    parser.add_argument("--series", action="append", default=["KXBTC15M"], help="Series ticker prefix filter")
    parser.add_argument("--ticker", action="append", default=[], help="Explicit market ticker filter")
    parser.add_argument("--log-dir", default="output/live/kalshi/raw")
    parser.add_argument("--metadata-refresh-interval-seconds", type=float, default=300.0)
    args = parser.parse_args()
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
