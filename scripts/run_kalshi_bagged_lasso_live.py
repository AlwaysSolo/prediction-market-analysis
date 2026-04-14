from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.kalshi import (  # noqa: E402
    DEFAULT_BAGGED_LASSO_MODEL_FILE,
    KalshiCollectorConfig,
    KalshiCredentials,
    KalshiEnvironment,
    KalshiExecutionConfig,
    KalshiExecutionEngine,
    KalshiExecutionMode,
    KalshiFeatureEngineConfig,
    KalshiFeatureStateEngine,
    KalshiMarketDataCollector,
    KalshiPathDependentBinaryLayeringEngine,
    KalshiRegularizedLogisticScorer,
    KalshiRegularizedLogisticScorerConfig,
    KalshiSignalRiskConfig,
    KalshiSignalRiskEngine,
)
from src.live.kalshi.client import KalshiLiveRestClient  # noqa: E402
from src.live.kalshi.live_archive import (  # noqa: E402
    KalshiLiveArchiveConfig,
    KalshiLiveArchiveManager,
    KalshiLiveArchiveRuntime,
)
from src.live.kalshi.types import KalshiRawStreamEvent  # noqa: E402
from scripts.run_kalshi_regularized_execution_engine import (  # noqa: E402
    resolve_policy_file,
    signal_config_from_env_and_policy,
)

DEDICATED_LIVE_SIGNAL_PROFILE = "dedicated-v1"
RESEARCH_PARITY_SIGNAL_PROFILE = "research-parity"
SIGNAL_PROFILE_CHOICES = (DEDICATED_LIVE_SIGNAL_PROFILE, RESEARCH_PARITY_SIGNAL_PROFILE)
DEDICATED_LIVE_TARGET_SERIES = "KXBTC15M"
DEDICATED_LIVE_HOURLY_CONTEXT_SERIES = "KXBTCD"

DEDICATED_LIVE_MAX_TAU_MINUTES = 10.0
DEDICATED_LIVE_BLOCKED_REGIME_LABELS = frozenset({"neutral"})
DEDICATED_LIVE_BANNED_COMBO_BUCKETS = frozenset(
    {
        "12-14|60-70|70-80|5-10",
        "10-12|40-50|50-60|5-10",
        "10-12|30-40|30-40|5-10",
    }
)


def _resolve_model_and_policy(run_dir: Path | None, policy_file: str | None) -> tuple[Path, Path]:
    if run_dir is not None:
        model_file = run_dir / "bagged_lasso" / "model.joblib"
        resolved_policy = run_dir / "policy.json"
    else:
        model_file = DEFAULT_BAGGED_LASSO_MODEL_FILE
        resolved_policy = resolve_policy_file(policy_file, model_file)
    if not model_file.exists():
        raise FileNotFoundError(f"Bagged-lasso model file not found: {model_file}")
    if resolved_policy is None or not resolved_policy.exists():
        raise FileNotFoundError("Bagged-lasso policy.json is required for the dedicated live runner.")
    return model_file, resolved_policy


def _build_live_signal_config(
    environment: KalshiEnvironment,
    policy_file: Path,
    *,
    signal_profile: str = DEDICATED_LIVE_SIGNAL_PROFILE,
    allow_stacking: bool = False,
    disable_regime_control: bool = False,
    edge_threshold_cents: float | None = None,
) -> KalshiSignalRiskConfig:
    base_config, _loaded_policy = signal_config_from_env_and_policy(environment, policy_file)
    resolved_edge_threshold_cents = base_config.edge_threshold_cents if edge_threshold_cents is None else edge_threshold_cents
    resolved_maintain_edge_cents = min(base_config.maintain_edge_cents, resolved_edge_threshold_cents)
    common_overrides = dict(
        apply_regime_hard_gate=not disable_regime_control,
        enable_bucket_ban_policy=True,
        enable_low_liquidity_chop_gate=True,
        low_liquidity_min_trade_count_300s=3.0,
        low_liquidity_min_contracts_sum_300s=10.0,
        low_liquidity_min_abs_signed_contracts_sum_300s=3.0,
        low_liquidity_price_volatility_300s_threshold=0.01,
        structural_regime_refresh_seconds=30.0,
        structural_regime_flip_confirmations=2,
        contracts_per_order=1,
        capital_pct_per_order=None,
        kelly_fraction_multiplier=None,
        kelly_fraction_cap_pct=None,
        allow_stacking=allow_stacking,
        maintain_edge_cents=resolved_maintain_edge_cents,
    )
    if signal_profile == RESEARCH_PARITY_SIGNAL_PROFILE:
        config = replace(
            base_config,
            **common_overrides,
            edge_threshold_cents=resolved_edge_threshold_cents if edge_threshold_cents is not None else 2.0,
            min_tau_minutes=0.0,
            max_tau_minutes=15.0,
            price_band_min_cents=0,
            price_band_max_cents=100,
            enable_combo_ban_policy=False,
            banned_combo_buckets=frozenset(),
            blocked_regime_labels=frozenset(),
        )
        return config
    if signal_profile != DEDICATED_LIVE_SIGNAL_PROFILE:
        raise ValueError(f"Unsupported signal profile: {signal_profile}")
    config = replace(
        base_config,
        **common_overrides,
        edge_threshold_cents=resolved_edge_threshold_cents,
        enable_combo_ban_policy=True,
        banned_combo_buckets=DEDICATED_LIVE_BANNED_COMBO_BUCKETS,
        blocked_regime_labels=(frozenset() if disable_regime_control else DEDICATED_LIVE_BLOCKED_REGIME_LABELS),
        max_tau_minutes=min(base_config.max_tau_minutes, DEDICATED_LIVE_MAX_TAU_MINUTES),
    )
    return config


def _build_execution_config(
    environment: KalshiEnvironment,
    *,
    mode: str,
    log_dir: Path,
) -> KalshiExecutionConfig:
    base = KalshiExecutionConfig.from_env(environment)
    execution_mode = KalshiExecutionMode.SHADOW if mode == "shadow" else KalshiExecutionMode.LIVE
    return replace(
        base,
        mode=execution_mode,
        log_dir=log_dir,
        skip_rest_orderbook_check_for_immediate_orders=True,
        enable_direct_trade_intent_handoff_in_live_mode=True,
        no_probe_immediate_limit_cushion_cents=2,
    )


def validate_live_runner_preflight(
    *,
    mode: str,
    confirm_live: bool,
    execution_config: KalshiExecutionConfig,
) -> None:
    if mode == "live" and not confirm_live:
        raise RuntimeError("Live mode requires --confirm-live.")
    if mode == "live" and not execution_config.enable_live_trading:
        raise RuntimeError(
            "Live mode requires KALSHI_PROD_EXECUTION_ENABLE_LIVE_TRADING=true (or shared KALSHI_EXECUTION_ENABLE_LIVE_TRADING=true)."
        )


async def _preflight_subaccount(
    collector_config: KalshiCollectorConfig,
    *,
    subaccount: int,
) -> tuple[dict, list[dict]]:
    client = KalshiLiveRestClient(collector_config)
    try:
        balance, positions = await asyncio.gather(
            asyncio.to_thread(client.get_balance, subaccount=subaccount),
            asyncio.to_thread(client.get_positions, subaccount=subaccount),
        )
    finally:
        client.close()
    return balance, positions


async def _fetch_subaccount_balances(
    collector_config: KalshiCollectorConfig,
) -> list[dict]:
    client = KalshiLiveRestClient(collector_config)
    try:
        return await asyncio.to_thread(client.get_subaccount_balances)
    finally:
        client.close()


def validate_configured_subaccount_number(
    *,
    configured_subaccount: int,
    subaccount_balances: list[dict],
) -> None:
    valid_subaccounts: list[int] = []
    for item in subaccount_balances:
        raw_number = item.get("subaccount_number")
        if raw_number is None:
            continue
        try:
            valid_subaccounts.append(int(raw_number))
        except (TypeError, ValueError):
            continue
    valid_subaccounts = sorted(set(valid_subaccounts))
    if configured_subaccount in valid_subaccounts:
        return
    if valid_subaccounts:
        valid_display = ", ".join(str(number) for number in valid_subaccounts)
        raise RuntimeError(
            "Configured execution subaccount "
            f"{configured_subaccount} is not a valid Kalshi subaccount number for this account. "
            f"Valid subaccount numbers: {valid_display}. "
            "Set KALSHI_PROD_EXECUTION_SUBACCOUNT to one of those numeric values."
        )
    raise RuntimeError(
        "Kalshi did not return any subaccount numbers for this account during preflight. "
        "Verify the API key has portfolio access and that the account has at least the primary subaccount."
    )


async def _await_ticker_stream(
    raw_queue: asyncio.Queue[KalshiRawStreamEvent],
    *,
    timeout_seconds: float,
) -> KalshiRawStreamEvent:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for production ticker stream during live preflight.")
        event = await asyncio.wait_for(raw_queue.get(), timeout=remaining)
        if event.channel == "ticker":
            return event


async def _consume_signal_updates(queue: asyncio.Queue) -> None:
    while True:
        update = await queue.get()
        print(
            "SIGNAL",
            update.ticker,
            f"tau={update.tau_minutes:.2f}",
            f"side={update.side}",
            f"price={update.reference_price_cents}c" if update.reference_price_cents is not None else "price=?",
            (
                f"max_price={update.max_acceptable_entry_price_cents}c"
                if update.max_acceptable_entry_price_cents is not None
                else "max_price=?"
            ),
            f"regime={update.regime_label}",
            (
                f"bucket_block={update.bucket_policy_side}:{update.bucket_policy_dimension}:{update.bucket_policy_bucket}"
                if update.bucket_policy_bucket is not None
                else ""
            ),
            "APPROVED" if update.approved else f"BLOCKED:{update.block_reason}",
        )


async def _consume_execution_updates(queue: asyncio.Queue, execution_engine: KalshiExecutionEngine) -> None:
    while True:
        update = await queue.get()
        snapshot = execution_engine.get_portfolio_snapshot()
        cash_display = (
            f"${snapshot.available_cash_dollars:.2f}"
            if snapshot is not None
            else (f"${update.available_cash_dollars:.2f}" if update.available_cash_dollars is not None else "n/a")
        )


async def _consume_layering_updates(queue: asyncio.Queue) -> None:
    while True:
        update = await queue.get()
        print(
            "LAYERING",
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
            "EXECUTION",
            update.ticker,
            f"decision={update.decision_id}",
            f"side={update.side}",
            f"contracts={update.contracts}",
            f"limit={update.limit_price_cents}c",
            f"status={update.status}",
            f"filled={update.filled_contracts}/{update.contracts}",
            f"fill_price={update.fill_price_cents}c" if update.fill_price_cents is not None else "fill_price=?",
            f"cash={cash_display}",
            f"result={update.settlement_result}" if update.settlement_result else "",
            f"msg={update.message}" if update.message else "",
        )


async def _run(args: argparse.Namespace) -> None:
    environment = KalshiEnvironment.PRODUCTION

    model_file, policy_file = _resolve_model_and_policy(
        Path(args.run_dir) if args.run_dir else None,
        args.policy_file,
    )
    credentials = KalshiCredentials.from_env(environment)

    signal_config = _build_live_signal_config(
        environment,
        policy_file,
        signal_profile=args.signal_profile,
        allow_stacking=args.allow_stacking,
        disable_regime_control=args.disable_regime_control,
        edge_threshold_cents=args.edge_threshold_cents,
    )
    layering_engine = None
    if args.enable_layering:
        signal_config = replace(signal_config, auto_reserve_trade_intents=False)
    execution_log_dir = Path(args.log_root) / "execution" / "bagged_lasso"
    signal_log_dir = Path(args.log_root) / "signal" / "bagged_lasso"
    execution_config = _build_execution_config(
        environment,
        mode=args.mode,
        log_dir=execution_log_dir,
    )
    validate_live_runner_preflight(
        mode=args.mode,
        confirm_live=args.confirm_live,
        execution_config=execution_config,
    )

    collector_config = KalshiCollectorConfig(
        environment=environment,
        credentials=credentials,
        log_dir=Path(args.log_root) / "raw",
        series_tickers=(DEDICATED_LIVE_TARGET_SERIES, DEDICATED_LIVE_HOURLY_CONTEXT_SERIES),
        market_tickers=tuple(args.ticker),
        metadata_refresh_interval_seconds=args.metadata_refresh_interval_seconds,
    )

    subaccount_balances = await _fetch_subaccount_balances(collector_config)
    validate_configured_subaccount_number(
        configured_subaccount=execution_config.subaccount,
        subaccount_balances=subaccount_balances,
    )
    balance_payload, positions_payload = await _preflight_subaccount(
        collector_config,
        subaccount=execution_config.subaccount,
    )

    print("Initializing dedicated bagged-lasso Kalshi live stack...")
    print(f"Runner mode: {args.mode}")
    print(f"Environment: {environment.value}")
    print(f"Subaccount: {execution_config.subaccount}")
    print(f"Model file: {model_file}")
    print(f"Policy file: {policy_file}")
    print(f"Signal profile: {args.signal_profile}")
    print(
        "Resolved signal config:",
        f"edge={signal_config.edge_threshold_cents:.1f}c",
        f"tau={signal_config.min_tau_minutes:.1f}-{signal_config.max_tau_minutes:.1f}",
        f"price_band={signal_config.price_band_min_cents}-{signal_config.price_band_max_cents}c",
        f"regime={signal_config.apply_regime_hard_gate}",
        f"blocked_regimes={','.join(sorted(signal_config.blocked_regime_labels)) or 'none'}",
        f"bucket_ban={signal_config.enable_bucket_ban_policy}",
        f"combo_ban={signal_config.enable_combo_ban_policy}",
        f"contracts={signal_config.contracts_per_order}",
        f"stacking={signal_config.allow_stacking}",
    )
    print(
        "Subaccount preflight:",
        f"balance_cents={balance_payload.get('balance')}",
        f"open_positions={len(positions_payload)}",
    )
    print(
        "Available subaccounts:",
        ",".join(
            str(item.get("subaccount_number"))
            for item in subaccount_balances
            if item.get("subaccount_number") is not None
        )
        or "none",
    )

    collector = KalshiMarketDataCollector(collector_config)
    preflight_raw_queue = collector.subscribe_raw_stream_queue()
    feature_engine = KalshiFeatureStateEngine(
        collector,
        config=KalshiFeatureEngineConfig(
            publish_series_tickers=(DEDICATED_LIVE_TARGET_SERIES,),
            hourly_context_target_series_ticker=DEDICATED_LIVE_TARGET_SERIES,
            hourly_context_series_ticker=DEDICATED_LIVE_HOURLY_CONTEXT_SERIES,
        ),
    )
    scorer = KalshiRegularizedLogisticScorer(
        feature_engine,
        KalshiRegularizedLogisticScorerConfig(model_file=model_file),
    )
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        signal_config,
        log_dir=signal_log_dir,
    )
    if args.enable_layering:
        layering_engine = KalshiPathDependentBinaryLayeringEngine(
            signal_engine,
            log_dir=Path(args.log_root) / "layering" / "bagged_lasso",
        )
    execution_engine = KalshiExecutionEngine(
        signal_engine,
        execution_config,
        trade_intent_source=layering_engine,
    )
    if layering_engine is not None:
        layering_engine.bind_execution_engine(execution_engine)
    archive_manager = KalshiLiveArchiveManager(
        collector,
        feature_engine,
        [
            KalshiLiveArchiveRuntime(
                label="bagged_lasso",
                family="bagged_lasso",
                model_file=model_file,
                scorer=scorer,
                signal_engine=signal_engine,
                execution_engine=execution_engine,
                layering_engine=layering_engine,
                calibration_enabled=getattr(scorer.config, "apply_calibration", False),
            )
        ],
        KalshiLiveArchiveConfig(
            archive_root=Path(args.archive_root) if args.archive_root else Path(args.log_root) / "archive",
            run_name=Path(args.log_root).name,
            environment=environment.value,
            compact_on_shutdown=not args.no_archive_compact_on_shutdown,
        ),
    )

    signal_queue = signal_engine.subscribe_queue()
    execution_queue = execution_engine.subscribe_queue()
    layering_queue = None if layering_engine is None else layering_engine.subscribe_queue()

    print("Starting collector...")
    await collector.start()
    await collector.wait_until_ready()
    first_ticker_event = await _await_ticker_stream(
        preflight_raw_queue,
        timeout_seconds=args.quote_start_timeout_seconds,
    )
    print(
        "Quote stream preflight:",
        f"ticker={first_ticker_event.market_ticker}",
        f"received_at={first_ticker_event.received_at.isoformat()}",
    )

    print("Starting feature, scorer, signal, execution, and archive layers...")
    await feature_engine.start()
    await scorer.start()
    await signal_engine.start()
    await execution_engine.start()
    if layering_engine is not None:
        await layering_engine.start()
    await archive_manager.start()

    print("Dedicated bagged-lasso live stack started.")
    print(f"Dashboard root: {args.log_root}")
    print("Open: http://localhost:8765/tools/live_trading_dashboard.html")
    print(f"Choose folder: {args.log_root}")

    signal_task = asyncio.create_task(_consume_signal_updates(signal_queue), name="bagged-lasso-live-signal")
    execution_task = asyncio.create_task(
        _consume_execution_updates(execution_queue, execution_engine),
        name="bagged-lasso-live-execution",
    )
    layering_task = None
    if layering_queue is not None:
        layering_task = asyncio.create_task(
            _consume_layering_updates(layering_queue),
            name="bagged-lasso-live-layering",
        )
    try:
        await asyncio.Event().wait()
    finally:
        signal_task.cancel()
        execution_task.cancel()
        if layering_task is not None:
            layering_task.cancel()
        for task in tuple(task for task in (signal_task, execution_task, layering_task) if task is not None):
            try:
                await task
            except asyncio.CancelledError:
                pass
        await archive_manager.stop()
        if layering_engine is not None:
            await layering_engine.stop()
        await execution_engine.stop()
        await signal_engine.stop()
        await scorer.stop()
        await feature_engine.stop()
        await collector.stop()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run the dedicated bagged-lasso Kalshi production bot in shadow or live mode."
    )
    parser.add_argument("--mode", choices=["shadow", "live"], default="shadow")
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument(
        "--signal-profile",
        choices=SIGNAL_PROFILE_CHOICES,
        default=DEDICATED_LIVE_SIGNAL_PROFILE,
        help=(
            "Signal policy profile. "
            "'dedicated-v1' keeps the stricter live-only regime/neutral/combo protections. "
            "'research-parity' matches the broader 2c / 0-15 / 0-100 research-style filter set."
        ),
    )
    parser.add_argument(
        "--allow-stacking",
        action="store_true",
        help="Enable stacking for the dedicated bagged-lasso runner. Off by default.",
    )
    parser.add_argument(
        "--disable-regime-control",
        action="store_true",
        help="Disable blocked-regime labels and the YES/downtrend regime gate for testing.",
    )
    parser.add_argument(
        "--edge-threshold-cents",
        type=float,
        default=None,
        help="Optional explicit edge threshold override. Useful for widening trade intake during shadow tests.",
    )
    parser.add_argument(
        "--enable-layering",
        action="store_true",
        help="Enable the path-dependent KXBTC15M layering engine instead of direct one-shot entries.",
    )
    parser.add_argument("--environment", choices=["production"], default="production")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--policy-file", default=None)
    parser.add_argument("--ticker", action="append", default=[], help="Optional explicit market ticker filter")
    parser.add_argument("--log-root", default="output/live/kalshi_bagged_lasso_live_v1")
    parser.add_argument("--archive-root", default=None)
    parser.add_argument("--no-archive-compact-on-shutdown", action="store_true")
    parser.add_argument("--metadata-refresh-interval-seconds", type=float, default=300.0)
    parser.add_argument("--quote-start-timeout-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
