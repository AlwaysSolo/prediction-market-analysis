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
    KalshiFeatureStateEngine,
    KalshiMarketDataCollector,
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
) -> KalshiSignalRiskConfig:
    base_config, _loaded_policy = signal_config_from_env_and_policy(environment, policy_file)
    common_overrides = dict(
        apply_regime_hard_gate=True,
        enable_bucket_ban_policy=True,
        contracts_per_order=1,
        capital_pct_per_order=None,
        kelly_fraction_multiplier=None,
        kelly_fraction_cap_pct=None,
        allow_stacking=allow_stacking,
    )
    if signal_profile == RESEARCH_PARITY_SIGNAL_PROFILE:
        return replace(
            base_config,
            **common_overrides,
            edge_threshold_cents=2.0,
            min_tau_minutes=0.0,
            max_tau_minutes=15.0,
            price_band_min_cents=0,
            price_band_max_cents=100,
            enable_combo_ban_policy=False,
            banned_combo_buckets=frozenset(),
            blocked_regime_labels=frozenset(),
        )
    if signal_profile != DEDICATED_LIVE_SIGNAL_PROFILE:
        raise ValueError(f"Unsupported signal profile: {signal_profile}")
    return replace(
        base_config,
        **common_overrides,
        enable_combo_ban_policy=True,
        banned_combo_buckets=DEDICATED_LIVE_BANNED_COMBO_BUCKETS,
        blocked_regime_labels=DEDICATED_LIVE_BLOCKED_REGIME_LABELS,
        max_tau_minutes=min(base_config.max_tau_minutes, DEDICATED_LIVE_MAX_TAU_MINUTES),
    )


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
    )


def validate_live_runner_preflight(
    *,
    mode: str,
    confirm_live: bool,
    execution_config: KalshiExecutionConfig,
) -> None:
    if execution_config.subaccount <= 0:
        raise RuntimeError("Dedicated bagged-lasso live runner requires a nonzero production subaccount.")
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
    )
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
        series_tickers=("KXBTC15M",),
        market_tickers=tuple(args.ticker),
        metadata_refresh_interval_seconds=args.metadata_refresh_interval_seconds,
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

    collector = KalshiMarketDataCollector(collector_config)
    preflight_raw_queue = collector.subscribe_raw_stream_queue()
    feature_engine = KalshiFeatureStateEngine(collector)
    scorer = KalshiRegularizedLogisticScorer(
        feature_engine,
        KalshiRegularizedLogisticScorerConfig(model_file=model_file),
    )
    signal_engine = KalshiSignalRiskEngine(
        scorer,
        signal_config,
        log_dir=signal_log_dir,
    )
    execution_engine = KalshiExecutionEngine(signal_engine, execution_config)
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
    try:
        await asyncio.Event().wait()
    finally:
        signal_task.cancel()
        execution_task.cancel()
        for task in (signal_task, execution_task):
            try:
                await task
            except asyncio.CancelledError:
                pass
        await archive_manager.stop()
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
