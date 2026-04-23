from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.indexers.kalshi.models import Market
from src.live.kalshi.config import KalshiEnvironment
from src.live.kalshi.signal_risk import calculate_realized_cash_metrics


@dataclass
class PortfolioSnapshotState:
    available_cash_dollars: float
    deployed_capital_dollars: float
    open_positions: list[str]
    pending_reservations: list[str]


@dataclass
class DecisionState:
    decision_id: str
    ticker: str
    side: str
    claimed_at: datetime
    contracts: int = 1
    limit_price_cents: int | None = None
    cash_required_dollars: float | None = None
    settled: bool = False


def _iter_event_rows(environment_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(environment_dir.rglob("events.jsonl")):
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event["_source_path"] = str(path)
            rows.append(event)
    rows.sort(key=lambda row: (str(row.get("logged_at") or ""), str(row.get("_source_path") or "")))
    return rows


def _append_event(path: Path, event_type: str, payload: dict, *, logged_at: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "logged_at": logged_at.isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _normalize_result(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    return normalized if normalized in {"YES", "NO"} else None


def _portfolio_snapshot_from_payload(payload: dict) -> PortfolioSnapshotState:
    return PortfolioSnapshotState(
        available_cash_dollars=float(payload.get("available_cash_dollars", 0.0) or 0.0),
        deployed_capital_dollars=float(payload.get("deployed_capital_dollars", 0.0) or 0.0),
        open_positions=[str(value) for value in payload.get("open_positions", []) or []],
        pending_reservations=[str(value) for value in payload.get("pending_reservations", []) or []],
    )


def _remove_one_ticker(values: list[str], ticker: str) -> None:
    try:
        values.remove(ticker)
    except ValueError:
        return


def _rebuild_snapshot_from_decisions(
    decisions: dict[str, DecisionState],
    *,
    initial_equity_dollars: float,
    cumulative_realized_pnl_dollars: float,
) -> PortfolioSnapshotState:
    open_decisions = [
        decision
        for decision in decisions.values()
        if not decision.settled and (decision.cash_required_dollars or 0.0) > 0.0
    ]
    deployed_capital_dollars = sum(float(decision.cash_required_dollars or 0.0) for decision in open_decisions)
    available_cash_dollars = float(initial_equity_dollars) + float(cumulative_realized_pnl_dollars) - deployed_capital_dollars
    return PortfolioSnapshotState(
        available_cash_dollars=available_cash_dollars,
        deployed_capital_dollars=deployed_capital_dollars,
        open_positions=[decision.ticker for decision in sorted(open_decisions, key=lambda item: (item.claimed_at, item.decision_id))],
        pending_reservations=[],
    )


def _fetch_public_market(ticker: str, *, environment: KalshiEnvironment) -> Market:
    with httpx.Client(base_url=environment.api_base_url, timeout=30.0) as client:
        response = client.get(f"/markets/{ticker}")
        response.raise_for_status()
        payload = response.json()
    return Market.from_dict(payload["market"])


def backfill_live_execution_settlements(
    run_dir: Path,
    *,
    environment: str = "production",
    lookup_environment: str | None = None,
    market_lookup: Callable[[str], Market] | None = None,
    settled_at_start: datetime | None = None,
) -> dict[str, dict[str, int]]:
    execution_root = run_dir / "execution"
    if not execution_root.exists():
        raise FileNotFoundError(f"Execution log directory not found: {execution_root}")

    target_environment = environment
    public_environment = KalshiEnvironment(lookup_environment or environment)
    lookup = market_lookup or (lambda ticker: _fetch_public_market(ticker, environment=public_environment))
    patched_by_model: dict[str, dict[str, int]] = {}

    for model_dir in sorted(path for path in execution_root.iterdir() if path.is_dir()):
        environment_dir = model_dir / target_environment
        if not environment_dir.exists():
            continue

        rows = _iter_event_rows(environment_dir)
        if not rows:
            continue

        mode: str | None = None
        initial_snapshot: PortfolioSnapshotState | None = None
        decisions: dict[str, DecisionState] = {}
        unpriced_decisions: list[str] = []
        previous_snapshot: PortfolioSnapshotState | None = None
        latest_snapshot: PortfolioSnapshotState | None = None
        cumulative_realized_pnl_dollars = 0.0

        for event in rows:
            event_type = str(event.get("event_type") or "")
            payload = event.get("payload") or {}
            logged_at = _parse_timestamp(event.get("logged_at"))
            if logged_at is None:
                continue

            if event_type == "execution_started":
                mode = str(payload.get("mode") or mode or "").lower() or mode
                continue

            if event_type == "intent_claimed":
                decision_id = str(payload.get("decision_id") or "")
                if not decision_id:
                    continue
                desired_contracts = payload.get("desired_contracts")
                remaining_contracts = payload.get("remaining_contracts_before_submit")
                contracts = int(
                    desired_contracts
                    if desired_contracts not in (None, "")
                    else remaining_contracts
                    if remaining_contracts not in (None, "")
                    else 1
                )
                decision = decisions.get(decision_id) or DecisionState(
                    decision_id=decision_id,
                    ticker=str(payload.get("ticker") or "unknown"),
                    side=str(payload.get("side") or "UNKNOWN").upper(),
                    claimed_at=logged_at,
                )
                decision.ticker = str(payload.get("ticker") or decision.ticker)
                decision.side = str(payload.get("side") or decision.side).upper()
                decision.limit_price_cents = (
                    int(payload["limit_price_cents"])
                    if payload.get("limit_price_cents") not in (None, "")
                    else decision.limit_price_cents
                )
                decision.contracts = max(1, contracts)
                decisions[decision_id] = decision
                if decision.cash_required_dollars is None:
                    unpriced_decisions.append(decision_id)
                continue

            if event_type == "simulated_position_settled":
                decision_id = str(payload.get("decision_id") or "")
                if not decision_id:
                    continue
                if decision_id in decisions:
                    decisions[decision_id].settled = True
                cumulative_realized_pnl_dollars = float(
                    payload.get("cumulative_realized_pnl_dollars", cumulative_realized_pnl_dollars) or cumulative_realized_pnl_dollars
                )
                continue

            if event_type.startswith(("paper_portfolio_", "shadow_portfolio_", "simulated_portfolio_")):
                snapshot = _portfolio_snapshot_from_payload(payload)
                if initial_snapshot is None:
                    initial_snapshot = snapshot
                if event_type.endswith("_claimed") and unpriced_decisions:
                    decision_id = unpriced_decisions.pop(0)
                    decision = decisions.get(decision_id)
                    if decision is not None and decision.cash_required_dollars is None:
                        inferred_cash = 0.0
                        if decision.limit_price_cents is not None:
                            _, _, inferred_cash = calculate_realized_cash_metrics(
                                entry_price_cents=decision.limit_price_cents,
                                contracts=decision.contracts,
                            )
                        if inferred_cash <= 0.0:
                            inferred_cash = snapshot.deployed_capital_dollars - (
                                previous_snapshot.deployed_capital_dollars if previous_snapshot is not None else 0.0
                            )
                        decision.cash_required_dollars = max(0.0, inferred_cash)
                previous_snapshot = snapshot
                latest_snapshot = snapshot

        if mode not in {"paper", "shadow"}:
            continue
        if latest_snapshot is None:
            continue
        if initial_snapshot is None:
            initial_snapshot = latest_snapshot

        unresolved = [
            decision
            for decision in decisions.values()
            if not decision.settled and (decision.cash_required_dollars or 0.0) > 0.0
        ]
        if not unresolved:
            continue

        unresolved.sort(key=lambda decision: (decision.claimed_at, decision.decision_id))
        market_cache: dict[str, Market] = {}
        patchable: list[tuple[DecisionState, Market]] = []
        for decision in unresolved:
            market = market_cache.get(decision.ticker)
            if market is None:
                try:
                    market = lookup(decision.ticker)
                except Exception:
                    continue
                market_cache[decision.ticker] = market
            if _normalize_result(market.result) is None:
                continue
            patchable.append((decision, market))

        if not patchable:
            continue

        now = settled_at_start or datetime.now(UTC)
        events_path = model_dir / target_environment / now.strftime("%Y-%m-%d") / "events.jsonl"
        initial_equity_dollars = (
            initial_snapshot.available_cash_dollars + initial_snapshot.deployed_capital_dollars
        )
        current_snapshot = _rebuild_snapshot_from_decisions(
            decisions,
            initial_equity_dollars=initial_equity_dollars,
            cumulative_realized_pnl_dollars=cumulative_realized_pnl_dollars,
        )

        patched_positions = 0
        for index, (decision, market) in enumerate(patchable):
            settlement_result = _normalize_result(market.result)
            if settlement_result is None:
                continue
            logged_at = now + timedelta(milliseconds=index)
            cash_required_dollars = float(decision.cash_required_dollars or 0.0)
            payout_dollars = float(decision.contracts) if settlement_result == decision.side else 0.0
            realized_pnl_dollars = payout_dollars - cash_required_dollars
            cumulative_realized_pnl_dollars += realized_pnl_dollars

            current_snapshot.available_cash_dollars += payout_dollars
            current_snapshot.deployed_capital_dollars = max(
                0.0,
                current_snapshot.deployed_capital_dollars - cash_required_dollars,
            )
            _remove_one_ticker(current_snapshot.open_positions, decision.ticker)

            _append_event(
                events_path,
                "simulated_position_settled",
                {
                    "decision_id": decision.decision_id,
                    "ticker": decision.ticker,
                    "side": decision.side,
                    "settlement_result": settlement_result,
                    "contracts": decision.contracts,
                    "cash_required_dollars": cash_required_dollars,
                    "realized_pnl_dollars": realized_pnl_dollars,
                    "cumulative_realized_pnl_dollars": cumulative_realized_pnl_dollars,
                },
                logged_at=logged_at,
            )
            _append_event(
                events_path,
                f"{mode}_portfolio_settled",
                {
                    "available_cash_dollars": current_snapshot.available_cash_dollars,
                    "deployed_capital_dollars": current_snapshot.deployed_capital_dollars,
                    "open_positions": list(current_snapshot.open_positions),
                    "pending_reservations": list(current_snapshot.pending_reservations),
                },
                logged_at=logged_at,
            )
            patched_positions += 1

        if patched_positions:
            patched_by_model[model_dir.name] = {"patched_positions": patched_positions}

    return patched_by_model


def watch_live_execution_settlements(
    run_dir: Path,
    *,
    environment: str = "production",
    lookup_environment: str | None = None,
    poll_seconds: float = 15.0,
) -> None:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")

    while True:
        patched = backfill_live_execution_settlements(
            run_dir,
            environment=environment,
            lookup_environment=lookup_environment,
        )
        if patched:
            stamp = datetime.now(UTC).isoformat()
            for model_name, stats in patched.items():
                print(f"{stamp} {model_name}: patched {stats['patched_positions']} positions", flush=True)
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill missing settlement events into paper/shadow Kalshi execution logs."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--environment", default="production")
    parser.add_argument(
        "--lookup-environment",
        default=None,
        help="Kalshi environment to use for public market settlement lookups. Defaults to --environment.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch the run directory and append settlement events as markets finalize.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=15.0,
        help="Polling interval in seconds for --watch mode.",
    )
    args = parser.parse_args()

    if args.watch:
        try:
            watch_live_execution_settlements(
                args.run_dir,
                environment=args.environment,
                lookup_environment=args.lookup_environment,
                poll_seconds=args.poll_seconds,
            )
        except KeyboardInterrupt:
            print("Stopped settlement watcher.")
        return

    patched = backfill_live_execution_settlements(
        args.run_dir,
        environment=args.environment,
        lookup_environment=args.lookup_environment,
    )
    if not patched:
        print("No missing paper/shadow settlements were backfilled.")
        return
    for model_name, stats in patched.items():
        print(f"{model_name}: patched {stats['patched_positions']} positions")


if __name__ == "__main__":
    main()
