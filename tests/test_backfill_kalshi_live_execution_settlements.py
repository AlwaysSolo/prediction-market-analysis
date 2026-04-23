from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.backfill_kalshi_live_execution_settlements import backfill_live_execution_settlements
from src.indexers.kalshi.models import Market


def _write_events(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_backfill_live_execution_settlements_appends_settlement_and_portfolio_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "live" / "run"
    events_path = run_dir / "execution" / "lasso" / "production" / "2026-01-01" / "events.jsonl"
    ticker = "KXBTC15M-TEST1"
    _write_events(
        events_path,
        [
            {
                "logged_at": "2026-01-01T12:00:00+00:00",
                "event_type": "execution_started",
                "payload": {"mode": "paper"},
            },
            {
                "logged_at": "2026-01-01T12:00:01+00:00",
                "event_type": "intent_claimed",
                "payload": {
                    "decision_id": "decision-1",
                    "ticker": ticker,
                    "side": "YES",
                    "limit_price_cents": 60,
                },
            },
            {
                "logged_at": "2026-01-01T12:00:01+00:00",
                "event_type": "paper_portfolio_claimed",
                "payload": {
                    "available_cash_dollars": 99.39,
                    "deployed_capital_dollars": 0.61,
                    "open_positions": [],
                    "pending_reservations": [ticker],
                },
            },
            {
                "logged_at": "2026-01-01T12:00:01.100000+00:00",
                "event_type": "paper_portfolio_accepted",
                "payload": {
                    "available_cash_dollars": 99.39,
                    "deployed_capital_dollars": 0.61,
                    "open_positions": [],
                    "pending_reservations": [ticker],
                },
            },
            {
                "logged_at": "2026-01-01T12:00:01.200000+00:00",
                "event_type": "paper_portfolio_filled",
                "payload": {
                    "available_cash_dollars": 99.39,
                    "deployed_capital_dollars": 0.61,
                    "open_positions": [ticker],
                    "pending_reservations": [],
                },
            },
        ],
    )

    market = Market(
        ticker=ticker,
        event_ticker="KXBTC15M",
        market_type="binary",
        title="t",
        yes_sub_title="y",
        no_sub_title="n",
        status="settled",
        yes_bid=None,
        yes_ask=None,
        no_bid=None,
        no_ask=None,
        last_price=60,
        volume=1,
        volume_24h=1,
        open_interest=1,
        result="yes",
        created_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        open_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        close_time=datetime(2026, 1, 1, 12, 15, tzinfo=UTC),
    )

    patched = backfill_live_execution_settlements(
        run_dir,
        environment="production",
        market_lookup=lambda requested_ticker: market if requested_ticker == ticker else None,  # type: ignore[return-value]
        settled_at_start=datetime(2026, 1, 1, 13, 0, tzinfo=UTC),
    )

    assert patched == {"lasso": {"patched_positions": 1}}

    rows = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    settlement = next(row for row in rows if row["event_type"] == "simulated_position_settled")
    assert settlement["payload"]["decision_id"] == "decision-1"
    assert settlement["payload"]["settlement_result"] == "YES"
    assert settlement["payload"]["realized_pnl_dollars"] == 0.38

    portfolio_settlement = next(row for row in rows if row["event_type"] == "paper_portfolio_settled")
    assert portfolio_settlement["payload"]["available_cash_dollars"] == 100.38
    assert portfolio_settlement["payload"]["deployed_capital_dollars"] == 0.0
    assert portfolio_settlement["payload"]["open_positions"] == []


def test_backfill_rebuilds_portfolio_from_decisions_instead_of_stale_latest_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "live" / "run"
    events_path = run_dir / "execution" / "elastic_net" / "production" / "2026-01-01" / "events.jsonl"
    settled_ticker = "KXBTC15M-SETTLED"
    unresolved_ticker = "KXBTC15M-UNRESOLVED"
    _write_events(
        events_path,
        [
            {
                "logged_at": "2026-01-01T12:00:00+00:00",
                "event_type": "execution_started",
                "payload": {"mode": "paper"},
            },
            {
                "logged_at": "2026-01-01T12:00:01+00:00",
                "event_type": "intent_claimed",
                "payload": {
                    "decision_id": "decision-settled",
                    "ticker": settled_ticker,
                    "side": "YES",
                    "limit_price_cents": 60,
                },
            },
            {
                "logged_at": "2026-01-01T12:00:01+00:00",
                "event_type": "paper_portfolio_claimed",
                "payload": {
                    "available_cash_dollars": 99.39,
                    "deployed_capital_dollars": 0.61,
                    "open_positions": [],
                    "pending_reservations": [settled_ticker],
                },
            },
            {
                "logged_at": "2026-01-01T12:00:01.100000+00:00",
                "event_type": "paper_portfolio_filled",
                "payload": {
                    "available_cash_dollars": 99.39,
                    "deployed_capital_dollars": 0.61,
                    "open_positions": [settled_ticker],
                    "pending_reservations": [],
                },
            },
            {
                "logged_at": "2026-01-01T12:15:00+00:00",
                "event_type": "simulated_position_settled",
                "payload": {
                    "decision_id": "decision-settled",
                    "ticker": settled_ticker,
                    "side": "YES",
                    "settlement_result": "YES",
                    "contracts": 1,
                    "cash_required_dollars": 0.61,
                    "realized_pnl_dollars": 0.39,
                    "cumulative_realized_pnl_dollars": 0.39,
                },
            },
            {
                "logged_at": "2026-01-01T12:15:00+00:00",
                "event_type": "paper_portfolio_settled",
                "payload": {
                    "available_cash_dollars": 100.39,
                    "deployed_capital_dollars": 0.0,
                    "open_positions": [],
                    "pending_reservations": [],
                },
            },
            {
                "logged_at": "2026-01-01T12:20:00+00:00",
                "event_type": "intent_claimed",
                "payload": {
                    "decision_id": "decision-unresolved",
                    "ticker": unresolved_ticker,
                    "side": "YES",
                    "limit_price_cents": 70,
                },
            },
            {
                "logged_at": "2026-01-01T12:20:00+00:00",
                "event_type": "paper_portfolio_claimed",
                "payload": {
                    "available_cash_dollars": 98.69,
                    "deployed_capital_dollars": 1.31,
                    "open_positions": [settled_ticker],
                    "pending_reservations": [unresolved_ticker],
                },
            },
            {
                "logged_at": "2026-01-01T12:20:00.100000+00:00",
                "event_type": "paper_portfolio_filled",
                "payload": {
                    "available_cash_dollars": 98.69,
                    "deployed_capital_dollars": 1.31,
                    "open_positions": [settled_ticker, unresolved_ticker],
                    "pending_reservations": [],
                },
            },
        ],
    )

    market = Market(
        ticker=unresolved_ticker,
        event_ticker="KXBTC15M",
        market_type="binary",
        title="t",
        yes_sub_title="y",
        no_sub_title="n",
        status="settled",
        yes_bid=None,
        yes_ask=None,
        no_bid=None,
        no_ask=None,
        last_price=70,
        volume=1,
        volume_24h=1,
        open_interest=1,
        result="yes",
        created_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        open_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        close_time=datetime(2026, 1, 1, 12, 30, tzinfo=UTC),
    )

    patched = backfill_live_execution_settlements(
        run_dir,
        environment="production",
        market_lookup=lambda requested_ticker: market if requested_ticker == unresolved_ticker else None,  # type: ignore[return-value]
        settled_at_start=datetime(2026, 1, 1, 13, 0, tzinfo=UTC),
    )

    assert patched == {"elastic_net": {"patched_positions": 1}}

    rows = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    final_portfolio_settlement = [
        row for row in rows if row["event_type"] == "paper_portfolio_settled"
    ][-1]
    assert final_portfolio_settlement["payload"]["open_positions"] == []
    assert final_portfolio_settlement["payload"]["deployed_capital_dollars"] == 0.0
    assert final_portfolio_settlement["payload"]["available_cash_dollars"] == 100.67


def test_backfill_is_idempotent_on_repeated_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "live" / "run"
    events_path = run_dir / "execution" / "lasso" / "production" / "2026-01-01" / "events.jsonl"
    ticker = "KXBTC15M-IDEMPOTENT"
    _write_events(
        events_path,
        [
            {
                "logged_at": "2026-01-01T12:00:00+00:00",
                "event_type": "execution_started",
                "payload": {"mode": "paper"},
            },
            {
                "logged_at": "2026-01-01T12:00:01+00:00",
                "event_type": "intent_claimed",
                "payload": {
                    "decision_id": "decision-1",
                    "ticker": ticker,
                    "side": "YES",
                    "limit_price_cents": 60,
                },
            },
            {
                "logged_at": "2026-01-01T12:00:01+00:00",
                "event_type": "paper_portfolio_claimed",
                "payload": {
                    "available_cash_dollars": 99.39,
                    "deployed_capital_dollars": 0.61,
                    "open_positions": [],
                    "pending_reservations": [ticker],
                },
            },
            {
                "logged_at": "2026-01-01T12:00:01.100000+00:00",
                "event_type": "paper_portfolio_filled",
                "payload": {
                    "available_cash_dollars": 99.39,
                    "deployed_capital_dollars": 0.61,
                    "open_positions": [ticker],
                    "pending_reservations": [],
                },
            },
        ],
    )

    market = Market(
        ticker=ticker,
        event_ticker="KXBTC15M",
        market_type="binary",
        title="t",
        yes_sub_title="y",
        no_sub_title="n",
        status="settled",
        yes_bid=None,
        yes_ask=None,
        no_bid=None,
        no_ask=None,
        last_price=60,
        volume=1,
        volume_24h=1,
        open_interest=1,
        result="yes",
        created_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        open_time=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        close_time=datetime(2026, 1, 1, 12, 15, tzinfo=UTC),
    )

    first = backfill_live_execution_settlements(
        run_dir,
        environment="production",
        market_lookup=lambda requested_ticker: market if requested_ticker == ticker else None,  # type: ignore[return-value]
        settled_at_start=datetime(2026, 1, 1, 13, 0, tzinfo=UTC),
    )
    second = backfill_live_execution_settlements(
        run_dir,
        environment="production",
        market_lookup=lambda requested_ticker: market if requested_ticker == ticker else None,  # type: ignore[return-value]
        settled_at_start=datetime(2026, 1, 1, 13, 5, tzinfo=UTC),
    )

    rows = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert first == {"lasso": {"patched_positions": 1}}
    assert second == {}
    assert sum(1 for row in rows if row["event_type"] == "simulated_position_settled") == 1
