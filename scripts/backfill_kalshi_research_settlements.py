from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.indexers.kalshi.models import parse_datetime


@dataclass(frozen=True)
class RecordedSample:
    sample_id: str
    ticker: str
    side: str
    contracts: int
    cash_required_dollars: float
    recorded_at: datetime


@dataclass(frozen=True)
class ModelState:
    open_samples: dict[str, RecordedSample]
    settled_count: int
    win_count: int
    loss_count: int
    cumulative_realized_pnl_dollars: float


def _parse_ticker_result(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f'Expected TICKER=YES|NO, got "{value}"')
    ticker, result = value.split("=", 1)
    normalized_ticker = ticker.strip()
    normalized_result = result.strip().upper()
    if not normalized_ticker:
        raise argparse.ArgumentTypeError(f'Invalid empty ticker in "{value}"')
    if normalized_result not in {"YES", "NO"}:
        raise argparse.ArgumentTypeError(f'Invalid settlement result "{result}" in "{value}"')
    return normalized_ticker, normalized_result


def _read_model_state(model_dir: Path, environment: str) -> ModelState:
    environment_dir = model_dir / environment
    open_samples: dict[str, RecordedSample] = {}
    settled_count = 0
    win_count = 0
    loss_count = 0
    cumulative_realized_pnl_dollars = 0.0

    if not environment_dir.exists():
        return ModelState(open_samples={}, settled_count=0, win_count=0, loss_count=0, cumulative_realized_pnl_dollars=0.0)

    for path in sorted(environment_dir.rglob("events.jsonl")):
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = event.get("payload", {})
            event_type = event.get("event_type")
            if event_type == "research_sample_recorded":
                sample = RecordedSample(
                    sample_id=str(payload["sample_id"]),
                    ticker=str(payload["ticker"]),
                    side=str(payload["side"]).upper(),
                    contracts=int(payload["contracts"]),
                    cash_required_dollars=float(payload["estimated_cash_required_dollars"]),
                    recorded_at=parse_datetime(event["logged_at"]),
                )
                open_samples[sample.sample_id] = sample
                continue
            if event_type != "research_sample_settled":
                continue
            sample_id = str(payload.get("sample_id") or "")
            open_samples.pop(sample_id, None)
            settled_count += 1
            is_win = bool(payload.get("is_win"))
            if is_win:
                win_count += 1
            else:
                loss_count += 1
            cumulative_realized_pnl_dollars += float(payload.get("realized_pnl_dollars", 0.0) or 0.0)

    return ModelState(
        open_samples=open_samples,
        settled_count=settled_count,
        win_count=win_count,
        loss_count=loss_count,
        cumulative_realized_pnl_dollars=cumulative_realized_pnl_dollars,
    )


def _append_event(path: Path, event_type: str, payload: dict, *, logged_at: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "logged_at": logged_at.isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def backfill_research_settlements(
    run_dir: Path,
    *,
    ticker_results: dict[str, str],
    environment: str = "demo",
) -> dict[str, dict[str, int]]:
    research_root = run_dir / "research"
    if not research_root.exists():
        raise FileNotFoundError(f"Research log directory not found: {research_root}")

    normalized_results = {ticker: result.upper() for ticker, result in ticker_results.items()}
    now = datetime.now(UTC)
    date_path = now.strftime("%Y-%m-%d")
    patched_by_model: dict[str, dict[str, int]] = {}

    for model_dir in sorted(path for path in research_root.iterdir() if path.is_dir()):
        state = _read_model_state(model_dir, environment)
        open_samples = [
            sample
            for sample in state.open_samples.values()
            if sample.ticker in normalized_results
        ]
        if not open_samples:
            continue

        open_samples.sort(key=lambda sample: (sample.recorded_at, sample.sample_id))
        events_path = model_dir / environment / date_path / "events.jsonl"
        settled_count = state.settled_count
        win_count = state.win_count
        loss_count = state.loss_count
        cumulative_realized_pnl_dollars = state.cumulative_realized_pnl_dollars
        remaining_open = len(state.open_samples)

        patched_count = 0
        for sample in open_samples:
            settlement_result = normalized_results[sample.ticker]
            payout_dollars = float(sample.contracts) if settlement_result == sample.side else 0.0
            realized_pnl_dollars = payout_dollars - sample.cash_required_dollars
            cumulative_realized_pnl_dollars += realized_pnl_dollars
            settled_count += 1
            remaining_open -= 1
            is_win = realized_pnl_dollars > 0.0
            if is_win:
                win_count += 1
            else:
                loss_count += 1

            _append_event(
                events_path,
                "research_sample_settled",
                {
                    "sample_id": sample.sample_id,
                    "ticker": sample.ticker,
                    "side": sample.side,
                    "settlement_result": settlement_result,
                    "is_win": is_win,
                    "contracts": sample.contracts,
                    "cash_required_dollars": sample.cash_required_dollars,
                    "realized_pnl_dollars": realized_pnl_dollars,
                    "cumulative_realized_pnl_dollars": cumulative_realized_pnl_dollars,
                },
                logged_at=now,
            )
            patched_count += 1

        _append_event(
            events_path,
            "research_summary_snapshot",
            {
                "open_sample_count": remaining_open,
                "settled_sample_count": settled_count,
                "win_count": win_count,
                "loss_count": loss_count,
                "cumulative_realized_pnl_dollars": cumulative_realized_pnl_dollars,
            },
            logged_at=now,
        )
        patched_by_model[model_dir.name] = {"patched_samples": patched_count}

    return patched_by_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill specific settled outcomes into research-only Kalshi logs.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--environment", default="demo")
    parser.add_argument(
        "--ticker-result",
        action="append",
        required=True,
        metavar="TICKER=YES|NO",
        help="Settlement outcome to backfill for an open research ticker.",
    )
    args = parser.parse_args()

    ticker_results = dict(_parse_ticker_result(value) for value in args.ticker_result)
    patched = backfill_research_settlements(
        args.run_dir,
        ticker_results=ticker_results,
        environment=args.environment,
    )
    if not patched:
        print("No matching open research samples found for the requested tickers.")
        return
    for model_name, stats in patched.items():
        print(f"{model_name}: patched {stats['patched_samples']} samples")


if __name__ == "__main__":
    main()
