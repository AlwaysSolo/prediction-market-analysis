from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SENTINEL = object()


def utc_now() -> datetime:
    return datetime.now(UTC)


class JsonlEventLogger:
    def __init__(self, base_dir: Path, environment: str):
        self.base_dir = base_dir
        self.environment = environment
        self._queue: asyncio.Queue[object] | None = None
        self._writer_task: asyncio.Task[None] | None = None

    def _ensure_writer(self) -> None:
        if self._writer_task is not None and not self._writer_task.done():
            return
        self._queue = asyncio.Queue()
        self._writer_task = asyncio.create_task(
            self._writer_loop(),
            name=f"kalshi-jsonl-logger-{self.environment}",
        )

    async def write(self, event_type: str, payload: dict[str, Any], event_time: datetime | None = None) -> None:
        event_time = event_time or utc_now()
        row = {
            "logged_at": utc_now().isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        path = self.base_dir / self.environment / event_time.strftime("%Y-%m-%d") / "events.jsonl"
        self._ensure_writer()
        assert self._queue is not None
        await self._queue.put((path, row))

    async def flush(self) -> None:
        if self._queue is None:
            return
        await self._queue.join()

    async def close(self) -> None:
        if self._queue is None or self._writer_task is None:
            return
        await self._queue.join()
        await self._queue.put(_SENTINEL)
        await self._writer_task
        self._queue = None
        self._writer_task = None

    async def _writer_loop(self) -> None:
        assert self._queue is not None
        queue = self._queue
        while True:
            item = await queue.get()
            try:
                if item is _SENTINEL:
                    return
                path, row = item
                assert isinstance(path, Path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, default=str) + "\n")
            finally:
                queue.task_done()
