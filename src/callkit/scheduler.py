"""Scheduler on top of the SDK: wraps any function that processes one item and manages concurrency, resuming, and capping.

    for r in run_items(items, ask, run="runs/E1", concurrency=8):          # uses SDK inside ask(payload)
        ...
    async for r in arun_items(items, aask, run="runs/E1", concurrency=32):  # aask is an async function
        ...

- Concurrency: the sync version uses a fixed-size thread pool, one item after another; the async version uses a fixed number of worker coroutines. Results are yielded one by one in completion order.
- Resuming: after each item finishes, a line is appended to ``items.jsonl`` in the run directory (success/failure, error, start/end times, call IDs produced); reruns skip
  already successful (item_id, repeat). Function return values are not persisted.
- Item context: before running the function, set ``callkit.context.CURRENT_ITEM``; calls made through a recording client or ``Client`` are tagged with
  item_id and repeat.
- Capping: before dispatching a new item, checks cumulative spend in the run directory; once the cap is reached, no new items are dispatched, in-flight items finish, and CapReached is raised after results are yielded.
- Errors: an exception from the function is recorded as failure for that item; other items still run; no item-level retries.
- Stopping early: the sync version cancels queued work and lets in-flight items finish; the async version cancels all worker coroutines (in-flight calls are cancelled).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import warnings
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from callkit.context import CURRENT_ITEM, ItemContext
from callkit.errors import CallkitError, CapReached
from callkit.store import RunStore, append_line, now_iso

LEDGER = "items.jsonl"


@dataclass
class ItemResult:
    item_id: str
    repeat: int
    ok: bool
    value: Any = None  # function return value (not persisted)
    error: str | None = None
    call_ids: list[str] = field(default_factory=list)  # call IDs produced by this item
    elapsed_s: float = 0.0


def run_items(
    items: Mapping[str, Any],
    fn: Callable[[Any], Any],
    *,
    run: str | Path,
    concurrency: int = 8,
    repeats: int = 1,
    cap_usd: float | None = None,
) -> Iterator[ItemResult]:
    """``items`` maps item_id to the argument passed to ``fn``. Fixed ``concurrency`` thread pool; results are yielded in completion order."""
    plan = _Plan(items, run, repeats, cap_usd)
    pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="callkit-item")

    def work(item_id: str, repeat: int) -> ItemResult | None:
        if not plan.may_start():
            return None
        ctx = ItemContext(item_id, repeat)
        token = CURRENT_ITEM.set(ctx)
        t0, started = time.monotonic(), now_iso()
        try:
            value, error = fn(items[item_id]), None
        # one item failing does not affect others; error is recorded in result and ledger
        except Exception as e:  # noqa: BLE001
            value, error = None, f"{type(e).__name__}: {e}"
        finally:
            CURRENT_ITEM.reset(token)
        return plan.finish(ctx, value, error, started, time.monotonic() - t0)

    try:
        futures = [pool.submit(work, *job) for job in plan.jobs]
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                yield res
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    plan.raise_if_capped()


async def arun_items(
    items: Mapping[str, Any],
    fn: Callable[[Any], Awaitable[Any]],
    *,
    run: str | Path,
    concurrency: int = 32,
    repeats: int = 1,
    cap_usd: float | None = None,
) -> AsyncIterator[ItemResult]:
    """Async version: ``fn`` is an async function; fixed ``concurrency`` worker coroutines yield results in completion order."""
    plan = _Plan(items, run, repeats, cap_usd)
    jobs = deque(plan.jobs)
    results: asyncio.Queue[ItemResult | _Broken | None] = asyncio.Queue()

    n = min(concurrency, len(jobs))
    workers = [asyncio.create_task(_worker(plan, fn, items, jobs, results)) for _ in range(n)]
    try:
        left = len(workers)
        while left:
            got = await results.get()
            if got is None:
                left -= 1
            elif isinstance(got, _Broken):
                # do not swallow: the remaining items were not run; the caller needs to know
                raise got.error
            else:
                yield got
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    plan.raise_if_capped()


async def _one(
    plan: _Plan, fn: Callable[[Any], Awaitable[Any]], items: Mapping[str, Any], job: tuple[str, int]
) -> ItemResult:
    ctx = ItemContext(*job)
    token = CURRENT_ITEM.set(ctx)
    t0, started = time.monotonic(), now_iso()
    try:
        value, error = await fn(items[ctx.item_id]), None
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise  # the scheduler itself is finishing (caller stopped): cancel as usual
        # something else inside the function was cancelled: count this item as failed
        value, error = None, "CancelledError"
    # one item failing does not affect others; error is recorded in result and ledger
    except Exception as e:  # noqa: BLE001
        value, error = None, f"{type(e).__name__}: {e}"
    finally:
        CURRENT_ITEM.reset(token)
    return plan.finish(ctx, value, error, started, time.monotonic() - t0)


async def _worker(
    plan: _Plan,
    fn: Callable[[Any], Awaitable[Any]],
    items: Mapping[str, Any],
    jobs: deque[tuple[str, int]],
    results: asyncio.Queue[ItemResult | _Broken | None],
) -> None:
    try:
        while jobs and plan.may_start():
            results.put_nowait(await _one(plan, fn, items, jobs.popleft()))
    except BaseException as e:
        # the scheduler itself failed (e.g. cannot write the ledger): hand it to the caller to raise
        results.put_nowait(_Broken(e))
        raise
    results.put_nowait(None)  # this worker coroutine is done


@dataclass
class _Broken:
    """A worker coroutine exited because the scheduler itself failed (not because an item failed)."""

    error: BaseException


class _Plan:
    """Shared by both versions: which (item_id, repeat) jobs to run, capping, and ledger."""

    def __init__(
        self, items: Mapping[str, Any], run: str | Path, repeats: int, cap_usd: float | None
    ) -> None:
        bad = [k for k in items if not isinstance(k, str)]
        if bad:
            raise CallkitError(f"item ids must be strings: {bad[:3]}")
        self.dir = Path(run).resolve()
        self.store = RunStore(self.dir)
        self.ledger = self.dir / LEDGER
        self.cap_usd = cap_usd
        done = _done(self.ledger)
        self.jobs = [(i, r) for i in items for r in range(repeats) if (i, r) not in done]
        self._capped = threading.Event()
        self._lock = threading.Lock()

    def may_start(self) -> bool:
        if self._capped.is_set():
            return False
        if self.cap_usd is not None and self.store.spent_usd() >= self.cap_usd:
            self._capped.set()
            return False
        return True

    def finish(
        self, ctx: ItemContext, value: Any, error: str | None, started: str, elapsed_s: float
    ) -> ItemResult:
        res = ItemResult(
            ctx.item_id, ctx.repeat, error is None, value, error, ctx.call_ids, elapsed_s
        )
        line = {
            "item_id": res.item_id,
            "repeat": res.repeat,
            "ok": res.ok,
            "error": res.error,
            "started_at": started,
            "finished_at": now_iso(),
            "call_ids": res.call_ids,
        }
        with self._lock:
            append_line(self.ledger, line)
        return res

    def raise_if_capped(self) -> None:
        if self._capped.is_set():
            raise CapReached(
                f"spent ${self.store.spent_usd():.4f}, reached the cap of ${self.cap_usd}; remaining items were not started"
            )


def _done(ledger: Path) -> set[tuple[str, int]]:
    """Successful (item_id, repeat) pairs in the ledger. Split the bytes into lines and decode line by line: a partially written Chinese character (crash mid-write) only corrupts that line,
    and U+2028 in item_id will not split a line."""
    if not ledger.exists():
        return set()
    done: set[tuple[str, int]] = set()
    for line in ledger.read_bytes().split(b"\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            warnings.warn(f"skipped an unreadable line in {ledger}", RuntimeWarning, stacklevel=4)
            continue
        if row.get("ok"):
            done.add((str(row["item_id"]), int(row["repeat"])))
    return done
