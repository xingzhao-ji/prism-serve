"""Single-owner dynamic batches with bounded admitted work and fail-closed execution.

All bookkeeping runs on one asyncio event loop. The injected backend can execute
on a dedicated thread; it must return one output per input in the same order.
Cancellation never claims to preempt an already launched CUDA operation.
"""

import asyncio
import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any


class Overloaded(Exception):
    pass


class DeadlineExceeded(Exception):
    pass


class Unavailable(Exception):
    pass


@dataclass(eq=False)
class Work:
    value: Any
    future: asyncio.Future
    admitted: float
    deadline: float
    active: bool = False
    outcome: str | None = None


class BatchScheduler:
    def __init__(self, infer, *, max_batch=8, max_pending=64, wait_ms=5):
        if not 1 <= max_batch <= max_pending <= 1024 or not 0 <= wait_ms <= 100:
            raise ValueError("invalid scheduler bounds")
        self.infer, self.max_batch, self.max_pending = infer, max_batch, max_pending
        self.wait = wait_ms / 1000
        self.queue = deque()
        self.live = set()
        self.event = asyncio.Event()
        self.counters = Counter(
            {
                key: 0
                for key in (
                    "admitted",
                    "completed",
                    "expired",
                    "cancelled",
                    "rejected",
                    "worker_failures",
                    "failed",
                )
            }
        )
        self.batch_sizes = Counter()
        self.ready = False
        self.task = None
        self.queue_wait_ms = 0.0

    async def start(self):
        if self.task is not None:
            raise RuntimeError("scheduler cannot restart; create a new model process")
        self.ready = True
        self.task = asyncio.create_task(self._run())

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.close()

    def stats(self):
        return {
            **self.counters,
            "pending": len(self.live),
            "unresolved": sum(work.outcome is None for work in self.live),
            "queued": len(self.queue),
            "batch_sizes": dict(self.batch_sizes),
            "queue_wait_ms_total": self.queue_wait_ms,
            "ready": self.ready,
        }

    def _record_outcome(self, work, outcome):
        # The worker and caller timeout can observe the same elapsed deadline
        # on one loop turn. Only the first terminal observation owns its count.
        if work.outcome is None:
            work.outcome = outcome
            self.counters[outcome] += 1

    async def submit(self, value, timeout_ms=1000):
        if not math.isfinite(timeout_ms) or not 1 <= timeout_ms <= 30000:
            raise ValueError("timeout_ms must be finite and in [1, 30000]")
        if not self.ready:
            raise Unavailable("model worker unavailable")
        if len(self.live) >= self.max_pending:
            self.counters["rejected"] += 1
            raise Overloaded("admission capacity reached")
        now = time.monotonic()
        work = Work(value, asyncio.get_running_loop().create_future(), now, now + timeout_ms / 1000)
        self.live.add(work)
        self.queue.append(work)
        self.counters["admitted"] += 1
        self.event.set()
        try:
            async with asyncio.timeout(timeout_ms / 1000):
                return await asyncio.shield(work.future)
        except TimeoutError as exc:
            self._record_outcome(work, "expired")
            work.future.cancel()
            raise DeadlineExceeded("request deadline exceeded") from exc
        except asyncio.CancelledError:
            self._record_outcome(work, "cancelled")
            work.future.cancel()
            raise
        finally:
            if work.future.cancelled() and not work.active:
                if work in self.queue:
                    self.queue.remove(work)
                self.live.discard(work)
                self.event.set()

    def _fail_all(self):
        self.ready = False
        for work in self.live:
            if not work.future.done():
                self._record_outcome(work, "failed")
                work.future.set_exception(Unavailable("model worker unavailable"))
        self.live.clear()
        self.queue.clear()

    async def close(self):
        self._fail_all()
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        try:
            while self.ready:
                self.event.clear()
                if not self.queue:
                    await self.event.wait()
                    continue
                first = self.queue[0]
                limit = min(first.admitted + self.wait, *(w.deadline for w in self.queue))
                remaining = limit - time.monotonic()
                if len(self.queue) < self.max_batch and remaining > 0:
                    try:
                        await asyncio.wait_for(self.event.wait(), remaining)
                        continue
                    except TimeoutError:
                        pass
                batch = []
                while self.queue and len(batch) < self.max_batch:
                    work = self.queue.popleft()
                    if work.future.done():
                        self.live.discard(work)
                    elif work.deadline <= time.monotonic():
                        work.future.set_exception(DeadlineExceeded("deadline before execution"))
                        self._record_outcome(work, "expired")
                        self.live.discard(work)
                    else:
                        work.active = True
                        batch.append(work)
                        self.queue_wait_ms += (time.monotonic() - work.admitted) * 1000
                if not batch:
                    continue
                self.batch_sizes[len(batch)] += 1
                results = await self.infer([w.value for w in batch])
                if len(results) != len(batch):
                    raise RuntimeError("backend violated output cardinality")
                for work, result in zip(batch, results, strict=True):
                    self.live.discard(work)
                    if not work.future.done():
                        if time.monotonic() >= work.deadline:
                            work.future.set_exception(DeadlineExceeded("deadline during execution"))
                            self._record_outcome(work, "expired")
                        else:
                            work.future.set_result(result)
                            self._record_outcome(work, "completed")
        except asyncio.CancelledError:
            raise
        except Exception:
            self.counters["worker_failures"] += 1
            self._fail_all()
