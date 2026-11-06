import asyncio
import time

import pytest

from prismserve.scheduler import BatchScheduler, DeadlineExceeded, Overloaded, Unavailable


@pytest.mark.asyncio
async def test_elapsed_deadline_is_counted_once_when_event_loop_is_delayed():
    async def delayed_completion(items):
        # Reproduce both the worker and request timeout noticing the same
        # deadline after a delayed event-loop turn, without timing a live GPU.
        time.sleep(0.02)
        return items

    async with BatchScheduler(delayed_completion, max_batch=1, wait_ms=0) as scheduler:
        with pytest.raises(DeadlineExceeded):
            await scheduler.submit("one", timeout_ms=1)
        stats = scheduler.stats()
        assert stats["admitted"] == 1
        assert stats["completed"] == 0
        assert stats["expired"] == 1


async def test_batches_concurrent_requests_in_order():
    batches = []

    async def infer(items):
        batches.append(items)
        return [x * 2 for x in items]

    async with BatchScheduler(infer, max_batch=4, max_pending=8, wait_ms=10) as scheduler:
        assert await asyncio.gather(*(scheduler.submit(x, 1000) for x in range(7))) == [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
        ]
        assert [len(b) for b in batches] == [4, 3]
        assert scheduler.stats()["completed"] == 7


async def test_capacity_includes_active_work_and_cancelled_queue_releases_slot():
    started, release = asyncio.Event(), asyncio.Event()

    async def infer(items):
        started.set()
        await release.wait()
        return items

    async with BatchScheduler(infer, max_batch=1, max_pending=2, wait_ms=0) as scheduler:
        first = asyncio.create_task(scheduler.submit(1, 1000))
        await started.wait()
        second = asyncio.create_task(scheduler.submit(2, 1000))
        await asyncio.sleep(0)
        with pytest.raises(Overloaded):
            await scheduler.submit(3, 100)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        release.set()
        assert await first == 1
        assert await scheduler.submit(4, 1000) == 4


async def test_deadline_expires_during_compute_without_publishing_late_result():
    async def infer(items):
        await asyncio.sleep(0.04)
        return items

    async with BatchScheduler(infer, max_batch=1, wait_ms=0) as scheduler:
        with pytest.raises(DeadlineExceeded):
            await scheduler.submit(1, 5)
        await asyncio.sleep(0.06)
        assert scheduler.stats()["completed"] == 0
        assert scheduler.stats()["pending"] == 0


async def test_worker_exception_fails_closed_and_drains_pending():
    async def infer(items):
        raise RuntimeError("device failure")

    async with BatchScheduler(infer, max_batch=2) as scheduler:
        results = await asyncio.gather(
            scheduler.submit(1, 1000), scheduler.submit(2, 1000), return_exceptions=True
        )
        assert all(isinstance(x, Unavailable) for x in results)
        assert not scheduler.ready
        with pytest.raises(Unavailable):
            await scheduler.submit(3, 1000)


async def test_shutdown_resolves_waiters_and_rejects_new_work():
    async def infer(items):
        await asyncio.sleep(10)
        return items

    scheduler = BatchScheduler(infer, wait_ms=0)
    await scheduler.start()
    task = asyncio.create_task(scheduler.submit(1, 1000))
    await asyncio.sleep(0.005)
    await scheduler.close()
    with pytest.raises(Unavailable):
        await task
    with pytest.raises(Unavailable):
        await scheduler.submit(2, 1000)


async def test_incorrect_backend_cardinality_is_fatal():
    async def infer(items):
        return []

    async with BatchScheduler(infer) as scheduler:
        with pytest.raises(Unavailable):
            await scheduler.submit(1, 1000)
        assert not scheduler.ready
