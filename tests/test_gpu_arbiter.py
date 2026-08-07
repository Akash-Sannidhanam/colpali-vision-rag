"""Tests for GPU exclusion, load shedding, and the ingest yield (src.gpu_arbiter).

Pure asyncio - no model, no threads beyond the one `run_exclusive`/`thread_gate` need,
and no sleeping on wall-clock timeouts longer than a test should take. The three
behaviours here are the ones that are invisible when they break in production: a query
queued behind an ingest, a lock released under a live forward pass, and a request left
hanging instead of shed.
"""

import asyncio
import functools
import threading
import time

import pytest

from src.gpu_arbiter import GpuArbiter, GpuBusy


def asyncio_test(fn):
    """Run an async test body under `asyncio.run`.

    The suite has no pytest-asyncio and does not need one for five tests - `test_server`
    already drives its one async helper this way. A decorator rather than an inline
    `asyncio.run(_body())` per test so the bodies stay readable `async def`s.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return wrapper


def _wait_until(predicate, timeout: float = 5.0) -> None:
    """Block this thread until `predicate()` holds. Stands in for a batch's real duration.

    A fake ingest batch returns instantly, so without this the whole document finishes
    before the waiting query's thread is even scheduled - and a gate test would pass
    while proving nothing. A real batch is seconds of forward pass, which is exactly the
    window a query queues up in.
    """
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the test's precondition")
        time.sleep(0.001)


# --- exclusion ---

@asyncio_test
async def test_two_holders_never_overlap():
    """The whole point: the one GPU-resident model runs one thing at a time."""
    gpu = GpuArbiter()
    concurrent = 0
    peak = 0

    async def work():
        nonlocal concurrent, peak
        async with gpu.acquire():
            concurrent += 1
            peak = max(peak, concurrent)
            await asyncio.sleep(0)          # yield, so an overlap would be observable
            concurrent -= 1

    await asyncio.gather(*(work() for _ in range(5)))
    assert peak == 1


@asyncio_test
async def test_run_exclusive_returns_the_worker_result():
    """The happy path: a blocking function runs off-loop and its value comes back."""
    gpu = GpuArbiter()
    assert await gpu.run_exclusive(lambda a, b: a + b, 2, 3) == 5


# --- shedding ---

@asyncio_test
async def test_acquire_sheds_when_the_model_stays_busy():
    """Past the timeout a waiter raises GpuBusy rather than hanging on the socket."""
    gpu = GpuArbiter()
    async with gpu.acquire():
        with pytest.raises(GpuBusy) as exc:
            async with gpu.acquire(timeout=0.01):
                pass                        # unreachable
    assert exc.value.retry_after >= 1       # a 0-second Retry-After retries immediately


@asyncio_test
async def test_a_shed_request_stops_counting_as_a_waiter():
    """A timed-out waiter must not leave the ingest yielding to a request that gave up."""
    gpu = GpuArbiter()
    async with gpu.acquire():
        with pytest.raises(GpuBusy):
            async with gpu.acquire(timeout=0.01):
                pass
        assert not gpu.contended()


@asyncio_test
async def test_timeout_zero_waits_indefinitely():
    """0 restores the pre-arbiter behaviour: wait, however long it takes."""
    gpu = GpuArbiter()
    holder_done = asyncio.Event()

    async def holder():
        async with gpu.acquire():
            await asyncio.sleep(0.05)
            holder_done.set()

    task = asyncio.create_task(holder())
    await asyncio.sleep(0)
    async with gpu.acquire(timeout=0):      # would have raised at any finite short timeout
        assert holder_done.is_set()
    await task


# --- disconnect safety ---

@asyncio_test
async def test_cancelling_the_caller_does_not_release_under_a_running_worker():
    """A client that disconnects mid-query must not hand the model to the next request.

    `asyncio.to_thread` cancellation cancels the awaiting task, never the thread, so a
    plain `async with lock:` releases while the forward pass is still running - two
    passes on one model, silently. This is the regression test for `run_exclusive`'s
    shield.
    """
    gpu = GpuArbiter()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_forward_pass():
        started.set()
        release.wait(timeout=5)
        finished.set()

    task = asyncio.create_task(gpu.run_exclusive(slow_forward_pass))
    await asyncio.to_thread(started.wait, 5)

    task.cancel()                                    # the client goes away
    await asyncio.sleep(0.02)                        # give a buggy release time to happen
    assert gpu.contended() or not finished.is_set()  # the worker is still running...
    with pytest.raises(GpuBusy):                     # ...and the model is still held
        async with gpu.acquire(timeout=0.01):
            pass

    release.set()                                    # let the forward pass finish
    with pytest.raises(asyncio.CancelledError):
        await task
    async with gpu.acquire(timeout=1):               # only now is it free
        assert finished.is_set()


# --- the ingest yield ---

@asyncio_test
async def test_gate_is_a_no_op_when_nothing_is_waiting():
    """The common case - every batch of an uncontended ingest - costs no round trip."""
    gpu = GpuArbiter()
    loop = asyncio.get_running_loop()
    gate = gpu.thread_gate(loop)

    async with gpu.acquire():
        # Called from a worker thread, exactly as ingest does. It must return without
        # scheduling anything on the loop, which is why this cannot deadlock even though
        # the loop is blocked waiting on the thread.
        await asyncio.to_thread(gate)


@asyncio_test
async def test_gate_hands_the_model_to_a_waiting_query_then_takes_it_back():
    """A query queued behind an ingest runs at the next page boundary, not at the end.

    This is the head-of-line-blocking fix: without the gate, `order` would be every
    batch followed by the query.
    """
    gpu = GpuArbiter()
    loop = asyncio.get_running_loop()
    gate = gpu.thread_gate(loop)
    order: list[str] = []
    query_may_start = threading.Event()

    def fake_ingest():
        """Three page batches, gating between them like ingest_pdf's loop."""
        for batch in range(3):
            gate()
            order.append(f"batch{batch}")
            if batch == 0:
                query_may_start.set()      # the query arrives during the second batch
                _wait_until(gpu.contended)  # ...and this batch is still being computed

    async def query():
        await asyncio.to_thread(query_may_start.wait, 5)
        async with gpu.acquire(timeout=5):
            order.append("query")

    ingest = asyncio.create_task(gpu.run_exclusive(fake_ingest))
    await asyncio.gather(ingest, query())

    # The query landed mid-document rather than after it - and the ingest resumed.
    assert order.index("query") < order.index("batch2")
    assert order[-1] == "batch2"


@asyncio_test
async def test_the_ingest_keeps_exclusive_access_across_a_yield():
    """Yielding hands the model over completely: never both at once, even mid-yield."""
    gpu = GpuArbiter()
    loop = asyncio.get_running_loop()
    gate = gpu.thread_gate(loop)
    holders = 0
    peak = 0
    lock = threading.Lock()
    query_may_start = threading.Event()

    def enter():
        nonlocal holders, peak
        with lock:
            holders += 1
            peak = max(peak, holders)

    def leave():
        nonlocal holders
        with lock:
            holders -= 1

    yielded = 0

    def fake_ingest():
        nonlocal yielded
        enter()
        for batch in range(3):
            query_may_start.set()
            if batch == 0:
                _wait_until(gpu.contended)  # the query queues while this batch computes
            leave()                        # about to give the model up
            if gpu.contended():
                yielded += 1
            gate()
            enter()                        # and to take it back
        leave()

    async def query():
        await asyncio.to_thread(query_may_start.wait, 5)
        async with gpu.acquire(timeout=5):
            enter()
            await asyncio.sleep(0)
            leave()

    await asyncio.gather(gpu.run_exclusive(fake_ingest), query())
    assert peak == 1
    assert yielded >= 1     # the handover actually happened, so peak==1 means something
