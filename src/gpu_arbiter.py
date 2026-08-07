"""Exclusive access to the one GPU-resident model, shared between queries and ingests.

The server holds a single ColQwen2 in memory and must never run two forward passes at
once, so every GPU-touching endpoint serializes here. This is a seam rather than a bare
`asyncio.Lock` for the same reason the protected `api` router exists in `src/server.py`:
the two things that are easy to get wrong are structural, not per-route opt-ins.

**Shedding, not hanging.** `acquire` is bounded (`GPU_WAIT_TIMEOUT_S`) and raises
`GpuBusy`, which `server` maps to 503 + `Retry-After` - the same shape `ratelimit` gives
a 429. A client told the server is busy can back off; one left hanging on a socket for
four minutes cannot.

**A disconnecting client must not release the lock.** `await asyncio.to_thread(...)`
cancels the *awaiting task*, never the thread, so a plain `async with lock:` around it
releases the moment the client goes away - while the worker thread is still mid-forward
pass, which is exactly the collision the lock exists to prevent. `run_exclusive` shields
the work and refuses to release until the thread actually returns.

**A long ingest yields between page batches.** An ingest holds the model for minutes,
and the fix is not to make queries wait more politely but to make the ingest give the
GPU back. `thread_gate` returns a plain sync callable for the ingest's worker thread
(`ingest.run_ingest(gate=...)`); at each page boundary it hands the lock to whoever is
queued and takes it back afterwards. A queued query waits about one page, not one
document.

Two properties hold this together and neither is visible from the code alone:

  * **`asyncio.Lock` wakes waiters FIFO.** `_yield_slice`'s re-acquire therefore lands
    behind exactly the requests queued at that instant and ahead of any arriving later,
    so an ingest on a busy server is delayed by a bounded amount rather than starved.
    A priority queue would buy nothing here.
  * **`_waiters` needs no lock.** It is written only on the event-loop thread (every
    mutation is inside a coroutine) and read from the ingest's worker thread, where an
    int read is atomic under the GIL. A stale read costs at most one skipped or one
    extra yield, both harmless - the gate is an optimization, and correctness rests on
    the lock.
"""

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from src.config import GPU_WAIT_TIMEOUT_S
from src.logging_setup import get_logger

log = get_logger("gpu")


class GpuBusy(Exception):
    """Raised when a request waited past its timeout for the model. Maps to 503."""

    def __init__(self, waited_s: float):
        super().__init__(
            f"The model is busy (waited {waited_s:.0f}s). Retry shortly."
        )
        self.waited_s = waited_s
        # Always at least 1: a client told to retry after 0 seconds retries immediately -
        # the same reasoning as ratelimit._enforce's Retry-After.
        self.retry_after = max(1, int(waited_s))


class GpuArbiter:
    """Serializes GPU work, sheds load past a timeout, and lets a long ingest yield."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._waiters = 0     # requests *queued* for the lock, never its holder (see acquire)

    def contended(self) -> bool:
        """Whether anyone is queued for the lock (safe to call off-loop).

        Holders are excluded deliberately - `acquire` decrements on acquisition - which is
        what makes this the right question for the gate: an ingest that counted itself
        would yield to nobody on every batch.
        """
        return self._waiters > 0

    @asynccontextmanager
    async def acquire(self, timeout: float | None = None):
        """Hold the GPU for the duration of the block, or raise `GpuBusy`.

        `timeout` defaults to `GPU_WAIT_TIMEOUT_S`; 0 or None waits indefinitely. The
        waiter count is incremented before the wait and decremented however it ends, so
        a request that times out stops advertising itself and a running ingest stops
        yielding to it.
        """
        wait = GPU_WAIT_TIMEOUT_S if timeout is None else timeout
        self._waiters += 1
        try:
            await asyncio.wait_for(self._lock.acquire(), wait or None)
        except TimeoutError:
            self._waiters -= 1
            log.warning("shed request waiting for the model", extra={"waited_s": wait})
            raise GpuBusy(wait) from None
        except BaseException:
            self._waiters -= 1
            raise
        # Decremented on acquisition, not on release: past this point the request is
        # *holding* the lock, so counting it as a waiter would make a running ingest
        # yield to a request that already has what it wants.
        self._waiters -= 1
        try:
            yield
        finally:
            self._lock.release()

    async def run_exclusive(
        self, fn: Callable[..., Any], *args: Any, timeout: float | None = None,
    ) -> Any:
        """Run a blocking GPU function on a worker thread, holding the lock throughout.

        The lock is not released until `fn` has actually returned, even if this coroutine
        is cancelled first (a disconnecting client, a request timeout upstream). See the
        module docstring - a released lock over a live forward pass is the failure this
        exists to prevent, and it fails silently.
        """
        async with self.acquire(timeout):
            task = asyncio.create_task(asyncio.to_thread(fn, *args))
            try:
                return await asyncio.shield(task)
            finally:
                if not task.done():
                    await asyncio.shield(task)

    def thread_gate(self, loop: asyncio.AbstractEventLoop) -> Callable[[], None]:
        """A sync `gate()` for a worker thread already holding the lock (see `ingest.Gate`).

        Blocks its caller while a queued request takes the model, then returns once the
        lock is back. A no-op - and specifically, no cross-thread round trip - when
        nothing is waiting, which is the common case for every batch of an uncontended
        ingest.
        """
        def gate() -> None:
            if not self.contended():
                return
            asyncio.run_coroutine_threadsafe(self._yield_slice(), loop).result()

        return gate

    async def _yield_slice(self) -> None:
        """Hand the lock to the queued waiters, then take it back (FIFO; see docstring).

        The precondition is that the caller's request already holds the lock. The check
        below only catches the *unlocked* case and turns asyncio's bare "Lock is not
        acquired" into something that names the caller; it deliberately does not claim to
        catch the worse case - releasing a lock held by a *different* request - because
        `asyncio.Lock` tracks no owner and that is not detectable here. What actually
        prevents it is that `thread_gate` is only ever handed to a worker running inside
        `run_exclusive`/`acquire`.
        """
        if not self._lock.locked():
            raise RuntimeError("gate() called without holding the GPU lock")
        self._lock.release()
        await self._lock.acquire()
