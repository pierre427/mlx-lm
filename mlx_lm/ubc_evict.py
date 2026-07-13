# Copyright © 2026 raullenchai and the Rapid-MLX contributors (original work)
# Copyright © 2026 Pierre Lamy (mlx-uag port and adaptation)
# SPDX-License-Identifier: Apache-2.0
"""macOS Unified Buffer Cache (UBC) eviction helper.

Ported and adapted from ``raullenchai/Rapid-MLX`` (`vllm_mlx/runtime/ubc_evict.py`,
Apache-2.0) for measurement in the mlx-uag tree. See
``wiki/docs/research/rapid-mlx-competitive-2026-07-13.md`` for provenance.

Background
----------
On macOS (Darwin), ``mmap(MAP_SHARED, PROT_READ)`` pages of a regular file
remain resident in the Unified Buffer Cache (UBC) after the caller ``munmap``'s
and ``close``'s the file. ``mlx.core.load`` mmaps each safetensors shard; after
``mx.eval`` materialises the tensors, the file's UBC mirror is a pure shadow of
data MLX now holds in its own buffers. Explicitly evicting that mirror can lower
post-load steady-state footprint, freeing UMA for KV cache / longer context.

Because the safetensors file stays on disk unchanged, eviction can never change
the *values* read back — worst case is a page-fault re-read of identical bytes.
The open empirical questions (answered by the test harness, not by this module)
are whether the RSS drop is real on top of macOS's own lazy UBC purging, and
whether it costs decode throughput via re-paging of any mmap-aliased weights.

Mechanism
---------
``msync(addr, len, MS_INVALIDATE)`` on a fresh ``MAP_SHARED|PROT_READ`` mapping
is the documented Darwin path that flushes a UBC mirror (XNU
``vm_object_deactivate_pages(..., kill_pages=TRUE, ...)``). For a read-only
mapping there is no dirty data to write back, so the call is pure eviction — no
I/O. ``madvise(MADV_DONTNEED)`` is advisory and does NOT release file-backed
pages; ``fcntl(F_NOCACHE)`` only affects future reads — both are the wrong tool.

The helper is a no-op on non-Darwin platforms, never raises, and exposes a
process-monotonic byte counter for logging.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
import threading
import time
from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Darwin mman.h constants — stable since 4.4 BSD.
_PROT_READ: int = 0x01
_MAP_SHARED: int = 0x0001
_MS_INVALIDATE: int = 0x0002

# Darwin mmap returns MAP_FAILED == (void *)-1 on failure.
_MMAP_FAILED: int = ctypes.c_void_p(-1).value

_libc_lock = threading.Lock()
_libc: ctypes.CDLL | None = None


def _get_libc() -> ctypes.CDLL | None:
    """Return the cached libc handle with the three syscalls typed, or None."""
    global _libc
    if sys.platform != "darwin":
        return None
    if _libc is not None:
        return _libc
    with _libc_lock:
        if _libc is not None:
            return _libc
        try:
            libname = ctypes.util.find_library("c") or "libSystem.dylib"
            lib = ctypes.CDLL(libname, use_errno=True)
            lib.mmap.argtypes = [
                ctypes.c_void_p,  # addr
                ctypes.c_size_t,  # length
                ctypes.c_int,  # prot
                ctypes.c_int,  # flags
                ctypes.c_int,  # fd
                ctypes.c_longlong,  # off_t (Darwin: 64-bit)
            ]
            lib.mmap.restype = ctypes.c_void_p
            lib.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            lib.msync.restype = ctypes.c_int
            lib.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            lib.munmap.restype = ctypes.c_int
            _libc = lib
        except OSError as e:  # pragma: no cover — defensive
            logger.debug("ubc_evict: libc load failed: %s", e)
            _libc = None
    return _libc


_counter_lock = threading.Lock()
_ubc_evicted_bytes_total: int = 0
_ubc_evict_calls_total: int = 0
_ubc_evict_failed_total: int = 0


def _bump_counter(evicted: int, *, failed: bool) -> None:
    global _ubc_evicted_bytes_total, _ubc_evict_calls_total, _ubc_evict_failed_total
    with _counter_lock:
        _ubc_evict_calls_total += 1
        if failed:
            _ubc_evict_failed_total += 1
        elif evicted > 0:
            _ubc_evicted_bytes_total += int(evicted)


def ubc_evict(path: str) -> int:
    """Evict the UBC mirror of ``path`` via ``msync(MS_INVALIDATE)``.

    Returns the file size on success (upper bound on bytes asked to discard),
    0 in every error path and on non-Darwin platforms. Never raises.
    """
    if sys.platform != "darwin":
        logger.debug("ubc_evict no-op on %s", sys.platform)
        _bump_counter(0, failed=False)
        return 0

    libc = _get_libc()
    if libc is None:
        logger.debug("ubc_evict: libc unavailable, no-op")
        _bump_counter(0, failed=True)
        return 0

    try:
        size = os.path.getsize(path)
    except OSError as e:
        logger.warning("ubc_evict: stat %s failed: %s", path, e)
        _bump_counter(0, failed=True)
        return 0
    if size <= 0:
        logger.debug("ubc_evict: %s is empty, no-op", path)
        _bump_counter(0, failed=False)
        return 0

    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as e:
        logger.warning("ubc_evict: open %s failed: %s", path, e)
        _bump_counter(0, failed=True)
        return 0

    try:
        ctypes.set_errno(0)
        addr = libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        if addr is None or addr == 0 or addr == _MMAP_FAILED:
            err = ctypes.get_errno()
            logger.warning(
                "ubc_evict: mmap %s failed errno=%d (%s)",
                path,
                err,
                os.strerror(err) if err else "unknown",
            )
            _bump_counter(0, failed=True)
            return 0
        msync_ok = False
        munmap_ok = False
        try:
            ctypes.set_errno(0)
            rc = libc.msync(addr, size, _MS_INVALIDATE)
            if rc != 0:
                err = ctypes.get_errno()
                logger.warning(
                    "ubc_evict: msync(MS_INVALIDATE) %s rc=%d errno=%d (%s)",
                    path,
                    rc,
                    err,
                    os.strerror(err) if err else "unknown",
                )
            else:
                msync_ok = True
        finally:
            # Always release the mapping; a munmap failure means we must NOT
            # report success (false metric + leaked GB-sized mapping).
            ctypes.set_errno(0)
            munmap_rc = libc.munmap(addr, size)
            if munmap_rc != 0:
                err = ctypes.get_errno()
                logger.warning(
                    "ubc_evict: munmap %s rc=%d errno=%d (%s)",
                    path,
                    munmap_rc,
                    err,
                    os.strerror(err) if err else "unknown",
                )
            else:
                munmap_ok = True
    finally:
        try:
            os.close(fd)
        except OSError:  # pragma: no cover
            pass

    if not (msync_ok and munmap_ok):
        _bump_counter(0, failed=True)
        return 0
    _bump_counter(size, failed=False)
    return size


def ubc_evict_paths(paths: Iterable[str]) -> int:
    """Evict each path; aggregate bytes evicted; never raise."""
    if sys.platform != "darwin":
        logger.debug("ubc_evict_paths no-op on %s", sys.platform)
        return 0
    total = 0
    t0 = time.monotonic()
    for p in paths:
        bytes_evicted = ubc_evict(str(p))
        if bytes_evicted > 0:
            logger.info(
                "ubc_evict: evicted %.1f MB from UBC for %s",
                bytes_evicted / (1024 * 1024),
                p,
            )
        total += bytes_evicted
    if total > 0:
        logger.info(
            "ubc_evict: pass complete total_mb=%.1f elapsed_s=%.2f",
            total / (1024 * 1024),
            time.monotonic() - t0,
        )
    return total


def snapshot() -> dict[str, int]:
    """Return a thread-safe snapshot of the UBC counters."""
    with _counter_lock:
        return {
            "ubc_evicted_bytes_total": _ubc_evicted_bytes_total,
            "ubc_evict_calls_total": _ubc_evict_calls_total,
            "ubc_evict_failed_total": _ubc_evict_failed_total,
        }


def reset_for_tests() -> None:
    """Test-only: zero the counters between cases."""
    global _ubc_evicted_bytes_total, _ubc_evict_calls_total, _ubc_evict_failed_total
    with _counter_lock:
        _ubc_evicted_bytes_total = 0
        _ubc_evict_calls_total = 0
        _ubc_evict_failed_total = 0


__all__ = ["ubc_evict", "ubc_evict_paths", "snapshot", "reset_for_tests"]
