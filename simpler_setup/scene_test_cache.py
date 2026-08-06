# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Persistent cache for compiled scene-test ``ChipCallable`` blobs."""

from __future__ import annotations

import contextlib
import ctypes
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

from .environment import PROJECT_ROOT

logger = logging.getLogger(__name__)

KERNEL_CACHE_DIR = PROJECT_ROOT / "build" / "cache" / "kernels"

_CACHE_VERSION = 1
_BINARY_FILE = "callable.bin"
_MANIFEST_FILE = "manifest.json"
_LOCK_DIR = ".locks"
_INCLUDE_RE = re.compile(rb"^\s*#\s*include\s*([<\"])([^>\"]+)[>\"]", re.MULTILINE)

# Entries are content-addressed, so a source change strands the old entry
# forever. Every hit refreshes its mtime, so this window is time-since-last-use
# and an entry still in service is never reclaimed regardless of its age.
_ENTRY_RETENTION_S = 14 * 24 * 3600


@cache
def _chip_callable_abi_token() -> str:
    """Return a fingerprint of the binding's serialized callable layout."""
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable  # noqa: PLC0415

    child = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT],
        binary=b"scene-test-cache-core-abi",
    )
    callable_obj = ChipCallable.build(
        signature=[ArgDirection.INOUT],
        func_name="scene_test_cache_abi",
        binary=b"scene-test-cache-chip-abi",
        children=[(17, child)],
        config_name="scene_test_cache_config",
    )
    raw = ctypes.string_at(int(callable_obj.buffer_ptr()), int(callable_obj.buffer_size()))
    return hashlib.sha256(raw).hexdigest()


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _stable_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (bool, int, float, str)):
        return enum_value
    raise TypeError(f"unsupported compile-cache key value: {type(value).__name__}")


def _resolve_include(name: str, quoted: bool, including_file: Path, include_dirs: tuple[Path, ...]) -> Path | None:
    candidates = ((including_file.parent,) if quoted else ()) + include_dirs
    for directory in candidates:
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


_scan_cache: dict[tuple[Path, int, int], tuple[bytes, tuple[tuple[str, bool], ...]]] = {}


def _scan_source(path: Path) -> tuple[bytes, tuple[tuple[str, bool], ...]]:
    """Return one file's digest and its ``#include`` directives as ``(name, quoted)``.

    Memoized on ``(path, mtime_ns, size)`` because the compilation units of one
    scene test share most of their include closure — the pto-isa headers alone
    are reached from every incore — and both the read and the digest are
    otherwise repeated once per unit.
    """
    try:
        stat = path.stat()
        token = (path, stat.st_mtime_ns, stat.st_size)
    except OSError:
        token = None
    if token is not None:
        memoized = _scan_cache.get(token)
        if memoized is not None:
            return memoized

    data = path.read_bytes()
    directives = []
    for match in _INCLUDE_RE.finditer(data):
        delimiter, raw_name = match.groups()
        try:
            name = raw_name.decode()
        except UnicodeDecodeError:
            continue
        directives.append((name, delimiter == b'"'))
    scanned = (hashlib.sha256(data).digest(), tuple(directives))
    if token is not None:
        _scan_cache[token] = scanned
    return scanned


def _source_closure(
    source: str | Path, include_dirs: Iterable[str | Path]
) -> tuple[Path, dict[Path, bytes], list[tuple[Path, str, Path]]]:
    """Return the root path, the digest of every file in its include closure, and the closure's edges.

    The closure holds only includes that resolve within ``include_dirs`` — or,
    for a quoted include, beside the including file — mirroring the search order
    the compiler applies to the same ``-I`` list. Headers the toolchain supplies
    from its own search path (ccec builtins, CANN headers under
    ``ASCEND_HOME_PATH``) resolve outside that list and are absent from the key;
    the compiler ``--version`` identity carried in the key metadata is what
    covers those. ``#if`` conditions are not evaluated, so the closure is an
    over-approximation of what any single build compiles.
    """
    roots = tuple(Path(directory).resolve() for directory in include_dirs if Path(directory).is_dir())
    source_path = Path(source).resolve()
    pending = [source_path]
    files: dict[Path, bytes] = {}
    edges = []
    while pending:
        path = pending.pop()
        if path in files:
            continue
        file_digest, directives = _scan_source(path)
        files[path] = file_digest
        for name, quoted in directives:
            dependency = _resolve_include(name, quoted, path, roots)
            if dependency is not None:
                edges.append((path, name, dependency))
                if dependency not in files:
                    pending.append(dependency)
    return source_path, files, edges


def compile_artifact_key(
    metadata: Mapping[str, Any], compilation_units: Iterable[tuple[str | Path, Iterable[str | Path]]]
) -> str:
    """Return a content key for callable metadata and source include closures."""
    digest = hashlib.sha256()
    abi_token = _chip_callable_abi_token().encode()
    digest.update(len(abi_token).to_bytes(8, "little"))
    digest.update(abi_token)
    encoded_metadata = json.dumps(_stable_value(metadata), sort_keys=True, separators=(",", ":")).encode()
    digest.update(len(encoded_metadata).to_bytes(8, "little"))
    digest.update(encoded_metadata)

    for source, include_dirs in compilation_units:
        source_path, file_digests, edges = _source_closure(source, include_dirs)
        digest.update(b"unit")
        digest.update(file_digests[source_path])
        for file_digest in sorted(file_digests.values()):
            digest.update(b"file")
            digest.update(file_digest)
        edge_records = sorted(
            (file_digests[parent], name.encode(), file_digests[dependency]) for parent, name, dependency in edges
        )
        for parent_digest, name, dependency_digest in edge_records:
            digest.update(b"edge")
            digest.update(parent_digest)
            digest.update(len(name).to_bytes(8, "little"))
            digest.update(name)
            digest.update(dependency_digest)
    return digest.hexdigest()


def _entry_paths(key: str) -> tuple[Path, Path, Path]:
    entry = KERNEL_CACHE_DIR / key
    return entry, entry / _BINARY_FILE, entry / _MANIFEST_FILE


def _touch(key: str) -> None:
    entry, _binary_path, _manifest_path = _entry_paths(key)
    with contextlib.suppress(OSError):
        os.utime(entry)


def _load(key: str):
    _entry, binary_path, manifest_path = _entry_paths(key)
    try:
        manifest = json.loads(manifest_path.read_text())
        raw = binary_path.read_bytes()
    except (OSError, json.JSONDecodeError):
        return None
    if manifest != {
        "version": _CACHE_VERSION,
        "key": key,
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }:
        return None

    from simpler.task_interface import ChipCallable  # noqa: PLC0415

    try:
        callable_obj = ChipCallable.from_bytes(raw)
        callable_obj.buffer_size()
    except (RuntimeError, ValueError):
        return None
    return callable_obj


def _publish(key: str, callable_obj) -> None:
    entry, binary_path, manifest_path = _entry_paths(key)
    entry.mkdir(parents=True, exist_ok=True)
    raw = ctypes.string_at(int(callable_obj.buffer_ptr()), int(callable_obj.buffer_size()))
    manifest = {
        "version": _CACHE_VERSION,
        "key": key,
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    suffix = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    binary_tmp = entry / f"{_BINARY_FILE}{suffix}"
    manifest_tmp = entry / f"{_MANIFEST_FILE}{suffix}"
    try:
        binary_tmp.write_bytes(raw)
        manifest_tmp.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        os.replace(binary_tmp, binary_path)
        os.replace(manifest_tmp, manifest_path)
    finally:
        binary_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


_pruned_dirs: set[Path] = set()


def prune_stale_entries() -> int:
    """Delete cache entries and lock files untouched for ``_ENTRY_RETENTION_S``.

    Runs at most once per cache directory per process, after a publish has
    already proved the directory writable. Returns the number of entries removed.
    """
    if KERNEL_CACHE_DIR in _pruned_dirs:
        return 0
    _pruned_dirs.add(KERNEL_CACHE_DIR)

    cutoff = time.time() - _ENTRY_RETENTION_S
    removed = 0
    try:
        entries = list(KERNEL_CACHE_DIR.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if entry.name == _LOCK_DIR:
            continue
        try:
            if entry.stat().st_mtime >= cutoff:
                continue
        except OSError:
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                entry.unlink()
        removed += 1

    for lock_path in (KERNEL_CACHE_DIR / _LOCK_DIR).glob("*.lock"):
        with contextlib.suppress(OSError):
            if lock_path.stat().st_mtime < cutoff:
                lock_path.unlink()
    if removed:
        logger.info("[SceneTestCache] pruned %d stale entr(ies) from %s", removed, KERNEL_CACHE_DIR)
    return removed


@contextlib.contextmanager
def _entry_lock(key: str):
    """Hold an exclusive lock for ``key``, or yield ``None`` when the cache directory is unusable.

    A read-only install (a wheel whose ``PROJECT_ROOT`` is inside
    ``site-packages``) or a filesystem without ``flock`` yields ``None``, which
    degrades the caller to plain compilation instead of failing a scene test
    that used to pass.
    """
    lock_file = None
    try:
        lock_dir = KERNEL_CACHE_DIR / _LOCK_DIR
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file = (lock_dir / f"{key}.lock").open("w")
        fcntl.flock(lock_file, fcntl.LOCK_EX)
    except OSError as error:
        if lock_file is not None:
            lock_file.close()
            lock_file = None
        logger.warning(
            "[SceneTestCache] disabled: %s is not usable (%s: %s); compiling without a cache",
            KERNEL_CACHE_DIR,
            type(error).__name__,
            error,
        )
    try:
        yield lock_file
    finally:
        if lock_file is not None:
            lock_file.close()


def get_or_compile(key: str, compile_fn: Callable[[], Any]):
    """Load one ``ChipCallable`` or compile and atomically publish it."""
    cached = _load(key)
    if cached is not None:
        _touch(key)
        logger.info("[SceneTestCache] hit: %s", key[:12])
        return cached

    with _entry_lock(key) as lock_file:
        if lock_file is None:
            return compile_fn()
        cached = _load(key)
        if cached is not None:
            _touch(key)
            logger.info("[SceneTestCache] hit after wait: %s", key[:12])
            return cached
        logger.info("[SceneTestCache] miss: %s", key[:12])
        callable_obj = compile_fn()
        try:
            _publish(key, callable_obj)
        except OSError as error:
            logger.warning(
                "[SceneTestCache] publish failed for %s (%s: %s); artifact not cached",
                key[:12],
                type(error).__name__,
                error,
            )
        else:
            prune_stale_entries()
        return callable_obj
