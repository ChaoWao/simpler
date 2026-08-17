# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resolve the managed PTO-ISA checkout under ``PROJECT_ROOT/build/pto-isa``.

``pto_isa.pin`` selects a GitHub commit. When GitHub acquisition fails, the
resolver uses Youhezhen GitCode ``master`` and records its actual commit because
the two repositories use different commit identities.
"""

import fcntl
import json
import logging
import re
import shutil
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from .environment import PROJECT_ROOT

# A fresh clone is only needed when no usable local checkout exists; when it is,
# guard the complete GitHub acquisition (clone plus landing on the pin) against
# transient failures before falling back to the Youhezhen GitCode master.
_CLONE_ATTEMPTS = 3
_CLONE_RETRY_BACKOFF_S = 2

logger = logging.getLogger(__name__)

_PTO_ISA_GITHUB_HTTPS = "https://github.com/hw-native-sys/pto-isa.git"
_PTO_ISA_GITCODE_HTTPS = "https://gitcode.com/Youhezhen/pto-isa.git"
_PTO_ISA_GITCODE_BRANCH = "master"
_PTO_ISA_FALLBACK_STATE_FILE = ".pto-isa-fallback.json"
_PTO_ISA_PIN_RE = re.compile(r"^[0-9a-fA-F]{40}$")
PTO_ISA_PIN_FILE = "pto_isa.pin"
PTO_ISA_BUILD_METADATA = "pto_isa_build.json"


def _fallback_state_path(clone_path: Path) -> Path:
    return clone_path.parent / _PTO_ISA_FALLBACK_STATE_FILE


def _read_fallback_state(clone_path: Path) -> Optional[dict]:
    try:
        payload = json.loads(_fallback_state_path(clone_path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_fallback_state(clone_path: Path, required_commit: str, actual_commit: str) -> None:
    state_path = _fallback_state_path(clone_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "required_commit_from_pin": required_commit,
        "actual_commit": actual_commit,
        "remote": _PTO_ISA_GITCODE_HTTPS,
        "branch": _PTO_ISA_GITCODE_BRANCH,
    }
    tmp_path = state_path.with_name(f"{state_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(state_path)


def _clear_fallback_state(clone_path: Path) -> None:
    _fallback_state_path(clone_path).unlink(missing_ok=True)


def _fallback_checkout_matches(
    clone_path: Path,
    required_commit: str,
    actual_commit: str,
) -> bool:
    state = _read_fallback_state(clone_path)
    if state is None:
        return False
    return (
        state.get("required_commit_from_pin") == required_commit
        and state.get("actual_commit") == actual_commit
        and state.get("remote") == _PTO_ISA_GITCODE_HTTPS
        and state.get("branch") == _PTO_ISA_GITCODE_BRANCH
    )


def _checkout_matches_resolution(clone_path: Path, required_commit: str, actual_commit: str) -> bool:
    return actual_commit == required_commit or _fallback_checkout_matches(
        clone_path,
        required_commit,
        actual_commit,
    )


def read_pto_isa_pin(pin_path: Optional[Path] = None) -> str:
    """Read and validate the repository PTO-ISA pin."""
    path = pin_path or (PROJECT_ROOT / PTO_ISA_PIN_FILE)
    try:
        value = path.read_text().strip()
    except FileNotFoundError as e:
        raise RuntimeError(f"PTO-ISA pin not found at {path}") from e
    except OSError as e:
        raise RuntimeError(f"Failed to read PTO-ISA pin at {path}: {e}") from e

    if not value:
        raise RuntimeError(f"Invalid PTO-ISA pin at {path}: expected a 40-character hex SHA, got an empty file")
    if not _PTO_ISA_PIN_RE.fullmatch(value):
        raise RuntimeError(f"Invalid PTO-ISA pin at {path}: expected a 40-character hex SHA, got {value!r}")
    return value.lower()


def get_pto_isa_head(pto_isa_root: str) -> str:
    """Return the full git HEAD SHA for a PTO-ISA checkout, or empty if unknown."""
    try:
        result = _run_git_resilient(["rev-parse", "HEAD"], cwd=Path(pto_isa_root), timeout=5)
        return result.stdout.strip().lower() if result.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def pto_isa_build_metadata_path(lib_dir: Path) -> Path:
    """Return the build metadata path under build/lib."""
    return lib_dir / PTO_ISA_BUILD_METADATA


def pto_isa_runtime_artifact_key(arch: str, variant: str, runtime_name: str) -> str:
    """Return the metadata key for one runtime artifact set."""
    return f"{arch}/{variant}/{runtime_name}"


def _metadata_commit(payload: dict) -> str:
    return (
        str(
            payload.get("required_commit_from_pin")
            or payload.get("actual_checkout_commit")
            or payload.get("pto_isa_commit", "")
        )
        .strip()
        .lower()
    )


def _metadata_actual_commit(payload: dict) -> str:
    return (
        str(
            payload.get("actual_checkout_commit")
            or payload.get("required_commit_from_pin")
            or payload.get("pto_isa_commit", "")
        )
        .strip()
        .lower()
    )


def _metadata_entry(required_commit: str, actual_commit: str, pto_isa_root: str) -> dict:
    return {
        "required_commit_from_pin": required_commit,
        "actual_checkout_commit": actual_commit,
        "pin_file": str((PROJECT_ROOT / PTO_ISA_PIN_FILE).resolve()),
        "checkout_path": str(Path(pto_isa_root).resolve()),
    }


def write_pto_isa_build_metadata(
    lib_dir: Path,
    pto_isa_root: str,
    runtime_keys: Iterable[str] = (),
) -> None:
    """Record the requested and actual PTO-ISA revisions used by runtime binaries."""
    required_commit = read_pto_isa_pin()
    actual_commit = get_pto_isa_head(pto_isa_root)
    if not actual_commit:
        raise RuntimeError(
            "Cannot record PTO-ISA build revision: "
            f"{pto_isa_root} is not a git checkout or git HEAD is unavailable. "
            "Building PTO-ISA-embedding onboard runtimes requires the managed build/pto-isa checkout."
        )
    if actual_commit != required_commit and not _fallback_checkout_matches(
        Path(pto_isa_root), required_commit, actual_commit
    ):
        raise RuntimeError(
            "PTO-ISA checkout mismatch while recording runtime build metadata: "
            f"pto_isa.pin requires {required_commit}, but {pto_isa_root} is at {actual_commit}."
        )

    keys = [runtime_keys] if isinstance(runtime_keys, str) else sorted(dict.fromkeys(runtime_keys))
    entry = _metadata_entry(required_commit, actual_commit, pto_isa_root)
    lib_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lib_dir / ".pto_isa_build.lock"
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        existing = read_pto_isa_build_metadata(lib_dir) or {}
        runtime_artifacts = {}
        if existing.get("schema_version") == 3 and isinstance(existing.get("runtime_artifacts"), dict):
            runtime_artifacts.update(existing["runtime_artifacts"])
        for key in keys:
            runtime_artifacts[key] = dict(entry)

        metadata = {"schema_version": 3, **entry, "runtime_artifacts": runtime_artifacts}
        metadata_path = pto_isa_build_metadata_path(lib_dir)
        tmp_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
        tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(metadata_path)


def read_pto_isa_build_metadata(lib_dir: Path) -> Optional[dict]:
    """Read installed runtime PTO-ISA metadata, if present."""
    metadata_path = pto_isa_build_metadata_path(lib_dir)
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"Invalid PTO-ISA build metadata at {metadata_path}: {e}") from e
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid PTO-ISA build metadata at {metadata_path}: expected JSON object")
    return payload


def validate_runtime_pto_isa_current_pin(lib_dir: Path, runtime_key: Optional[str] = None) -> None:
    """Raise when pre-built runtime binaries do not match the current resolution.

    Called only for platforms that embed PTO-ISA headers into host runtimes
    (see :func:`simpler_setup.runtime_builder.platform_embeds_pto_isa`). Missing
    metadata is a hard failure — silent skip would let a pin bump load a stale
    ``host_runtime.so``.
    """
    required_commit = read_pto_isa_pin()
    metadata_path = pto_isa_build_metadata_path(lib_dir)
    metadata = read_pto_isa_build_metadata(lib_dir)
    if metadata is None:
        raise RuntimeError(
            "Missing PTO-ISA build metadata: "
            f"current pto_isa.pin requires {required_commit}, but {metadata_path} "
            "is absent.\n"
            "Reinstall simpler or rebuild this runtime so build/lib records the pin."
        )

    if metadata.get("schema_version") == 3 and runtime_key is not None:
        artifacts = metadata.get("runtime_artifacts")
        if not isinstance(artifacts, dict):
            raise RuntimeError(f"Invalid PTO-ISA build metadata at {metadata_path}: expected runtime_artifacts object")
        artifact = artifacts.get(runtime_key)
        if artifact is None:
            raise RuntimeError(
                "Stale PTO-ISA runtime binaries: current pto_isa.pin requires "
                f"{required_commit}, but {metadata_path} has no entry for runtime {runtime_key!r}.\n"
                "Reinstall simpler or rebuild this runtime so build/lib matches pto_isa.pin."
            )
        if not isinstance(artifact, dict):
            raise RuntimeError(
                f"Invalid PTO-ISA build metadata at {metadata_path}: "
                f"runtime_artifacts[{runtime_key!r}] must be a JSON object"
            )
        build_commit = _metadata_commit(artifact)
        actual_build_commit = _metadata_actual_commit(artifact)
    else:
        build_commit = _metadata_commit(metadata)
        actual_build_commit = _metadata_actual_commit(metadata)
    if build_commit and build_commit != required_commit:
        raise RuntimeError(
            "Stale PTO-ISA runtime binaries: current pto_isa.pin requires "
            f"{required_commit}, but installed runtimes were requested for {build_commit}.\n"
            f"Build metadata: {metadata_path}\n"
            "Reinstall simpler or rebuild runtimes so build/lib matches pto_isa.pin."
        )

    resolved_commit = get_pto_isa_resolved_commit()
    if not actual_build_commit or actual_build_commit == resolved_commit:
        return

    raise RuntimeError(
        "Stale PTO-ISA runtime binaries: the current PTO-ISA checkout resolves "
        f"pto_isa.pin {required_commit} to {resolved_commit}, but installed runtimes "
        f"were built with {actual_build_commit}.\n"
        f"Build metadata: {metadata_path}\n"
        "Reinstall simpler or rebuild runtimes so build/lib matches the resolved checkout."
    )


def get_pto_isa_clone_path() -> Path:
    """Managed auto-clone target for PTO-ISA, anchored to PROJECT_ROOT."""
    return PROJECT_ROOT / "build" / "pto-isa"


def get_pto_isa_resolved_commit() -> str:
    """Return the commit whose headers should key build artifacts.

    The requested pin remains the default. Once a verified Youhezhen/master
    fallback exists for that pin, its actual HEAD keys runtime build caches and
    compile-time provenance instead.
    """
    required_commit = read_pto_isa_pin()
    clone_path = get_pto_isa_clone_path()
    actual_commit = get_pto_isa_head(str(clone_path))
    if _checkout_matches_resolution(clone_path, required_commit, actual_commit):
        return actual_commit
    return required_commit


def _is_cloned(path: Path) -> bool:
    """Return True if `path` looks like a valid PTO-ISA clone (has include/)."""
    return (path / "include").is_dir()


def _is_git_available() -> bool:
    try:
        result = subprocess.run(["git", "--version"], check=False, capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_git(
    args: list,
    cwd: Optional[Path] = None,
    timeout: int = 30,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run a git subcommand and capture stdout/stderr as text."""
    return subprocess.run(
        ["git"] + args,
        check=check,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
    )


def _is_dubious_ownership_error(stderr: str) -> bool:
    return "detected dubious ownership" in stderr and "safe.directory" in stderr


def _run_git_with_safe_directory(
    args: list,
    cwd: Path,
    timeout: int = 30,
    check: bool = False,
) -> subprocess.CompletedProcess:
    return _run_git(["-c", f"safe.directory={cwd.resolve()}", *args], cwd=cwd, timeout=timeout, check=check)


def _run_git_resilient(
    args: list,
    cwd: Path,
    timeout: int = 30,
    check: bool = False,
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    """Run git, retrying dubious-ownership failures with per-command safe.directory."""
    result = _run_git(args, cwd=cwd, timeout=timeout, check=False)
    if result.returncode != 0 and _is_dubious_ownership_error(result.stderr):
        if verbose:
            logger.info(f"Using pto-isa safe.directory for {cwd}")
        result = _run_git_with_safe_directory(args, cwd=cwd, timeout=timeout, check=False)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, ["git", *args], result.stdout, result.stderr)
    return result


def _remove_clone(target: Path, verbose: bool) -> None:
    """Unconditionally remove `target` so a fresh managed clone can replace it.

    This removes even a *valid* clone when it is at the wrong (or a dirty)
    revision, and also clears partial clones between source attempts. Handles a
    directory, a plain file, or a (possibly broken) symlink.
    """
    _clear_fallback_state(target)
    if not (target.exists() or target.is_symlink()):
        return
    if verbose:
        logger.info(f"Removing pto-isa checkout at {target} to resolve the requested revision afresh")
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target, ignore_errors=True)
    else:
        target.unlink(missing_ok=True)


def _land_on_commit(clone_path: Path, commit: str, verbose: bool) -> bool:
    """Force-detach-checkout a freshly cloned tree onto `commit`. False on failure.

    `--force` is load-bearing, not defensive: pto-isa's default branch carries
    paths that differ only in case (e.g. ``docs/isa/TADDDEQRELU.md`` vs
    ``docs/isa/TAddDeqRelu.md``). On a case-insensitive filesystem (macOS CI)
    they collide onto one inode, so even a *fresh* clone's working tree reports
    them as modified, and a plain checkout to the pin aborts with "local changes
    would be overwritten." Forcing discards that pseudo-dirt and lands exactly on
    the pin. A full clone already carries every branch's history, but keep a
    fetch fallback in case the pin is not reachable from the default fetch.
    Fetch the exact pin first so a stale remote-ref advertisement cannot make a
    newly-pushed commit invisible; fall back to the advertised refs for servers
    that do not allow fetching a reachable object by SHA.
    """
    try:
        result = _run_git_resilient(
            ["checkout", "--detach", "--force", commit], cwd=clone_path, timeout=30, verbose=verbose
        )
        if result.returncode != 0:
            if verbose:
                logger.info(f"pto-isa commit {commit} missing locally, fetching the exact pin from origin...")
            fetch_result = _run_git_resilient(
                ["fetch", "--no-tags", "origin", commit],
                cwd=clone_path,
                timeout=120,
                verbose=verbose,
            )
            if fetch_result.returncode != 0:
                if verbose:
                    logger.info("Exact PTO-ISA pin fetch failed; falling back to advertised origin refs...")
                _run_git_resilient(["fetch", "origin"], cwd=clone_path, timeout=120, check=True, verbose=verbose)
            _run_git_resilient(
                ["checkout", "--detach", "--force", commit], cwd=clone_path, timeout=30, check=True, verbose=verbose
            )
        actual = get_pto_isa_head(str(clone_path))
        if actual != commit:
            logger.warning(f"pto-isa checkout verification failed: expected {commit}, got {actual or '<unknown>'}")
            return False
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to check out pto-isa commit {commit}: {e.stderr if hasattr(e, 'stderr') else e}")
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Unexpected error checking out pto-isa commit {commit}: {e}")
        return False


def _land_on_branch_head(clone_path: Path, branch: str, verbose: bool) -> bool:
    """Force-detach-checkout the branch tip fetched by a fresh clone."""
    remote_ref = f"refs/remotes/origin/{branch}"
    try:
        _run_git_resilient(
            ["checkout", "--detach", "--force", remote_ref],
            cwd=clone_path,
            timeout=30,
            check=True,
            verbose=verbose,
        )
        branch_result = _run_git_resilient(
            ["rev-parse", remote_ref],
            cwd=clone_path,
            timeout=30,
            check=True,
            verbose=verbose,
        )
        expected = branch_result.stdout.strip().lower()
        actual = get_pto_isa_head(str(clone_path))
        if not expected or actual != expected:
            logger.warning(
                f"pto-isa branch checkout verification failed: expected {remote_ref} "
                f"at {expected or '<unknown>'}, got {actual or '<unknown>'}"
            )
            return False
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to check out pto-isa branch {branch}: {e.stderr if hasattr(e, 'stderr') else e}")
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Unexpected error checking out pto-isa branch {branch}: {e}")
        return False


def _is_pristine_at_commit(clone_path: Path, commit: str, verbose: bool) -> bool:
    """Return whether the checkout is clean at the pin or its recorded fallback."""
    actual_commit = get_pto_isa_head(str(clone_path))
    if not _checkout_matches_resolution(clone_path, commit, actual_commit):
        return False
    try:
        result = _run_git_resilient(["status", "--porcelain"], cwd=clone_path, timeout=30, verbose=verbose)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to check pto-isa checkout cleanliness: {e.stderr if hasattr(e, 'stderr') else e}")
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Unexpected error checking pto-isa checkout cleanliness: {e}")
        return False
    return result.returncode == 0 and not result.stdout.strip()


def _clone_from_remote(
    target: Path,
    commit: str,
    remote: str,
    attempts: int,
    verbose: bool,
    branch_head: Optional[str] = None,
) -> bool:
    """Try complete pinned or branch-tip acquisitions from one remote."""
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            time.sleep(_CLONE_RETRY_BACKOFF_S * (attempt - 1))
        _remove_clone(target, verbose)
        logger.info(
            f"Cloning pto-isa from {remote} to {target} at {branch_head or commit} "
            f"(attempt {attempt}/{attempts}, may take up to a minute)..."
        )

        failure = ""
        try:
            # --no-checkout: never materialize the default branch. Its working
            # tree carries case-colliding docs/isa/TADDDEQRELU* paths; only lay
            # down the selected commit or branch-tip tree below.
            clone_args = ["clone", "--no-checkout"]
            if branch_head is not None:
                clone_args.extend(["--branch", branch_head])
            clone_args.extend([remote, str(target)])
            result = _run_git(clone_args, timeout=300)
            if result.returncode != 0:
                failure = result.stderr.strip() or f"git clone exited with status {result.returncode}"
            elif (
                _land_on_branch_head(target, branch_head, verbose=verbose)
                if branch_head is not None
                else _land_on_commit(target, commit, verbose=verbose)
            ):
                if verbose:
                    revision = f"{branch_head} HEAD" if branch_head is not None else commit
                    logger.info(f"pto-isa cloned from {remote} at {revision}: {target}")
                return True
            else:
                revision = f"branch {branch_head}" if branch_head is not None else f"commit {commit}"
                failure = f"clone succeeded but {revision} could not be checked out"
        except subprocess.TimeoutExpired:
            failure = "clone operation timed out"
        except Exception as e:  # noqa: BLE001
            failure = str(e)

        if verbose:
            logger.warning(f"pto-isa acquisition from {remote} attempt {attempt}/{attempts} failed:\n{failure}")
        # A successful clone that could not land on the selected revision is unusable;
        # remove it just like a partial clone before retrying or changing source.
        _remove_clone(target, verbose)

    return False


def _clone(target: Path, commit: str, verbose: bool) -> bool:
    """Clone the GitHub commit, falling back to Youhezhen GitCode ``master``."""
    if not _is_git_available():
        if verbose:
            logger.warning("git command not available, cannot clone pto-isa")
        return False

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if verbose:
            logger.warning(f"Failed to create clone parent dir: {e}")
        return False

    try:
        if _clone_from_remote(
            target,
            commit,
            _PTO_ISA_GITHUB_HTTPS,
            attempts=_CLONE_ATTEMPTS,
            verbose=verbose,
        ):
            return True

        logger.warning(
            f"GitHub could not provide PTO-ISA commit {commit} after {_CLONE_ATTEMPTS} attempts; "
            f"falling back to {_PTO_ISA_GITCODE_HTTPS} branch {_PTO_ISA_GITCODE_BRANCH}"
        )
        if not _clone_from_remote(
            target,
            commit,
            _PTO_ISA_GITCODE_HTTPS,
            attempts=1,
            verbose=verbose,
            branch_head=_PTO_ISA_GITCODE_BRANCH,
        ):
            return False
        actual_commit = get_pto_isa_head(str(target))
        if not actual_commit:
            _remove_clone(target, verbose)
            return False
        _write_fallback_state(target, commit, actual_commit)
        return True
    except Exception as e:  # noqa: BLE001
        if verbose:
            logger.warning(f"Failed to clone pto-isa: {e}")
        _remove_clone(target, verbose)
        return False


def ensure_pto_isa_root(verbose: bool = False) -> str:
    """Resolve the managed PTO-ISA checkout. Return its absolute path."""
    required_commit = read_pto_isa_pin()
    clone_path = get_pto_isa_clone_path()
    lock_path = clone_path.parent / ".pto-isa.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        resolved = _ensure_locked(clone_path, required_commit=required_commit, verbose=verbose)

    if resolved is None:
        raise OSError(
            f"PTO-ISA not available.\n"
            f"  The managed checkout must live at {clone_path} and resolve {PROJECT_ROOT / PTO_ISA_PIN_FILE}.\n"
            f"  If auto-clone failed, manually clone {_PTO_ISA_GITHUB_HTTPS} there and check out\n"
            f"  the requested commit {required_commit}.\n"
            f"  Automatic fallback source:\n"
            f"    {_PTO_ISA_GITCODE_HTTPS} branch {_PTO_ISA_GITCODE_BRANCH}"
        )
    return resolved


def _ensure_locked(clone_path: Path, required_commit: str, verbose: bool) -> Optional[str]:
    """Inner logic executed while holding the file lock."""
    # Only a clean pin or matching recorded fallback is reusable.
    if _is_cloned(clone_path) and _is_pristine_at_commit(clone_path, required_commit, verbose=verbose):
        return str(clone_path.resolve())

    # Re-clone instead of overwriting a dirty or unrelated checkout.
    if not _clone(clone_path, required_commit, verbose=verbose):
        # A parallel process holding a separate lock may have prepared the
        # checkout concurrently; accept only an exact pin or recorded fallback.
        actual_commit = get_pto_isa_head(str(clone_path))
        if not (_is_cloned(clone_path) and _checkout_matches_resolution(clone_path, required_commit, actual_commit)):
            return None
        if verbose:
            logger.info("pto-isa prepared at the requested resolution by another process")

    if not _is_cloned(clone_path):
        if verbose:
            logger.warning(f"pto-isa path exists but missing include directory: {clone_path / 'include'}")
        return None

    actual_commit = get_pto_isa_head(str(clone_path))
    if not _checkout_matches_resolution(clone_path, required_commit, actual_commit):
        if verbose:
            logger.warning(
                "Fresh pto-isa checkout does not match the requested resolution: "
                f"pin {required_commit}, HEAD {actual_commit or '<unknown>'}"
            )
        _remove_clone(clone_path, verbose)
        return None

    return str(clone_path.resolve())
