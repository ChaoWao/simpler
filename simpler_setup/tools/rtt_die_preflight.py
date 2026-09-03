#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full A5 AICPU affinity preflight: orch via atomic-flag handshake, sched via COND die scores."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from simpler_setup.environment import PROJECT_ROOT

SCHEDULER_COUNT = 4
PROBE_SCHEMA_VERSION = 3
PLAN_SCHEMA_VERSION = 3
MEASUREMENT_METHOD = "atomic-flag-orch+cond-die-v1"
PLAN_SOURCE_POOL_TOO_SMALL = "pool-too-small-contiguous"
PLAN_SOURCE_PROBE_FAILED = "probe-failed-contiguous"
PLAN_SOURCE_MANUAL = "manual"
PLAN_SOURCE_AUTO_FIRST_RUN = "auto-first-run"
# Phys pick die0,die1,die1,die0 → pack to logical [P0,P3,P1,P2] so S0/S1→die0, S2/S3→die1.
PHYS_PICK_DIES = (0, 1, 1, 0)
PROBE_SAMPLES_PER_CORE = 100
RELATIVE_PLAN = Path("build/config/aicpu_affinity_plan.json")
RELATIVE_PLAN_LEGACY = Path("build/config/aicpu_rtt_die_plan.json")

_TOOL = Path(__file__).resolve().parent / "aicpu_device_query"
_CACHE = PROJECT_ROOT / "build" / "cache" / "aicpu_device_query"
_DISPATCHER = PROJECT_ROOT / "build" / "lib" / "a5" / "dispatcher" / "libsimpler_aicpu_dispatcher.so"


def default_plan_path() -> Path:
    return PROJECT_ROOT / RELATIVE_PLAN


def cpus_side_path(plan_path: Path, device_id: int) -> Path:
    stem = plan_path.name[: -len(".json")] if plan_path.name.endswith(".json") else plan_path.name
    return plan_path.with_name(f"{stem}.{device_id}.cpus")


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _ascend_env() -> dict[str, str]:
    home = os.environ.get("ASCEND_HOME_PATH", "").strip()
    home_p = (
        Path(home)
        if home
        else next(
            (
                p
                for p in (Path("/usr/local/Ascend/ascend-toolkit/latest"), Path("/usr/local/Ascend/cann-9.2.0"))
                if p.is_dir()
            ),
            None,
        )
    )
    if home_p is None or not (home_p / "set_env.sh").is_file():
        raise RuntimeError("ASCEND_HOME_PATH not found")
    out = subprocess.run(
        ["bash", "-c", f'source "{home_p / "set_env.sh"}" && env -0'], check=True, stdout=subprocess.PIPE
    )
    env = dict(os.environ)
    for entry in out.stdout.split(b"\0"):
        if b"=" in entry:
            k, _, v = entry.partition(b"=")
            env[k.decode()] = v.decode()
    env["ASCEND_HOME_PATH"] = str(home_p)
    return env


def _cmake_build(src: Path, build: Path, env: dict[str, str], *, cross: bool = False) -> None:
    cfg = ["cmake", "-S", str(src), "-B", str(build)]
    if cross:
        ah = Path(env["ASCEND_HOME_PATH"]) / "tools" / "hcc" / "bin"
        cfg += [
            f"-DCMAKE_C_COMPILER={ah / 'aarch64-target-linux-gnu-gcc'}",
            f"-DCMAKE_CXX_COMPILER={ah / 'aarch64-target-linux-gnu-g++'}",
        ]
    subprocess.run(cfg, check=True, cwd=PROJECT_ROOT, env=env)
    subprocess.run(["cmake", "--build", str(build), f"-j{os.cpu_count() or 1}"], check=True, cwd=PROJECT_ROOT, env=env)


def _run_query_device_hal(device_id: int, mode: str) -> str:
    """Build aicpu_device_query artifacts if needed; return query_device_hal stdout."""
    if mode not in ("--rtt-json", "--json"):
        raise ValueError(f"unsupported mode: {mode}")
    if not (_TOOL / "host").is_dir():
        raise RuntimeError(f"aicpu_device_query sources missing under {_TOOL}")
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake is required to build the aicpu_device_query backend")
    env = _ascend_env()
    dispatcher = Path(env["SIMPLER_DISPATCHER_SO"]) if env.get("SIMPLER_DISPATCHER_SO") else _DISPATCHER
    if not dispatcher.is_file():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "simpler_setup.build_runtimes",
                "--lib-dir",
                str(PROJECT_ROOT / "build" / "lib"),
                "--cache-dir",
                str(PROJECT_ROOT / "build" / "cache"),
                "--platforms",
                "a5",
            ],
            check=True,
            cwd=PROJECT_ROOT,
            env={**env, "PYTHONPATH": f"{PROJECT_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"},
        )
        dispatcher = _DISPATCHER
        if not dispatcher.is_file():
            raise RuntimeError(f"Missing dispatcher SO: {dispatcher}")
    query_so = (
        Path(env["SIMPLER_AICPU_QUERY_SO"])
        if env.get("SIMPLER_AICPU_QUERY_SO")
        else _CACHE / "device" / "libaicpu_query.so"
    )
    if not query_so.is_file():
        _cmake_build(_TOOL / "device", _CACHE / "device", env, cross=True)
        query_so = _CACHE / "device" / "libaicpu_query.so"
        if not query_so.is_file():
            raise RuntimeError(f"Missing query SO: {query_so}")
    host_bin = _CACHE / "host" / "query_device_hal"
    if not host_bin.is_file():
        _cmake_build(_TOOL / "host", _CACHE / "host", env)
        if not host_bin.is_file():
            raise RuntimeError(f"Missing host launcher: {host_bin}")
    env = {**env, "SIMPLER_DISPATCHER_SO": str(dispatcher), "SIMPLER_AICPU_QUERY_SO": str(query_so)}
    return subprocess.run(
        [str(host_bin), str(device_id), mode],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        cwd=PROJECT_ROOT,
        env=env,
    ).stdout


def pick_orchestrator(pool: Sequence[Mapping[str, Any]]) -> int:
    """Return pool_idx with the smallest avg_handshake_ticks (tie: smaller idx)."""
    if not pool:
        raise ValueError("empty pool")
    best = min(pool, key=lambda e: (int(e["avg_handshake_ticks"]), int(e["pool_idx"])))
    return int(best["pool_idx"])


def pack_schedulers_from_die_scores(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Phys die0,die1,die1,die0 picks → logical [P0,P3,P1,P2] (S0/S1 die0, S2/S3 die1)."""
    if len(candidates) < SCHEDULER_COUNT:
        raise ValueError(f"need at least {SCHEDULER_COUNT} non-orch candidates")
    remaining = [dict(e) for e in candidates]
    picks: list[dict[str, Any]] = []
    for target_die in PHYS_PICK_DIES:
        key = "die0_sum_ticks" if target_die == 0 else "die1_sum_ticks"
        remaining.sort(key=lambda e, k=key: (int(e[k]), int(e["cpu_id"])))
        chosen = dict(remaining.pop(0))
        chosen["assigned_die"] = target_die
        picks.append(chosen)
    logical = [picks[0], picks[3], picks[1], picks[2]]
    for i, entry in enumerate(logical):
        entry["logical_idx"] = i
        entry["assigned_die"] = 0 if i < 2 else 1
    return [int(e["cpu_id"]) for e in logical], logical


def build_allowed_cpus_from_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    """Turn device probe JSON into an authoritative device plan node."""
    if probe.get("pool_too_small"):
        occupy = [_require_int(cpu, "user_pool_cpus[]") for cpu in probe["user_pool_cpus"]]
        if len(occupy) < 2:
            raise ValueError("pool_too_small requires at least 2 CPUs")
        return build_contiguous_fallback_node(
            soc_name=str(probe["soc_name"]),
            occupy_cpus=occupy,
            plan_source=PLAN_SOURCE_POOL_TOO_SMALL,
            pool_too_small=True,
        )

    pool_raw = probe.get("pool")
    if not isinstance(pool_raw, list) or len(pool_raw) < 5:
        raise ValueError("probe pool must contain at least 5 entries")

    pool_entries: list[dict[str, Any]] = []
    for raw in pool_raw:
        if not isinstance(raw, dict):
            raise ValueError("pool entry must be an object")
        pool_entries.append(
            {
                "pool_idx": _require_int(raw.get("pool_idx"), "pool_idx"),
                "cpu_id": _require_int(raw.get("cpu_id"), "cpu_id"),
                "avg_handshake_ticks": _require_int(raw.get("avg_handshake_ticks"), "avg_handshake_ticks"),
                "die0_sum_ticks": _require_int(raw.get("die0_sum_ticks"), "die0_sum_ticks"),
                "die1_sum_ticks": _require_int(raw.get("die1_sum_ticks"), "die1_sum_ticks"),
                "is_orch": _require_int(raw.get("is_orch"), "is_orch"),
            }
        )

    orch_idx = _require_int(probe.get("orch_pool_idx"), "orch_pool_idx")
    orch_entries = [e for e in pool_entries if int(e["pool_idx"]) == orch_idx]
    if len(orch_entries) != 1:
        orch_idx = pick_orchestrator(pool_entries)
        orch_entries = [e for e in pool_entries if int(e["pool_idx"]) == orch_idx]
    orch = orch_entries[0]
    sched_cpus, schedulers = pack_schedulers_from_die_scores(
        [e for e in pool_entries if int(e["pool_idx"]) != orch_idx]
    )
    allowed = sched_cpus + [int(orch["cpu_id"])]
    if len(set(allowed)) != len(allowed):
        raise ValueError("allowed_cpus contains duplicates")

    return {
        "soc_name": probe["soc_name"],
        "architecture": "a5",
        "plan_source": MEASUREMENT_METHOD,
        "user_pool_cpus": [_require_int(cpu, "user_pool_cpus[]") for cpu in probe.get("user_pool_cpus", [])],
        "allowed_cpus": allowed,
        "orch_cpu": int(orch["cpu_id"]),
        "orch_avg_handshake_ticks": int(orch["avg_handshake_ticks"]),
        "schedulers": [
            {
                "logical_idx": int(s["logical_idx"]),
                "cpu_id": int(s["cpu_id"]),
                "assigned_die": int(s["assigned_die"]),
                "die0_sum_ticks": int(s["die0_sum_ticks"]),
                "die1_sum_ticks": int(s["die1_sum_ticks"]),
            }
            for s in schedulers
        ],
        "pool_too_small": False,
    }


def validate_probe_result(raw: object, expected_device: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("probe output must be a JSON object")
    if _require_int(raw.get("schema_version"), "schema_version") != PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported affinity probe schema_version")
    if raw.get("measurement_method") != MEASUREMENT_METHOD:
        raise ValueError("unsupported measurement_method")
    soc_name = raw.get("soc_name")
    if not isinstance(soc_name, str) or not soc_name.startswith("Ascend950"):
        raise ValueError(f"unsupported probe soc_name: {soc_name!r}")
    if _require_int(raw.get("device_id"), "device_id") != expected_device:
        raise ValueError("probe output device_id does not match --device")
    if raw.get("pool_too_small"):
        pool = raw.get("user_pool_cpus")
        if not isinstance(pool, list) or len(pool) < 2:
            raise ValueError("pool_too_small requires user_pool_cpus with >= 2 entries")
        return {
            "schema_version": PROBE_SCHEMA_VERSION,
            "measurement_method": MEASUREMENT_METHOD,
            "soc_name": soc_name,
            "device_id": expected_device,
            "pool_too_small": True,
            "user_pool_cpus": [_require_int(cpu, "user_pool_cpus[]") for cpu in pool],
        }
    if _require_int(raw.get("samples_per_core", PROBE_SAMPLES_PER_CORE), "samples_per_core") != PROBE_SAMPLES_PER_CORE:
        raise ValueError("unexpected samples_per_core")
    return raw  # type: ignore[return-value]


def run_affinity_probe(device_id: int) -> dict[str, Any]:
    try:
        stdout = _run_query_device_hal(device_id, "--rtt-json")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"affinity probe backend failed with exit code {exc.returncode}") from exc
    try:
        raw = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("affinity probe backend returned invalid JSON") from exc
    return validate_probe_result(raw, device_id)


def load_plan(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": PLAN_SCHEMA_VERSION, "socs": {}}
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or not isinstance(plan.get("socs"), dict):
        raise ValueError(f"existing plan is malformed: {path}")
    if _require_int(plan.get("schema_version"), "schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(f"existing plan uses an unsupported schema_version; regenerate it: {path}")
    return plan


def merge_device_plan(
    plan: dict[str, Any], *, soc_name: str, device_id: int, device_node: dict[str, Any]
) -> dict[str, Any]:
    plan.setdefault("socs", {}).setdefault(soc_name, {}).setdefault("devices", {})[str(device_id)] = device_node
    plan["schema_version"] = PLAN_SCHEMA_VERSION
    plan["_comment"] = (
        "A5 AICPU affinity plan (authoritative allowed_cpus). "
        "Logical S0/S1 own die0; S2/S3 own die1. Generated by simpler_setup.tools.rtt_die_preflight."
    )
    return plan


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def atomic_write_plan(path: Path, plan: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(plan, indent=2) + "\n")


def write_cpus_side_file(plan_path: Path, *, device_id: int, device_node: Mapping[str, Any]) -> Path:
    side = cpus_side_path(plan_path, device_id)
    _atomic_write_text(
        side,
        (
            f"soc={device_node['soc_name']}\n"
            f"source={device_node['plan_source']}\n"
            f"pool_too_small={1 if device_node.get('pool_too_small', False) else 0}\n"
            f"cpus={','.join(str(int(cpu)) for cpu in device_node['allowed_cpus'])}\n"
        ),
    )
    return side


def build_contiguous_fallback_node(
    *,
    soc_name: str,
    occupy_cpus: Sequence[int],
    plan_source: str = PLAN_SOURCE_PROBE_FAILED,
    pool_too_small: bool | None = None,
) -> dict[str, Any]:
    """Contiguous OCCUPY order fallback (pool<5 or probe failure). Last slot is orch."""
    pool = [int(cpu) for cpu in occupy_cpus]
    if len(pool) < 2 or len(set(pool)) != len(pool):
        raise ValueError("fallback occupy_cpus must be unique and contain at least 1S+1O")
    active = max(2, min(len(pool), SCHEDULER_COUNT + 1))
    allowed = list(pool[:active])
    too_small = len(pool) < SCHEDULER_COUNT + 1 if pool_too_small is None else pool_too_small
    die0_slots = max(1, (active - 1) // 2)
    return {
        "soc_name": soc_name,
        "architecture": "a5",
        "plan_source": plan_source,
        "user_pool_cpus": pool,
        "allowed_cpus": allowed,
        "orch_cpu": allowed[-1],
        "schedulers": [
            {
                "logical_idx": idx,
                "cpu_id": cpu,
                "assigned_die": 0 if idx < die0_slots else 1,
                "die0_sum_ticks": 0,
                "die1_sum_ticks": 0,
            }
            for idx, cpu in enumerate(allowed[:-1])
        ],
        "pool_too_small": too_small,
    }


def build_device_node_from_allowed(
    *,
    soc_name: str,
    allowed_cpus: Sequence[int],
    plan_source: str = PLAN_SOURCE_MANUAL,
) -> dict[str, Any]:
    if len(allowed_cpus) < 2 or len(set(allowed_cpus)) != len(allowed_cpus):
        raise ValueError("allowed_cpus must be unique and contain at least 1S+1O")
    return {
        "soc_name": soc_name,
        "architecture": "a5",
        "plan_source": plan_source,
        "allowed_cpus": list(map(int, allowed_cpus)),
        "orch_cpu": int(allowed_cpus[-1]),
        "schedulers": [
            {
                "logical_idx": idx,
                "cpu_id": int(cpu),
                "assigned_die": 0 if idx < 2 else 1,
                "die0_sum_ticks": 0,
                "die1_sum_ticks": 0,
            }
            for idx, cpu in enumerate(allowed_cpus[:-1])
        ],
        "pool_too_small": len(allowed_cpus) < SCHEDULER_COUNT + 1,
    }


def try_parse_occupy_from_topo_json(raw: object) -> tuple[str, list[int]] | None:
    """Best-effort extract (soc_name, sorted user cpus) from aicpu-device-query --json."""
    if not isinstance(raw, dict):
        return None
    soc_name = raw.get("soc_name")
    if not isinstance(soc_name, str) or not soc_name:
        return None

    def _as_cpu_list(values: object) -> list[int] | None:
        if not isinstance(values, list) or len(values) < 2:
            return None
        cpus: list[int] = []
        for item in values:
            try:
                cpus.append(_require_int(item.get("cpu_id") if isinstance(item, dict) else item, "cpu"))
            except (AttributeError, ValueError):
                return None
        unique = sorted(set(cpus))
        return unique if len(unique) >= 2 else None

    for key in ("user_pool_cpus", "occupy_cpus", "os_schedulable_cpus"):
        parsed = _as_cpu_list(raw.get(key))
        if parsed is not None:
            return soc_name, parsed

    launch_plan = raw.get("launch_plan")
    if isinstance(launch_plan, dict):
        parsed = _as_cpu_list(launch_plan.get("allowed_cpus"))
        if parsed is not None:
            return soc_name, parsed

    masks = raw.get("device_masks")
    if isinstance(masks, dict):
        occupy_obj = masks.get("occupy")
        if isinstance(occupy_obj, dict) and occupy_obj.get("valid"):
            value = occupy_obj.get("value")
            occupy_int: int | None = None
            if isinstance(value, str):
                try:
                    occupy_int = int(value, 0)
                except ValueError:
                    pass
            elif isinstance(value, int) and not isinstance(value, bool):
                occupy_int = value
            if occupy_int is not None and occupy_int > 0:
                cpus = [bit for bit in range(64) if (occupy_int >> bit) & 1]
                if len(cpus) >= 2:
                    return soc_name, cpus
    return None


def run_topo_json_fallback(device_id: int) -> tuple[str, list[int]]:
    """Lighter --json probe used only to write contiguous fallback plans."""
    parsed = try_parse_occupy_from_topo_json(json.loads(_run_query_device_hal(device_id, "--json")))
    if parsed is None:
        raise RuntimeError("topo --json did not yield a usable occupy CPU list")
    return parsed


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]


def persist_device_plan(out_path: Path, *, soc_name: str, device_id: int, device_node: dict[str, Any]) -> None:
    plan = merge_device_plan(load_plan(out_path), soc_name=soc_name, device_id=device_id, device_node=device_node)
    atomic_write_plan(out_path, plan)
    write_cpus_side_file(out_path, device_id=device_id, device_node=device_node)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe A5 AICPU affinity and write the authoritative allowed_cpus plan."
    )
    parser.add_argument("--device", type=int, default=0, help="Logical ACL device id")
    parser.add_argument("--out", type=Path, default=None, help=f"Output JSON path (default: {default_plan_path()})")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true", help="Run the full affinity preflight on device")
    mode.add_argument("--allowed-cpus", help="Offline authoritative allowed_cpus (S0,S1,S2,S3,O); requires --soc")
    mode.add_argument(
        "--fallback-occupy",
        help="Write contiguous fallback plan from occupy CPU list (comma-separated); requires --soc",
    )
    parser.add_argument("--soc", help="SoC name for offline / fallback modes")
    parser.add_argument(
        "--plan-source",
        default=None,
        help="Optional plan_source override (e.g. auto-first-run when ChipWorker.init probes)",
    )
    args = parser.parse_args(argv)

    out_path = args.out if args.out is not None else default_plan_path()
    try:
        if args.probe:
            if args.soc is not None:
                parser.error("--probe obtains --soc from hardware")
            try:
                probe = run_affinity_probe(args.device)
                device_node = build_allowed_cpus_from_probe(probe)
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as probe_exc:
                print(f"rtt_die_preflight: probe failed ({probe_exc}); writing contiguous fallback", file=sys.stderr)
                try:
                    soc_name, occupy = run_topo_json_fallback(args.device)
                    device_node = build_contiguous_fallback_node(
                        soc_name=soc_name,
                        occupy_cpus=occupy,
                        plan_source=args.plan_source if args.plan_source is not None else PLAN_SOURCE_PROBE_FAILED,
                    )
                except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as fallback_exc:
                    raise RuntimeError(
                        f"affinity probe failed and contiguous fallback also failed: {fallback_exc}"
                    ) from probe_exc
            else:
                if args.plan_source is not None:
                    device_node["plan_source"] = args.plan_source
            soc_name = device_node["soc_name"]
        elif args.fallback_occupy is not None:
            if args.soc is None:
                parser.error("--fallback-occupy requires --soc")
            soc_name = args.soc
            device_node = build_contiguous_fallback_node(
                soc_name=soc_name,
                occupy_cpus=parse_int_list(args.fallback_occupy),
                plan_source=args.plan_source if args.plan_source is not None else PLAN_SOURCE_PROBE_FAILED,
            )
        else:
            if args.soc is None or args.allowed_cpus is None:
                parser.error("offline mode requires --soc and --allowed-cpus")
            soc_name = args.soc
            device_node = build_device_node_from_allowed(
                soc_name=soc_name,
                allowed_cpus=parse_int_list(args.allowed_cpus),
                plan_source=args.plan_source if args.plan_source is not None else PLAN_SOURCE_MANUAL,
            )

        persist_device_plan(out_path, soc_name=soc_name, device_id=args.device, device_node=device_node)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"rtt_die_preflight: {exc}", file=sys.stderr)
        return 1
    side = cpus_side_path(out_path, args.device)
    print(
        f"Wrote affinity plan: {out_path} side={side} soc={soc_name} device={args.device} "
        f"allowed_cpus={device_node['allowed_cpus']} source={device_node['plan_source']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
