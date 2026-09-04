# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for A5 AICPU affinity preflight ranking and plan writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simpler_setup.tools import rtt_die_preflight as preflight


def test_pick_orchestrator_min_avg_tiebreak():
    pool = [
        {"pool_idx": 0, "avg_handshake_ticks": 100},
        {"pool_idx": 1, "avg_handshake_ticks": 50},
        {"pool_idx": 2, "avg_handshake_ticks": 50},
        {"pool_idx": 3, "avg_handshake_ticks": 80},
    ]
    assert preflight.pick_orchestrator(pool) == 1


def test_pack_schedulers_phys_order_to_logical_die_contract():
    # Candidates after orch removal. Scores chosen so phys picks are:
    # die0 best=cpu10, die1 best=cpu20, die1 2nd=cpu21, die0 2nd=cpu11
    candidates = [
        {"cpu_id": 10, "die0_sum_ticks": 100, "die1_sum_ticks": 900},
        {"cpu_id": 11, "die0_sum_ticks": 200, "die1_sum_ticks": 800},
        {"cpu_id": 20, "die0_sum_ticks": 900, "die1_sum_ticks": 100},
        {"cpu_id": 21, "die0_sum_ticks": 800, "die1_sum_ticks": 150},
    ]
    allowed, schedulers = preflight.pack_schedulers_from_die_scores(candidates)
    assert allowed == [10, 11, 20, 21]
    assert [s["assigned_die"] for s in schedulers] == [0, 0, 1, 1]
    assert [s["logical_idx"] for s in schedulers] == [0, 1, 2, 3]


def test_build_allowed_cpus_from_probe_full():
    probe = {
        "schema_version": 3,
        "measurement_method": preflight.MEASUREMENT_METHOD,
        "soc_name": "Ascend950PR_9599",
        "device_id": 0,
        "user_pool_cpus": [3, 4, 5, 6, 7, 8],
        "orch_pool_idx": 5,
        "pool": [
            {
                "pool_idx": 0,
                "cpu_id": 3,
                "avg_handshake_ticks": 40,
                "die0_sum_ticks": 100,
                "die1_sum_ticks": 900,
                "is_orch": 0,
            },
            {
                "pool_idx": 1,
                "cpu_id": 4,
                "avg_handshake_ticks": 41,
                "die0_sum_ticks": 200,
                "die1_sum_ticks": 800,
                "is_orch": 0,
            },
            {
                "pool_idx": 2,
                "cpu_id": 5,
                "avg_handshake_ticks": 42,
                "die0_sum_ticks": 900,
                "die1_sum_ticks": 100,
                "is_orch": 0,
            },
            {
                "pool_idx": 3,
                "cpu_id": 6,
                "avg_handshake_ticks": 43,
                "die0_sum_ticks": 800,
                "die1_sum_ticks": 150,
                "is_orch": 0,
            },
            {
                "pool_idx": 4,
                "cpu_id": 7,
                "avg_handshake_ticks": 44,
                "die0_sum_ticks": 700,
                "die1_sum_ticks": 700,
                "is_orch": 0,
            },
            {
                "pool_idx": 5,
                "cpu_id": 8,
                "avg_handshake_ticks": 10,
                "die0_sum_ticks": 0,
                "die1_sum_ticks": 0,
                "is_orch": 1,
            },
        ],
    }
    node = preflight.build_allowed_cpus_from_probe(probe)
    assert node["orch_cpu"] == 8
    assert node["allowed_cpus"][-1] == 8
    assert len(node["allowed_cpus"]) == 5
    assert node["schedulers"][0]["assigned_die"] == 0
    assert node["schedulers"][1]["assigned_die"] == 0
    assert node["schedulers"][2]["assigned_die"] == 1
    assert node["schedulers"][3]["assigned_die"] == 1
    assert node["plan_source"] == preflight.MEASUREMENT_METHOD


def test_build_allowed_cpus_pool_too_small():
    probe = {
        "schema_version": 3,
        "measurement_method": preflight.MEASUREMENT_METHOD,
        "soc_name": "Ascend950PR_9599",
        "device_id": 0,
        "pool_too_small": True,
        "user_pool_cpus": [3, 4, 5],
    }
    node = preflight.build_allowed_cpus_from_probe(probe)
    assert node["pool_too_small"] is True
    assert node["allowed_cpus"] == [3, 4, 5]
    assert node["orch_cpu"] == 5
    assert node["plan_source"] == preflight.PLAN_SOURCE_POOL_TOO_SMALL


def test_contiguous_fallback_and_side_file(tmp_path: Path):
    out = tmp_path / "aicpu_affinity_plan.json"
    node = preflight.build_contiguous_fallback_node(
        soc_name="Ascend950PR_9599",
        occupy_cpus=[8, 3, 4, 5, 6, 7],
        plan_source=preflight.PLAN_SOURCE_PROBE_FAILED,
    )
    # Contiguous order preserves input order (not resorted).
    assert node["allowed_cpus"] == [8, 3, 4, 5, 6]
    assert node["orch_cpu"] == 6
    preflight.persist_device_plan(out, soc_name="Ascend950PR_9599", device_id=1, device_node=node)
    side = preflight.cpus_side_path(out, 1)
    assert side.is_file()
    text = side.read_text(encoding="utf-8")
    assert "soc=Ascend950PR_9599\n" in text
    assert f"source={preflight.PLAN_SOURCE_PROBE_FAILED}\n" in text
    assert "cpus=8,3,4,5,6\n" in text


def test_try_parse_occupy_from_topo_json_objects():
    raw = {
        "soc_name": "Ascend950PR_9599",
        "os_schedulable_cpus": [
            {"cpu_id": 3, "die_id": 0},
            {"cpu_id": 4, "die_id": 0},
            {"cpu_id": 5, "die_id": 1},
        ],
    }
    parsed = preflight.try_parse_occupy_from_topo_json(raw)
    assert parsed is not None
    soc, cpus = parsed
    assert soc == "Ascend950PR_9599"
    assert cpus == [3, 4, 5]


def test_cli_fallback_occupy(tmp_path: Path):
    out = tmp_path / "plan.json"
    rc = preflight.main(
        [
            "--device",
            "0",
            "--soc",
            "Ascend950PR_9599",
            "--fallback-occupy",
            "3,4,5,6,7,8",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["socs"]["Ascend950PR_9599"]["devices"]["0"]["allowed_cpus"] == [3, 4, 5, 6, 7]
    assert loaded["socs"]["Ascend950PR_9599"]["devices"]["0"]["plan_source"] == (preflight.PLAN_SOURCE_PROBE_FAILED)
    side = preflight.cpus_side_path(out, 0).read_text(encoding="utf-8")
    assert "cpus=3,4,5,6,7\n" in side
    assert f"source={preflight.PLAN_SOURCE_PROBE_FAILED}\n" in side


def test_validate_probe_rejects_old_schema():
    with pytest.raises(ValueError, match="schema_version"):
        preflight.validate_probe_result(
            {
                "schema_version": 2,
                "measurement_method": "serialized-cond-rtt-v2",
                "soc_name": "Ascend950PR_9599",
                "device_id": 0,
            },
            0,
        )


def test_atomic_write_and_device_buckets(tmp_path: Path):
    out = tmp_path / "aicpu_affinity_plan.json"
    node0 = preflight.build_device_node_from_allowed(
        soc_name="Ascend950PR_9599", allowed_cpus=[3, 4, 5, 6, 7], plan_source="manual"
    )
    plan = preflight.merge_device_plan(
        preflight.load_plan(out), soc_name="Ascend950PR_9599", device_id=0, device_node=node0
    )
    preflight.atomic_write_plan(out, plan)

    node1 = preflight.build_device_node_from_allowed(
        soc_name="Ascend950PR_9599", allowed_cpus=[10, 11, 12, 13, 14], plan_source="manual"
    )
    plan = preflight.merge_device_plan(
        preflight.load_plan(out), soc_name="Ascend950PR_9599", device_id=1, device_node=node1
    )
    preflight.atomic_write_plan(out, plan)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 3
    devices = loaded["socs"]["Ascend950PR_9599"]["devices"]
    assert devices["0"]["allowed_cpus"] == [3, 4, 5, 6, 7]
    assert devices["1"]["allowed_cpus"] == [10, 11, 12, 13, 14]


def test_cli_offline_allowed_cpus(tmp_path: Path):
    out = tmp_path / "plan.json"
    rc = preflight.main(
        [
            "--device",
            "2",
            "--soc",
            "Ascend950PR_9599",
            "--allowed-cpus",
            "3,4,5,6,7",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["socs"]["Ascend950PR_9599"]["devices"]["2"]["allowed_cpus"] == [3, 4, 5, 6, 7]
