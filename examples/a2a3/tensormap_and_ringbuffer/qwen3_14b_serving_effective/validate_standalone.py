#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate the frozen inputs for the standalone Qwen decode benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sums(root: Path) -> int:
    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in lines:
        digest, name = line.split(None, 1)
        relative = Path(name.removeprefix("./"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe checksum path: {relative}")
        if _sha256(root / relative) != digest:
            raise ValueError(f"checksum mismatch: {relative}")
    return len(lines)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("standalone_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--stack-manifest", type=Path, required=True)
    parser.add_argument("--fixture-module", type=Path, required=True)
    args = parser.parse_args()

    stack = json.loads(args.stack_manifest.read_text(encoding="utf-8"))
    inputs = stack["inputs"]
    if _sha256(args.fixture / "manifest.json") != inputs["fixture_manifest_sha256"]:
        raise ValueError("fixture manifest differs from the frozen stack manifest")
    if _sha256(args.fixture / "SHA256SUMS") != inputs["fixture_sha256sums_sha256"]:
        raise ValueError("fixture SHA256SUMS differs from the frozen stack manifest")
    if _sha256(args.artifact / "SHA256SUMS") != inputs["artifact_sha256sums_sha256"]:
        raise ValueError("artifact SHA256SUMS differs from the frozen stack manifest")

    fixture_module = _load_module(args.fixture_module.resolve())
    fixture = fixture_module.load_fixture(
        args.fixture.resolve(), expected_versions=None
    )
    fixture.require_executable()
    fixture.verify_external_inputs(args.model_dir.resolve())
    golden = fixture.load_golden()
    expected_prompt_hash = stack["workload"]["prompt_token_ids_sha256"]
    if fixture.manifest["prompt"]["token_ids_sha256"] != expected_prompt_hash:
        raise ValueError("fixture prompt token IDs differ from the frozen workload")
    if int(fixture.manifest["physical_layout"]["num_pages"]) != 691:
        raise ValueError("fixture KV backing capacity must match compare1_compat")

    artifact_file_count = _verify_sums(args.artifact.resolve())
    decode_meta_path = args.artifact / "decode" / "distributed_meta.json"
    if _sha256(decode_meta_path) != inputs["decode_distributed_meta_sha256"]:
        raise ValueError("decode distributed metadata differs from the frozen baseline")
    decode_meta = json.loads(decode_meta_path.read_text(encoding="utf-8"))
    params = [item["name"].split("__ssa_", 1)[0] for item in decode_meta["params"]]
    if len(params) != 25 or "sampled_ids" not in params or "sampled_ids_host" in params:
        raise ValueError("expected the compare1_compat 25-parameter decode ABI")
    config = decode_meta["distributed_config"]
    if (
        config["runtime"] != "tensormap_and_ringbuffer"
        or config["aicpu_thread_num"] != 4
    ):
        raise ValueError("decode runtime configuration differs from compare1_compat")

    result = {
        "schema": "simpler-standalone-decode-cpu-gate-v1",
        "passed": True,
        "runtime_requires_pypto_serving": False,
        "fixture_schema": fixture.manifest["schema"],
        "used_page_count": len(fixture.metadata["used_page_ids"]),
        "golden_decode_steps": int(golden["step_index"].numel()),
        "prompt_token_ids_sha256": expected_prompt_hash,
        "kv_backing_page_count": int(fixture.manifest["physical_layout"]["num_pages"]),
        "decode_abi_parameter_count": len(params),
        "artifact_file_count": artifact_file_count,
        "runtime": config["runtime"],
        "aicpu_thread_num": config["aicpu_thread_num"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
