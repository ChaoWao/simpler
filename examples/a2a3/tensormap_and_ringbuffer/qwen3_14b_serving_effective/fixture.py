#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Load and validate a portable Serving-TMR prefill fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import load_file


SCHEMA_NAME = "serving-tmr-standalone-fixture-v1"
EXPECTED_METADATA = {
    "prompt_token_ids": ((16, 3338), torch.int32),
    "seq_lens_after_first_token": ((16,), torch.int32),
    "next_slot_mapping": ((16,), torch.int32),
    "first_generated_token_ids": ((16,), torch.int32),
    "block_table": ((16, 32), torch.int32),
    "tokens_used_after_prefill": ((16,), torch.int32),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_type_matches(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "boolean": isinstance(value, bool),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
    }.get(expected, True)


def _condition_matches(value: Any, schema: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    for name, rule in schema.get("properties", {}).items():
        if name not in value:
            return False
        if "const" in rule and value[name] != rule["const"]:
            return False
    return True


def _validate_json_schema(
    value: Any, schema: dict[str, Any], path: str = "manifest"
) -> None:
    expected_type = schema.get("type")
    if expected_type is not None and not _json_type_matches(value, expected_type):
        raise ValueError(f"{path} must have JSON type {expected_type}")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} must equal {schema['const']!r}")

    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = sorted(set(required) - set(value))
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise ValueError(f"{path} has unsupported fields: {extras}")
        for name, rule in properties.items():
            if name in value:
                _validate_json_schema(value[name], rule, f"{path}.{name}")

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path} has too many items")

    for rule in schema.get("allOf", []):
        if _condition_matches(value, rule.get("if", {})):
            _validate_json_schema(value, rule.get("then", {}), path)


def _verify_sums(root: Path) -> dict[Path, str]:
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        raise FileNotFoundError(sums_path)
    expected: dict[Path, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1)
        relative_path = Path(relative.removeprefix("./"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"invalid SHA256SUMS path: {relative}")
        if relative_path in expected:
            raise ValueError(f"duplicate SHA256SUMS path: {relative_path}")
        expected[relative_path] = digest
    actual = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(expected) != actual:
        raise ValueError(
            f"SHA256SUMS file set mismatch: missing={sorted(actual - set(expected))} "
            f"extra={sorted(set(expected) - actual)}"
        )
    for relative, digest in expected.items():
        if _sha256(root / relative) != digest:
            raise ValueError(f"SHA256 mismatch: {relative}")
    return expected


@dataclass(frozen=True)
class Fixture:
    root: Path
    manifest: dict[str, Any]
    metadata: dict[str, torch.Tensor]

    @property
    def metadata_only(self) -> bool:
        return bool(self.manifest["metadata_only"])

    @property
    def artifact_dir(self) -> Path:
        return (self.root / self.manifest["artifact"]["path"]).resolve()

    @property
    def golden_path(self) -> Path:
        return (self.root / self.manifest["golden_contract"]["path"]).resolve()

    def require_executable(self) -> None:
        if self.metadata_only:
            raise RuntimeError(
                "metadata-only fixture cannot allocate device state or execute"
            )

    def iter_kv_shards(self, kind: str) -> Iterator[tuple[int, Path, str]]:
        self.require_executable()
        entries = [
            entry for entry in self.manifest["kv_shards"] if entry["kind"] == kind
        ]
        for entry in sorted(entries, key=lambda item: int(item["layer"])):
            yield int(entry["layer"]), self.root / entry["path"], entry["tensor"]

    def verify_external_inputs(self, model_dir: Path) -> None:
        for entry in self.manifest["checkpoint_files"]:
            path = model_dir / entry["name"]
            if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
                raise ValueError(f"checkpoint file mismatch: {entry['name']}")
            if _sha256(path) != entry["sha256"]:
                raise ValueError(f"checkpoint SHA256 mismatch: {entry['name']}")

    def verify_artifact_and_golden(self) -> None:
        self.require_executable()
        artifact = self.manifest["artifact"]
        if _sha256(self.artifact_dir / "manifest.json") != artifact["manifest_sha256"]:
            raise ValueError("artifact manifest SHA256 mismatch")
        if _sha256(self.artifact_dir / "SHA256SUMS") != artifact["sha256sums_sha256"]:
            raise ValueError("artifact SHA256SUMS mismatch")
        _verify_sums(self.artifact_dir)
        if not self.golden_path.is_file():
            raise FileNotFoundError(self.golden_path)
        _verify_sums(self.golden_path.parent)

    def load_golden(self) -> dict[str, torch.Tensor]:
        self.require_executable()
        golden = load_file(str(self.golden_path), device="cpu")
        shapes = self.manifest["golden_contract"]["tensor_shapes"]
        if set(golden) != set(shapes):
            raise ValueError("golden tensor names do not match the fixture contract")
        for name, shape in shapes.items():
            tensor = golden[name]
            if tensor.dtype != torch.int32 or tuple(tensor.shape) != tuple(shape):
                raise ValueError(f"golden tensor contract mismatch: {name}")
        if not torch.equal(golden["step_index"], torch.arange(127, dtype=torch.int32)):
            raise ValueError("golden step_index must be 0..126")
        if not torch.equal(
            golden["decode_input_token_ids"][0],
            self.metadata["first_generated_token_ids"],
        ):
            raise ValueError("golden first decode input differs from the prefill token")
        layout = self.manifest["physical_layout"]
        page_size = int(layout["page_size"])
        num_pages = int(layout["num_pages"])
        positions = golden["seq_lens"] - 1
        slots = golden["slot_mapping"]
        if not torch.equal(slots.remainder(page_size), positions.remainder(page_size)):
            raise ValueError("golden slot_mapping offset differs from seq_lens")
        page_ids = slots.div(page_size, rounding_mode="floor")
        if int(page_ids.min()) < 0 or int(page_ids.max()) >= num_pages:
            raise ValueError("golden slot_mapping page is outside physical KV storage")
        return golden

    def allocate_kv(self, runtime: Any, kind: str) -> Any:
        self.require_executable()
        layout = self.manifest["physical_layout"]
        num_layers = int(layout["num_layers"])
        num_pages = int(layout["num_pages"])
        num_kv_heads = int(layout["num_kv_heads"])
        page_size = int(layout["page_size"])
        head_dim = int(layout["head_dim"])
        rows = num_layers * num_pages * num_kv_heads * page_size
        used_page_ids = self.metadata["used_page_ids"].to(torch.long)
        host = torch.zeros((rows, head_dim), dtype=torch.bfloat16)
        host_layers = host.view(
            num_layers, num_pages, num_kv_heads, page_size, head_dim
        )
        entries = list(self.iter_kv_shards(kind))
        if [layer for layer, _, _ in entries] != list(range(num_layers)):
            raise ValueError(f"{kind} KV shards do not cover every layer")
        for layer, path, tensor_name in entries:
            shard = load_file(str(path), device="cpu")[tensor_name]
            expected_shape = (len(used_page_ids), num_kv_heads, page_size, head_dim)
            if shard.dtype != torch.bfloat16 or tuple(shard.shape) != expected_shape:
                raise ValueError(f"KV shard contract mismatch: {path.name}")
            host_layers[layer].index_copy_(0, used_page_ids, shard)
            del shard
        return runtime.alloc_tensor(tuple(host.shape), host.dtype, init=host)


def _validate_metadata(
    manifest: dict[str, Any], metadata: dict[str, torch.Tensor]
) -> None:
    expected_names = set(EXPECTED_METADATA) | {"used_page_ids", "allocated_page_ids"}
    if set(metadata) != expected_names:
        raise ValueError("metadata tensor names do not match fixture schema v1")
    for name, (shape, dtype) in EXPECTED_METADATA.items():
        tensor = metadata[name]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"metadata tensor contract mismatch: {name}")
    for name in ("used_page_ids", "allocated_page_ids"):
        tensor = metadata[name]
        if tensor.ndim != 1 or tensor.dtype != torch.int32:
            raise ValueError(f"metadata tensor contract mismatch: {name}")

    layout = manifest["physical_layout"]
    num_pages = int(layout["num_pages"])
    page_size = int(layout["page_size"])
    block_table = metadata["block_table"]
    allocated = sorted(
        {int(value) for value in block_table.flatten().tolist() if value >= 0}
    )
    if allocated != metadata["allocated_page_ids"].tolist():
        raise ValueError("allocated_page_ids differs from block_table")
    if not allocated or allocated[0] < 0 or allocated[-1] >= num_pages:
        raise ValueError("block_table page is outside physical KV storage")

    prompt_tokens = int(manifest["prompt_tokens"])
    used_blocks_per_row = math.ceil(prompt_tokens / page_size)
    used = sorted(
        {
            int(value)
            for value in block_table[:, :used_blocks_per_row].flatten().tolist()
            if value >= 0
        }
    )
    if used != metadata["used_page_ids"].tolist():
        raise ValueError("used_page_ids differs from the prompt block mapping")
    if len(used) != int(layout["used_page_count"]):
        raise ValueError("used_page_count differs from metadata")

    page_index, page_offset = divmod(prompt_tokens, page_size)
    expected_slots = block_table[:, page_index] * page_size + page_offset
    if not torch.equal(expected_slots, metadata["next_slot_mapping"]):
        raise ValueError("next_slot_mapping does not name the first decode position")
    if not torch.equal(
        metadata["seq_lens_after_first_token"],
        torch.full((16,), prompt_tokens + 1, dtype=torch.int32),
    ):
        raise ValueError("seq_lens_after_first_token is inconsistent with the prompt")
    if not torch.equal(
        metadata["tokens_used_after_prefill"],
        torch.full((16,), prompt_tokens, dtype=torch.int32),
    ):
        raise ValueError("tokens_used_after_prefill is inconsistent with the prompt")
    if not torch.equal(
        metadata["prompt_token_ids"], metadata["prompt_token_ids"][0].repeat(16, 1)
    ):
        raise ValueError("batch prompt rows must be identical")


def load_fixture(
    root: Path, expected_versions: dict[str, str] | None = None
) -> Fixture:
    root = root.resolve()
    fixture_sums = _verify_sums(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "manifest.schema.json").read_text(encoding="utf-8"))
    _validate_json_schema(manifest, schema)
    if manifest["schema"] != SCHEMA_NAME:
        raise ValueError(f"unsupported fixture schema: {manifest['schema']}")
    boundary = manifest["snapshot_boundary"]
    required_boundary = {
        "all_chunked_prefill_complete": True,
        "first_token_produced": True,
        "actual_decode_dispatches_before_snapshot": 0,
        "prefill_and_sampling_fences_retired": True,
    }
    for name, expected in required_boundary.items():
        if boundary.get(name) != expected:
            raise ValueError(f"snapshot boundary mismatch: {name}")
    if expected_versions is not None:
        for name, expected in expected_versions.items():
            if manifest["versions"].get(name) != expected:
                raise ValueError(f"fixture version mismatch: {name}")

    metadata_path = root / manifest["metadata"]["path"]
    if _sha256(metadata_path) != manifest["metadata"]["sha256"]:
        raise ValueError("metadata SHA256 differs from manifest")
    metadata = load_file(str(metadata_path), device="cpu")
    _validate_metadata(manifest, metadata)

    fixture = Fixture(root=root, manifest=manifest, metadata=metadata)
    if not fixture.metadata_only:
        expected_shards = 2 * int(manifest["kv_shard_contract"]["layers"])
        if len(manifest["kv_shards"]) != expected_shards:
            raise ValueError(f"full fixture must contain {expected_shards} KV shards")
        for entry in manifest["kv_shards"]:
            path = root / entry["path"]
            if (
                path.stat().st_size != int(entry["bytes"])
                or fixture_sums.get(Path(entry["path"])) != entry["sha256"]
            ):
                raise ValueError(f"KV shard checksum mismatch: {entry['path']}")
        fixture.verify_artifact_and_golden()
    return fixture


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    fixture = load_fixture(_parse_args().fixture)
    print(
        json.dumps(
            {
                "fixture": str(fixture.root),
                "schema": fixture.manifest["schema"],
                "metadata_only": fixture.metadata_only,
                "used_page_count": len(fixture.metadata["used_page_ids"]),
                "decode_dispatches_remaining": fixture.manifest[
                    "decode_dispatches_remaining"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
