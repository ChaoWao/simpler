#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end validation of the native prepare/launch/poll/wait/finalize seam."""

import tempfile

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _build_chip_task_args, _compare_outputs
from simpler_setup.tools.strace_timing import group_invocations, parse_spans

_VECTOR_KERNELS = "../vector_example/kernels/aiv"
_SLOT = 0
_GENERATION = 1
_SIZE = 128 * 128
_CHAIN_LENGTH = 64


@scene_test(level=2, runtime="host_build_graph")
class TestNativeRunLifecycle(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/long_vector_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{_VECTOR_KERNELS}/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{_VECTOR_KERNELS}/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "phase_split_preserves_blocking_compatibility",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4},
            "params": {"a": 2.0, "b": 3.0},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("a", torch.full((_SIZE,), params["a"], dtype=torch.float32)),
            TensorArg("b", torch.full((_SIZE,), params["b"], dtype=torch.float32)),
            TensorArg("out", torch.zeros(_SIZE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.out[:] = args.a + args.b + _CHAIN_LENGTH

    def test_run(self, st_platform, st_worker, request, capfd):
        super().test_run(st_platform, st_worker, request)

        spans = list(parse_spans(capfd.readouterr().err.splitlines()))
        invocations = [inv for inv in group_invocations(spans) if "simpler_run" in inv.by_name()]
        # Two of these are the abandoned diagnostic prepares below, which record a
        # simpler_run invocation without ever reaching simpler_run.runner_run.
        expected_invocations = 5 if st_platform.endswith("sim") else 9
        assert len(invocations) == expected_invocations

        common_depths = {
            "simpler_run": 0,
            "simpler_run.bind": 1,
            "simpler_run.validate": 1,
        }
        launched_depths = {
            **common_depths,
            "simpler_run.runner_run": 1,
            "simpler_run.runner_run.device_wall": 2,
        }
        launched_count = 0
        for invocation in invocations:
            by_name = invocation.by_name()
            expected_depths = launched_depths if "simpler_run.runner_run" in by_name else common_depths
            launched_count += "simpler_run.runner_run" in by_name
            assert expected_depths.keys() <= by_name.keys()
            assert len({span.hid for span in invocation.spans}) == 1
            assert sum(span.name == "simpler_run" for span in invocation.spans) == 1
            for name, depth in expected_depths.items():
                assert by_name[name].depth == depth

            root = by_name["simpler_run"]
            root_end = root.ts + root.dur
            for name in expected_depths.keys() - {"simpler_run", "simpler_run.runner_run.device_wall"}:
                stage = by_name[name]
                assert root.ts <= stage.ts <= stage.ts + stage.dur <= root_end
        expected_launched = 2 if st_platform.endswith("sim") else 6
        assert launched_count == expected_launched
        if not st_platform.endswith("sim"):
            root_attrs = [inv.by_name()["simpler_run"].attrs for inv in invocations]
            assert all(
                key in attrs
                for attrs in root_attrs
                for key in ("run_id=", "slot=", "generation=", "dispatch_id=", "run_epoch=")
            )
            assert any("slot=1" in attrs and "generation=1" in attrs for attrs in root_attrs)

    def _run_and_validate_l2(  # noqa: PLR0913, PLR0915 -- lifecycle contract is intentionally sequential
        self,
        worker,
        callable_obj,
        case,
        rounds=1,
        skip_golden=False,
        enable_chip_swimlane=False,
        enable_dump_args=False,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
        output_prefix="",
    ):
        del rounds, skip_golden, enable_chip_swimlane, enable_dump_args
        del enable_pmu, enable_dep_gen, enable_scope_stats, output_prefix
        config = self._build_config(case["config"])
        chip_worker = worker._chip_worker
        assert chip_worker is not None
        supports_concurrent_prepare = bool(chip_worker._impl.supports_concurrent_native_prepare)
        chip_worker._register_callable_at_slot(_SLOT, callable_obj)
        native_run = None
        successor_run = None
        try:
            test_args = self.generate_args(case["params"])
            chip_args, output_names = _build_chip_task_args(test_args, self.CALLABLE["orchestration"]["signature"])
            golden_args = test_args.clone()
            self.compute_golden(golden_args, case["params"])

            stream_count_before_prepare = chip_worker.run_stream_set_create_count
            native_run = chip_worker._prepare_native_run_with_pipeline_lease(
                _SLOT, chip_args, _SLOT, _GENERATION, config=config
            )
            first_run = native_run
            expected_stream_count = stream_count_before_prepare + int(supports_concurrent_prepare)
            assert chip_worker.run_stream_set_create_count == expected_stream_count
            assert torch.count_nonzero(test_args.out) == 0, "prepare crossed the device launch fence"
            with pytest.raises(RuntimeError, match="unfinished native run|owns the runner|active predecessor"):
                chip_worker._prepare_native_run_with_pipeline_lease(_SLOT, chip_args, 1, _GENERATION, config=config)
            with pytest.raises(RuntimeError, match="unregister_callable failed"):
                chip_worker._unregister_slot(_SLOT)
            with pytest.raises(RuntimeError, match="register_callable failed"):
                chip_worker._register_callable_at_slot(1, callable_obj)

            # A prepared run can be abandoned explicitly. Finalize releases its
            # claim and registry dependencies without launching or copying back.
            chip_worker._finalize_native_run(native_run)
            native_run = None
            assert torch.count_nonzero(test_args.out) == 0
            chip_worker._unregister_slot(_SLOT)
            chip_worker._register_callable_at_slot(_SLOT, callable_obj)

            # Prepare initializes this run's diagnostics collectors, so abandoning
            # it must release them. Each collector refuses a second initialize()
            # while its shared memory is still mapped, so a collector retained by
            # an abandoned run makes the next prepare on this runner fail.
            with tempfile.TemporaryDirectory(prefix="simpler-abandon-diagnostics-") as abandon_dir:
                abandon_config = self._build_config(case["config"])
                abandon_config.enable_scope_stats = True
                abandon_config.output_prefix = abandon_dir
                abandon_args = self.generate_args(case["params"])
                abandon_chip_args, _ = _build_chip_task_args(abandon_args, self.CALLABLE["orchestration"]["signature"])
                for _ in range(2):
                    abandoned = chip_worker._prepare_native_run_with_pipeline_lease(
                        _SLOT, abandon_chip_args, _SLOT, _GENERATION, config=abandon_config
                    )
                    chip_worker._finalize_native_run(abandoned)
                assert torch.count_nonzero(abandon_args.out) == 0

            test_args = self.generate_args(case["params"])
            chip_args, output_names = _build_chip_task_args(test_args, self.CALLABLE["orchestration"]["signature"])
            golden_args = test_args.clone()
            self.compute_golden(golden_args, case["params"])
            native_run = chip_worker._prepare_native_run_with_pipeline_lease(
                _SLOT, chip_args, _SLOT, _GENERATION, config=config
            )
            assert first_run.generation == native_run.generation
            assert first_run.run_epoch != native_run.run_epoch
            with pytest.raises(RuntimeError, match="stale|wrong phase"):
                chip_worker._launch_native_run(first_run)
            chip_worker._launch_native_run(native_run)
            chip_worker._wait_native_run(native_run)
            assert chip_worker._poll_native_run(native_run)
            chip_worker._finalize_native_run(native_run)
            _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

            with pytest.raises(RuntimeError, match="stale|finalized|wrong phase"):
                chip_worker._poll_native_run(first_run)
            native_run = None

            # The existing blocking surface remains the compatibility path.
            second_args = self.generate_args(case["params"])
            second_chip_args, second_output_names = _build_chip_task_args(
                second_args, self.CALLABLE["orchestration"]["signature"]
            )
            second_golden = second_args.clone()
            self.compute_golden(second_golden, case["params"])
            chip_worker._run_slot(_SLOT, second_chip_args, config=config)
            _compare_outputs(second_args, second_golden, second_output_names, self.RTOL, self.ATOL)

            if supports_concurrent_prepare:

                def build_run_args():
                    run_args = self.generate_args(case["params"])
                    run_chip_args, run_output_names = _build_chip_task_args(
                        run_args, self.CALLABLE["orchestration"]["signature"]
                    )
                    run_golden = run_args.clone()
                    self.compute_golden(run_golden, case["params"])
                    return run_args, run_chip_args, run_output_names, run_golden

                # The successor owns a distinct bank and fresh stream while A
                # still owns the execution claim. A failed early launch must
                # leave B prepared so the same token can launch after A's
                # complete fence and finalization.
                active_args, active_chip_args, active_outputs, active_golden = build_run_args()
                successor_args, successor_chip_args, successor_outputs, successor_golden = build_run_args()
                stream_count = chip_worker.run_stream_set_create_count
                native_run = chip_worker._prepare_native_run_with_pipeline_lease(
                    _SLOT, active_chip_args, 0, _GENERATION + 1, config=config
                )
                assert chip_worker.run_stream_set_create_count == stream_count + 1
                chip_worker._launch_native_run(native_run)
                successor_run = chip_worker._prepare_native_run_with_pipeline_lease(
                    _SLOT, successor_chip_args, 1, _GENERATION, config=config
                )
                assert chip_worker.run_stream_set_create_count == stream_count + 2
                bank0 = chip_worker.arena_bank_gm_heap_base(0)
                bank1 = chip_worker.arena_bank_gm_heap_base(1)
                assert bank0 != 0
                assert bank1 != 0
                assert bank0 != bank1
                assert torch.count_nonzero(successor_args.out) == 0

                with pytest.raises(RuntimeError, match="launch_native_run failed") as claim_error:
                    chip_worker._launch_native_run(successor_run)
                assert "slot=1" in str(claim_error.value)
                assert "generation=1" in str(claim_error.value)
                assert "run_epoch=" in str(claim_error.value)

                chip_worker._wait_native_run(native_run)
                chip_worker._finalize_native_run(native_run)
                native_run = None
                _compare_outputs(active_args, active_golden, active_outputs, self.RTOL, self.ATOL)

                chip_worker._launch_native_run(successor_run)
                chip_worker._wait_native_run(successor_run)
                chip_worker._finalize_native_run(successor_run)
                successor_run = None
                _compare_outputs(successor_args, successor_golden, successor_outputs, self.RTOL, self.ATOL)

                with tempfile.TemporaryDirectory(prefix="simpler-native-diagnostics-") as output_dir:
                    diagnostic_config = self._build_config(case["config"])
                    diagnostic_config.enable_dep_gen = True
                    diagnostic_config.output_prefix = output_dir

                    # A diagnostic successor cannot overlap an ordinary active
                    # run, even though the predecessor otherwise permits one.
                    active_args, active_chip_args, active_outputs, active_golden = build_run_args()
                    native_run = chip_worker._prepare_native_run_with_pipeline_lease(
                        _SLOT, active_chip_args, 0, _GENERATION + 2, config=config
                    )
                    chip_worker._launch_native_run(native_run)
                    with pytest.raises(RuntimeError, match="active predecessor"):
                        chip_worker._prepare_native_run_with_pipeline_lease(
                            _SLOT, successor_chip_args, 1, _GENERATION + 1, config=diagnostic_config
                        )
                    chip_worker._wait_native_run(native_run)
                    chip_worker._finalize_native_run(native_run)
                    native_run = None
                    _compare_outputs(active_args, active_golden, active_outputs, self.RTOL, self.ATOL)

                    # A diagnostic predecessor also cannot admit an ordinary
                    # successor while it owns the execution claim.
                    active_args, active_chip_args, active_outputs, active_golden = build_run_args()
                    native_run = chip_worker._prepare_native_run_with_pipeline_lease(
                        _SLOT, active_chip_args, 0, _GENERATION + 3, config=diagnostic_config
                    )
                    chip_worker._launch_native_run(native_run)
                    with pytest.raises(RuntimeError, match="active predecessor"):
                        chip_worker._prepare_native_run_with_pipeline_lease(
                            _SLOT, successor_chip_args, 1, _GENERATION + 1, config=config
                        )
                    chip_worker._wait_native_run(native_run)
                    chip_worker._finalize_native_run(native_run)
                    native_run = None
                    _compare_outputs(active_args, active_golden, active_outputs, self.RTOL, self.ATOL)
        finally:
            for unfinished_run in (successor_run, native_run):
                if unfinished_run is None:
                    continue
                try:
                    chip_worker._finalize_native_run(unfinished_run)
                except Exception:
                    pass
            chip_worker._unregister_slot(_SLOT)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
