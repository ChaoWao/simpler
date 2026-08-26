from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HOST_COMM = REPO_ROOT / "src/a2a3/platform/onboard/host/comm_hccl.cpp"
HOST_CMAKE = REPO_ROOT / "src/a2a3/platform/onboard/host/CMakeLists.txt"


def test_a2a3_host_uses_unified_sdma_workspace_interface() -> None:
    source = HOST_COMM.read_text()

    assert '#include "pto/comm/workspace.hpp"' in source
    assert "sdma_workspace_manager.hpp" not in source
    assert "std::unique_ptr<pto::comm::sdma::SdmaWorkspaceManager>" not in source
    assert "pto::comm::Workspace sdma_workspace{}" in source
    assert "pto::comm::WorkspaceRequest req{}" in source
    assert "pto::comm::CreateWorkspace(pto::comm::DmaEngine::SDMA, req, &h->sdma_workspace)" in source
    assert "h->host_ctx.workSpace = reinterpret_cast<uint64_t>(h->sdma_workspace.addr)" in source
    assert "h->host_ctx.workSpaceSize = h->sdma_workspace.bytes" in source
    assert "16 * 1024" not in source


def test_a2a3_sdma_workspace_release_paths_are_explicit() -> None:
    source = HOST_COMM.read_text()

    assert "static void destroy_sdma_workspace(CommHandle h)" in source
    assert "pto::comm::DestroyWorkspace(&h->sdma_workspace)" in source
    assert "static void abandon_sdma_workspace(CommHandle h)" in source
    assert source.count("pto::comm::AbandonWorkspace(&h->sdma_workspace)") >= 2
    assert "destroy_sdma_workspace(h);" in source
    assert "abandon_sdma_workspace(h);" in source


def test_a2a3_host_cmake_maps_simpler_sdma_to_unified_workspace_macros() -> None:
    cmake = HOST_CMAKE.read_text()

    assert "SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=1" in cmake
    assert "PTO_COMM_WORKSPACE_SDMA_SUPPORTED=1" in cmake
    assert "PTO_COMM_WORKSPACE_URMA_SUPPORTED=0" in cmake
    assert "PTO_COMM_WORKSPACE_RDMA_SUPPORTED=0" in cmake
