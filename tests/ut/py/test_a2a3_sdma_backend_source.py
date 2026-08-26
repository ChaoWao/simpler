from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SDMA_KERNEL = REPO_ROOT / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/backend/sdma/sdma_completion_kernel.h"
SDMA_WAIT = REPO_ROOT / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_async_wait.h"
SDMA_MAILBOX = REPO_ROOT / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/aicore_completion_mailbox.h"
SDMA_TYPES = REPO_ROOT / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/aicore_completion_mailbox_types.h"


def test_a2a3_sdma_backend_uses_post_done_completion_flow() -> None:
    kernel = SDMA_KERNEL.read_text()
    wait = SDMA_WAIT.read_text()

    assert "PrepareEventCheck" not in kernel
    assert "GetEventRecord" not in kernel
    assert "session.sdmaSession" not in kernel
    assert "COMPLETION_TYPE_SDMA_POST_DONE" in kernel
    assert "post_id" in kernel

    assert "COMPLETION_TYPE_SDMA_POST_DONE" in wait
    assert "backend_cookie" in wait
    assert "poll_sdma_post_done" in wait


def test_a2a3_sdma_mailbox_carries_backend_cookie() -> None:
    mailbox = SDMA_MAILBOX.read_text()
    types = SDMA_TYPES.read_text()

    assert "uint64_t backend_cookie" in mailbox
    assert "backend_cookie" in mailbox and "try_push_condition(" in mailbox
    assert "backend_cookie" in types
    assert "COMPLETION_TYPE_SDMA_POST_DONE" in types
