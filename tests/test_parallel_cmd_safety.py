"""六个并行录制/资源安全修复点的回归测试。"""

from __future__ import annotations

import threading

import pytest

from graphics.parallel_cmd_record import (
    CommandAllocator,
    CommandRecorder,
    CrossQueueScheduler,
    DeferredReleaseQueue,
    FenceNotReady,
    FenceTimeline,
    ParallelCommandSystem,
    ReplayChecker,
    StateTable,
    allocator_fence_recycle,
    cross_queue_dependency,
    deferred_release,
    determinism,
    fixed_record_order,
    replay_check,
)


def test_global_state_table_is_shared_across_threads():
    table = StateTable()
    table.register("texture", "sampled")

    def transition_from_worker():
        table.transition("texture", "render_target", queue="graphics")

    thread = threading.Thread(target=transition_from_worker)
    thread.start()
    thread.join()

    state = table.get("texture")
    assert state is not None
    assert state.state == "render_target"
    assert table.barriers[-1].before == "sampled"
    assert table.barriers[-1].after == "render_target"


def test_allocator_cannot_reset_until_fence_completes():
    report = allocator_fence_recycle()
    assert report.ok is True
    assert report.unsafe_reset_blocked is True
    assert report.recycled

    fences = FenceTimeline()
    allocator = CommandAllocator(fences)
    block = allocator.allocate(frame=1)
    allocator.mark_submitted(block.block_id, fence=5)

    assert allocator.recycle(completed_fence=4) == []
    with pytest.raises(FenceNotReady):
        allocator.reset(block.block_id, completed_fence=4)

    fences.signal(5)
    assert allocator.recycle() == [block.block_id]
    reused = allocator.allocate(frame=2)
    assert reused.block_id == block.block_id
    assert reused.reset_count >= 1


def test_cross_queue_dependency_orders_producer_before_consumer():
    report = cross_queue_dependency("vertex_buffer")
    assert report.ok is True
    assert report.producer_fence == 1
    assert report.consumer_wait == 1

    scheduler = CrossQueueScheduler()
    compute = scheduler.submit(
        "compute",
        [{"op": "read_vb", "resource": "vb", "access": "read"}],
        waits=[7],
    )
    graphics = scheduler.submit(
        "graphics",
        [{"op": "write_vb", "resource": "vb", "access": "write"}],
        fence=7,
    )

    schedule = scheduler.schedule_report()
    assert schedule.ok is True
    assert schedule.order.index(graphics.id) < schedule.order.index(compute.id)


def test_missing_cross_queue_wait_is_rejected_even_if_recorded_first():
    scheduler = CrossQueueScheduler()
    scheduler.submit(
        "compute",
        [{"op": "read_vb", "resource": "vb", "access": "read"}],
    )
    scheduler.submit(
        "graphics",
        [{"op": "write_vb", "resource": "vb", "access": "write"}],
        fence=1,
    )

    report = scheduler.schedule_report()
    assert report.ok is False
    assert any("without waiting fence" in error for error in report.errors)


def test_deferred_handle_is_not_reusable_before_fence():
    report = deferred_release()
    assert report.ok is True
    assert report.early_collect == []
    assert report.reused_handle_after_fence == report.handle

    handles = DeferredReleaseQueue()
    handle = handles.create("texture")
    handles.request_release(handle, fence=3)

    assert handles.collect(2) == []
    assert handles.is_pending(handle)
    assert handle not in handles.free_ids

    assert handles.collect(3) == [handle]
    assert handles.create("texture") == handle


def test_fixed_packet_order_is_independent_of_thread_arrival():
    report = fixed_record_order()
    assert report.ok is True
    assert report.deterministic is True
    assert report.order == [
        "clear_color",
        "bind_backbuffer",
        "shadow_pass",
        "main_pass",
        "draw_scene",
    ]

    for _ in range(10):
        assert fixed_record_order().hash == report.hash


def test_replay_check_detects_post_seal_tampering():
    recorder = CommandRecorder(frame=1)
    recorder.record(
        [
            {"op": "clear", "sequence": 0},
            {"op": "draw", "sequence": 1},
        ],
        order=0,
    )
    recorder.seal()
    assert ReplayChecker().check(recorder).match is True

    recorder.build()[1].op = "tampered"
    mismatch = replay_check(recorder)
    assert mismatch.match is False
    assert mismatch.mismatch_count == 1
    assert mismatch.first_mismatch.index == 1


def test_end_to_end_frame_recycles_only_after_present_fence():
    system = ParallelCommandSystem()
    system.begin_frame(7)
    system.record(
        [{"op": "transition", "resource": "tex", "state": "render_target"}]
    )
    submission = system.submit(queue="graphics")

    assert system.frame is not None
    allocation_id = system.frame.allocation_id
    assert system.allocator.allocations[allocation_id].status == "submitted"

    present = system.present(7)
    assert present.fence == submission.fence
    assert allocation_id in present.recycled
    assert system.allocator.allocations[allocation_id].status == "free"

    validation = system.validate()
    assert validation.ok is True, validation.errors
    audit = system.audit()
    assert audit.ok is True
    assert audit.metrics.frames == 1
    assert audit.metrics.barriers == 1


def test_determinism_report_is_stable():
    first = determinism()
    second = determinism()
    assert first.deterministic is True
    assert first.replay_match is True
    assert first.hashes == second.hashes
