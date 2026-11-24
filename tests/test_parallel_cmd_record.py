import pytest
import graphics.parallel_cmd_record as mod

CASES = [
    ("global_state_table", (1,)),
    ("allocator_fence_recycle", (1,)),
    ("cross_queue_dependency", (1,)),
    ("deferred_release", (1,)),
    ("fixed_record_order", (1,)),
    ("replay_check", (1,)),
    ("begin_frame", (1,)),
    ("record", (1,)),
    ("submit", (1,)),
    ("present", (1,)),
    ("validate_record", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
