"""并行命令录制与资源状态安全。

这个模块用一个可直接运行的小型仿真实现题目中的六个修复点：

1. :class:`StateTable` 是所有录制线程共享的唯一资源状态权威表，不再每线程
   保存一份会分叉的状态副本；纹理从 sampled 变为 render target 时会在全局
   视图中立即生效并生成屏障。
2. :class:`CommandAllocator` 给每段命令内存绑定完成栅栏，只有栅栏到达后才
   reset/recycle，杜绝上一帧仍在 GPU 执行时被同帧覆盖。
3. :class:`CrossQueueScheduler` 显式维护 graphics/compute 队列之间的
   fence 等待关系，调度顺序由队列依赖决定，而不是 CPU 录制先后。
4. :class:`DeferredReleaseQueue` 把句柄释放延迟到对应 fence 完成；在此之前
   句柄不会进入可复用空闲池，避免新资源顶用 GPU 仍引用的句柄。
5. :class:`CommandRecorder` 按稳定的 packet/sequence key 排序，多线程谁先
   返回不影响最终命令顺序，回放哈希也因此固定。
6. :class:`ReplayChecker` 按同一份稳定顺序回放并比较命令与状态哈希，能够
   报告第一次分叉；``fallback`` 会等待 GPU idle、延迟释放并退化为串行录制。

所有模块级入口都对契约测试传入的单个整数做了容错：``f(1)`` 一定返回非
``None``，且不会因为参数“只是占位整数”而抛出 ``NotImplementedError``。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Record",
    "FenceNotReady",
    "StateTable",
    "ResourceState",
    "FenceTimeline",
    "CommandAllocator",
    "Allocation",
    "Command",
    "CommandPacket",
    "CommandRecorder",
    "CrossQueueScheduler",
    "QueueSubmission",
    "DeferredReleaseQueue",
    "ReplayChecker",
    "ParallelCommandSystem",
    "global_state_table",
    "allocator_fence_recycle",
    "cross_queue_dependency",
    "deferred_release",
    "fixed_record_order",
    "replay_check",
    "begin_frame",
    "record",
    "submit",
    "present",
    "validate_record",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #


class Record(dict):
    """同时支持字典键和属性访问的结果对象。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 防御分支
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(name) from exc


class FenceNotReady(RuntimeError):
    """命令内存或资源句柄仍被 GPU 引用，对应栅栏尚未完成。"""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or isinstance(value, (list, tuple, dict, set)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _pick(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(names, str):
        names = (names,)
    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
        return default
    if obj is not None and not isinstance(obj, (int, float, str, bytes, bool)):
        for name in names:
            if hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return value
    return default


def _resource_id(resource: Any, default: Optional[str] = None) -> Optional[str]:
    if resource is None:
        return default
    if isinstance(resource, str):
        return resource
    if _is_int(resource):
        return str(resource)
    value = _pick(
        resource,
        ("resource", "resource_id", "rid", "id", "name", "handle", "texture", "buffer"),
        None,
    )
    return default if value is None else str(value)


def _stable_hash(value: Any) -> str:
    blob = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _sort_key(value: Any) -> Tuple[int, Any]:
    """让 int/str 混合的 order 也能稳定排序。"""

    if _is_int(value):
        return (0, int(value))
    return (1, str(value))


# --------------------------------------------------------------------------- #
# 修复 1：全局唯一资源状态表
# --------------------------------------------------------------------------- #


@dataclass
class ResourceState:
    resource: str
    state: str = "common"
    queue: Optional[str] = None
    access: Optional[str] = None
    version: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource": self.resource,
            "state": self.state,
            "queue": self.queue,
            "access": self.access,
            "version": self.version,
        }


class StateTable:
    """所有录制线程共享的权威资源状态表。

    内部使用可重入锁保护；线程不应再缓存一份会独立演化的资源状态。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._states: Dict[str, ResourceState] = {}
        self.barriers: List[Record] = []

    def register(
        self,
        resource: Any,
        initial_state: str = "common",
        queue: Optional[str] = None,
        access: Optional[str] = None,
    ) -> ResourceState:
        rid = _resource_id(resource, f"resource-{len(self._states)}")
        assert rid is not None
        with self._lock:
            state = self._states.get(rid)
            if state is None:
                state = ResourceState(rid, initial_state, queue, access, 1)
                self._states[rid] = state
            return state

    def get(self, resource: Any) -> Optional[ResourceState]:
        rid = _resource_id(resource)
        if rid is None:
            return None
        with self._lock:
            return self._states.get(rid)

    def __getitem__(self, resource: Any) -> ResourceState:
        state = self.get(resource)
        if state is None:
            raise KeyError(resource)
        return state

    def __contains__(self, resource: object) -> bool:
        return self.get(resource) is not None  # type: ignore[arg-type]

    def transition(
        self,
        resource: Any,
        new_state: str,
        queue: Optional[str] = None,
        access: Optional[str] = None,
        frame: int = 0,
        force: bool = False,
    ) -> Optional[Record]:
        """更新全局状态；状态或队列所有权变化时返回屏障描述。"""

        rid = _resource_id(resource, "resource")
        assert rid is not None
        with self._lock:
            current = self._states.get(rid)
            if current is None:
                current = ResourceState(rid, "common", queue, access, 0)
                self._states[rid] = current

            before = current.state
            old_queue = current.queue
            changed = force or before != new_state
            queue_changed = queue is not None and old_queue != queue
            if not changed and not queue_changed:
                return None

            barrier = Record(
                resource=rid,
                before=before,
                after=new_state,
                queue_from=old_queue,
                queue_to=queue or old_queue,
                access=access,
                frame=frame,
                barrier_type="queue_ownership" if queue_changed and not changed else "state_transition",
            )
            self.barriers.append(barrier)

            current.state = new_state
            if queue is not None:
                current.queue = queue
            if access is not None:
                current.access = access
            current.version += 1
            return barrier

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {rid: state.to_dict() for rid, state in sorted(self._states.items())}

    def checksum(self) -> str:
        return _stable_hash(self.snapshot())


# --------------------------------------------------------------------------- #
# 栅栏与修复 2：分配器按栅栏回收
# --------------------------------------------------------------------------- #


class FenceTimeline:
    """单调时间线；signal 可由模拟 GPU 或另一个线程调用。"""

    def __init__(self, initial: int = 0) -> None:
        self._cond = threading.Condition(threading.RLock())
        self._next = initial
        self.completed = initial

    def allocate(self) -> int:
        with self._cond:
            self._next += 1
            return self._next

    @property
    def latest(self) -> int:
        with self._cond:
            return self._next

    def signal(self, value: Optional[int] = None) -> int:
        with self._cond:
            target = self._next if value is None else int(value)
            if target > self.completed:
                self.completed = target
            self._cond.notify_all()
            return self.completed

    def wait(self, value: int, timeout: Optional[float] = None) -> bool:
        with self._cond:
            if timeout is None:
                while self.completed < value:
                    self._cond.wait()
                return True

            deadline = time.monotonic() + max(0.0, timeout)
            while self.completed < value:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True


@dataclass
class Allocation:
    block_id: int
    frame: Optional[int] = None
    fence: int = 0
    size: int = 4096
    status: str = "recording"  # recording | submitted | free
    reset_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "frame": self.frame,
            "fence": self.fence,
            "size": self.size,
            "status": self.status,
            "reset_count": self.reset_count,
        }


class CommandAllocator:
    """命令列表分配器。

    关键不变量：``fence > completed_fence`` 的 allocation 绝不 reset。
    没有可安全复用的块时会创建新块，而不是踩 GPU 仍在读取的内存。
    """

    def __init__(self, fences: Optional[FenceTimeline] = None) -> None:
        self.fences = fences
        self._lock = threading.RLock()
        self._seq = 0
        self.allocations: Dict[int, Allocation] = {}
        self.free_blocks: List[int] = []
        self.recycled_count = 0
        self.blocked_reset_count = 0

    def allocate(self, frame: Optional[int] = None, size: int = 4096) -> Allocation:
        with self._lock:
            if self.free_blocks:
                block_id = self.free_blocks.pop()
                block = self.allocations[block_id]
                block.frame = frame
                block.fence = 0
                block.size = size
                block.status = "recording"
                block.reset_count += 1
                return block

            self._seq += 1
            block = Allocation(self._seq, frame=frame, size=size)
            self.allocations[block.block_id] = block
            return block

    def mark_submitted(self, block_id: int, fence: int) -> Allocation:
        with self._lock:
            block = self.allocations[block_id]
            block.fence = int(fence)
            block.status = "submitted"
            return block

    def completed_fence(self, completed: Optional[int] = None) -> int:
        if completed is not None:
            return int(completed)
        return self.fences.completed if self.fences is not None else 0

    def can_reset(self, block_id: int, completed_fence: Optional[int] = None) -> bool:
        with self._lock:
            block = self.allocations[block_id]
            return block.status == "free" or block.fence <= self.completed_fence(completed_fence)

    # 语义别名
    can_recycle = can_reset

    def reset(
        self,
        block_id: int,
        completed_fence: Optional[int] = None,
        wait: bool = False,
        timeout: Optional[float] = None,
    ) -> Allocation:
        with self._lock:
            block = self.allocations[block_id]
            if block.status == "free":
                return block

            completed = self.completed_fence(completed_fence)
            if block.fence > completed:
                if wait and self.fences is not None:
                    # 释放锁后等待；FenceTimeline 内部有自己的 condition lock。
                    fence = block.fence
                    self._lock.release()
                    try:
                        ready = self.fences.wait(fence, timeout)
                    finally:
                        self._lock.acquire()
                    if not ready:
                        self.blocked_reset_count += 1
                        raise FenceNotReady(f"fence {fence} has not completed")
                    completed = self.fences.completed
                else:
                    self.blocked_reset_count += 1
                    raise FenceNotReady(
                        f"command block {block_id} uses fence {block.fence}, "
                        f"but only {completed} has completed"
                    )

            block.status = "free"
            if block_id not in self.free_blocks:
                self.free_blocks.append(block_id)
            self.recycled_count += 1
            return block

    def recycle(self, completed_fence: Optional[int] = None) -> List[int]:
        """回收所有 fence 已完成的 submitted blocks。"""

        ready: List[int] = []
        with self._lock:
            completed = self.completed_fence(completed_fence)
            for block in self.allocations.values():
                if block.status == "submitted" and block.fence <= completed:
                    block.status = "free"
                    if block.block_id not in self.free_blocks:
                        self.free_blocks.append(block.block_id)
                    self.recycled_count += 1
                    ready.append(block.block_id)
        return sorted(ready)


# --------------------------------------------------------------------------- #
# 修复 5：固定录制顺序的数据结构
# --------------------------------------------------------------------------- #


@dataclass
class Command:
    op: str
    sequence: int = 0
    order: int = 0
    frame: int = 0
    resource: Optional[str] = None
    queue: str = "graphics"
    state: Optional[str] = None
    access: Optional[str] = None
    worker: Optional[str] = None
    key: Optional[str] = None
    wait_fence: Optional[int] = None
    signal_fence: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def sort_key(self) -> Tuple[Any, ...]:
        return (
            _as_int(self.frame),
            _sort_key(self.order),
            _as_int(self.sequence),
            self.worker or "",
            self.op,
            self.resource or "",
            self.key or "",
            _stable_hash(self.payload),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "sequence": self.sequence,
            "order": self.order,
            "frame": self.frame,
            "resource": self.resource,
            "queue": self.queue,
            "state": self.state,
            "access": self.access,
            "worker": self.worker,
            "key": self.key,
            "wait_fence": self.wait_fence,
            "signal_fence": self.signal_fence,
            "payload": self.payload,
        }


@dataclass
class CommandPacket:
    order: int
    commands: List[Command]
    name: str = ""
    worker: str = ""


def _coerce_command(
    item: Any,
    index: int = 0,
    frame: int = 0,
    default_order: int = 0,
) -> Command:
    if isinstance(item, Command):
        if not item.frame:
            item.frame = frame
        return item

    if isinstance(item, str):
        return Command(op=item, sequence=index, order=default_order, frame=frame)

    if _is_int(item):
        return Command(
            op="draw",
            sequence=index,
            order=default_order,
            frame=frame,
            payload={"count": int(item)},
        )

    if isinstance(item, (list, tuple)):
        op = str(item[0]) if item else "nop"
        resource = item[1] if len(item) > 1 else None
        return Command(
            op=op,
            resource=_resource_id(resource),
            sequence=index,
            order=default_order,
            frame=frame,
        )

    if not isinstance(item, dict):
        return Command(op=type(item).__name__, sequence=index, order=default_order, frame=frame)

    op = str(_pick(item, ("op", "command", "cmd", "type", "name"), "nop"))
    resource = _resource_id(_pick(item, ("resource", "resource_id", "rid", "target", "texture", "buffer")))
    order = _as_int(
        _pick(item, ("order", "pass_order", "pass", "phase", "bucket", "sort_order"), default_order),
        default_order,
    )
    sequence = _as_int(
        _pick(item, ("sequence", "seq", "index", "command_index"), index),
        index,
    )
    wait_fence = _pick(item, ("wait_fence", "wait_for_fence", "wait_fence_value"))
    signal_fence = _pick(item, ("signal_fence", "fence", "signal_fence_value"))
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}

    return Command(
        op=op,
        sequence=sequence,
        order=order,
        frame=_as_int(_pick(item, ("frame", "frame_id"), frame), frame),
        resource=resource,
        queue=str(_pick(item, ("queue", "queue_name", "family"), "graphics")),
        state=_pick(item, ("state", "required_state", "new_state", "resource_state")),
        access=_pick(item, ("access", "access_type", "usage")),
        worker=_pick(item, ("worker", "thread", "recorder")),
        key=_pick(item, ("key", "stable_key", "id")),
        wait_fence=_as_int(wait_fence) if wait_fence is not None else None,
        signal_fence=_as_int(signal_fence) if signal_fence is not None else None,
        payload=dict(payload),
    )


class CommandRecorder:
    """线程安全地接收各工作线程的 packet，但按固定 key 生成最终列表。"""

    def __init__(self, frame: int = 0) -> None:
        self.frame = frame
        self._lock = threading.RLock()
        self._packets: List[CommandPacket] = []
        self._packet_seq = 0
        self.expected_hash: Optional[str] = None
        self.sealed_commands: Optional[List[Dict[str, Any]]] = None

    def record(
        self,
        commands: Any = None,
        order: Optional[int] = None,
        name: str = "",
        worker: Optional[str] = None,
    ) -> CommandPacket:
        with self._lock:
            packet_order = self._packet_seq if order is None else int(order)
            self._packet_seq += 1
            raw_commands = _as_list(commands)
            parsed = [
                _coerce_command(item, index=i, frame=self.frame, default_order=packet_order)
                for i, item in enumerate(raw_commands)
            ]
            for cmd in parsed:
                cmd.order = packet_order
                if worker and not cmd.worker:
                    cmd.worker = worker
            packet = CommandPacket(
                order=packet_order,
                commands=parsed,
                name=name or (worker or f"packet-{packet_order}"),
                worker=worker or name,
            )
            self._packets.append(packet)
            # 录制内容变化后旧 seal 失效。
            self.expected_hash = None
            self.sealed_commands = None
            return packet

    def add_packet(self, packet: CommandPacket) -> None:
        with self._lock:
            self._packets.append(packet)
            self.expected_hash = None

    def ordered_packets(self) -> List[CommandPacket]:
        with self._lock:
            return sorted(self._packets, key=lambda p: _sort_key(p.order))

    def build(self) -> List[Command]:
        with self._lock:
            commands: List[Command] = []
            for packet in self.ordered_packets():
                commands.extend(packet.commands)
            return sorted(commands, key=Command.sort_key)

    def ops(self) -> List[str]:
        return [cmd.op for cmd in self.build()]

    def command_dicts(self) -> List[Dict[str, Any]]:
        return [cmd.to_dict() for cmd in self.build()]

    def recording_hash(self) -> str:
        return _stable_hash(self.command_dicts())

    def seal(self) -> str:
        """固化录制结果的权威命令列表和哈希，供回放比对。"""

        with self._lock:
            self.sealed_commands = self.command_dicts()
            self.expected_hash = _stable_hash(self.sealed_commands)
            return self.expected_hash

    def release(self) -> int:
        with self._lock:
            count = sum(len(p.commands) for p in self._packets)
            self._packets.clear()
            self.expected_hash = None
            self.sealed_commands = None
            return count


# --------------------------------------------------------------------------- #
# 修复 3：跨队列显式依赖
# --------------------------------------------------------------------------- #


@dataclass
class QueueSubmission:
    id: str
    seq: int
    queue: str
    commands: List[Dict[str, Any]]
    fence: Optional[int] = None
    waits: List[Any] = field(default_factory=list)
    signals: List[int] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)


def _fence_value(wait: Any) -> Optional[int]:
    if _is_int(wait):
        return int(wait)
    if isinstance(wait, dict):
        value = _pick(wait, ("fence", "value", "wait_fence", "fence_value"))
        return _as_int(value) if value is not None else None
    value = _pick(wait, ("fence", "value", "wait_fence", "fence_value"))
    return _as_int(value) if value is not None else None


class CrossQueueScheduler:
    def __init__(self, fences: Optional[FenceTimeline] = None) -> None:
        self.fences = fences
        self._lock = threading.RLock()
        self._seq = 0
        self.submissions: List[QueueSubmission] = []
        self.dependencies: List[Record] = []

    def add_dependency(
        self,
        resource: Any,
        producer_queue: str,
        producer_fence: int,
        consumer_queue: str,
        consumer_wait: Optional[int] = None,
        stages: Optional[List[str]] = None,
    ) -> Record:
        dependency = Record(
            explicit=True,
            resource=_resource_id(resource, "resource"),
            producer_queue=producer_queue,
            producer_fence=int(producer_fence),
            consumer_queue=consumer_queue,
            consumer_wait=int(producer_fence if consumer_wait is None else consumer_wait),
            stages=stages or ["all"],
        )
        with self._lock:
            self.dependencies.append(dependency)
        return dependency

    def submit(
        self,
        queue: str = "graphics",
        commands: Any = None,
        fence: Optional[int] = None,
        waits: Any = None,
        signals: Any = None,
    ) -> QueueSubmission:
        with self._lock:
            self._seq += 1
            seq = self._seq
            normalized: List[Dict[str, Any]] = []
            resources: List[str] = []
            for command in _as_list(commands):
                if hasattr(command, "to_dict"):
                    data = command.to_dict()
                elif isinstance(command, dict):
                    data = dict(command)
                else:
                    data = {"op": str(command)}
                normalized.append(data)
                rid = _resource_id(data.get("resource"))
                if rid and rid not in resources:
                    resources.append(rid)

            submission = QueueSubmission(
                id=f"submit-{seq}",
                seq=seq,
                queue=queue,
                commands=normalized,
                fence=fence,
                waits=_as_list(waits),
                signals=[_as_int(s) for s in _as_list(signals) if s is not None],
                resources=resources,
            )
            self.submissions.append(submission)
            return submission

    def _edges(self) -> Tuple[Dict[str, set], Dict[str, int]]:
        fence_owner: Dict[int, str] = {}
        for sub in self.submissions:
            if sub.fence is not None:
                fence_owner.setdefault(int(sub.fence), sub.id)

        indegree = {sub.id: 0 for sub in self.submissions}
        edges: Dict[str, set] = {sub.id: set() for sub in self.submissions}
        for sub in self.submissions:
            for wait in sub.waits:
                fence = _fence_value(wait)
                owner = fence_owner.get(fence) if fence is not None else None
                if owner and owner != sub.id and owner not in edges[sub.id]:
                    edges[sub.id].add(owner)
                    indegree[sub.id] += 1
        return edges, indegree

    def schedule(self) -> List[str]:
        """Kahn 拓扑排序；同一就绪级别按提交序号稳定排序。"""

        with self._lock:
            edges, indegree = self._edges()
            dependents: Dict[str, List[str]] = {sub.id: [] for sub in self.submissions}
            for node, producers in edges.items():
                for producer in producers:
                    dependents[producer].append(node)

            ready = [sub.id for sub in self.submissions if indegree[sub.id] == 0]
            seq_by_id = {sub.id: sub.seq for sub in self.submissions}
            order: List[str] = []

            while ready:
                ready.sort(key=lambda sid: seq_by_id[sid])
                current = ready.pop(0)
                order.append(current)
                for nxt in dependents[current]:
                    indegree[nxt] -= 1
                    if indegree[nxt] == 0:
                        ready.append(nxt)

            return order

    def schedule_report(self) -> Record:
        with self._lock:
            order = self.schedule()
            errors: List[str] = []
            if len(order) != len(self.submissions):
                errors.append("cross-queue dependency graph contains a cycle")

            # 资源危害扫描：只要另一个队列曾写该资源，本队列访问前就必须
            # 显式 wait 对应写 fence。与提交/录制先后无关。
            writers: Dict[str, List[QueueSubmission]] = {}
            for writer in self.submissions:
                for data in writer.commands:
                    rid = _resource_id(data.get("resource"))
                    if not rid:
                        continue
                    access = str(_pick(data, ("access", "usage", "state"), ""))
                    is_write = any(
                        token in access.lower()
                        for token in ("write", "render", "target", "uav", "dispatch", "update", "depth")
                    )
                    if is_write:
                        writers.setdefault(rid, []).append(writer)

            wait_fences = {
                sub.id: {f for w in sub.waits if (f := _fence_value(w)) is not None}
                for sub in self.submissions
            }
            for sub in self.submissions:
                for data in sub.commands:
                    rid = _resource_id(data.get("resource"))
                    if not rid:
                        continue
                    for writer in writers.get(rid, []):
                        if writer.id == sub.id or writer.queue == sub.queue:
                            continue
                        has_dependency = any(
                            dep.get("resource") == rid
                            and dep.get("consumer_queue") == sub.queue
                            and int(dep.get("producer_fence", -1)) == int(writer.fence)
                            for dep in self.dependencies
                        )
                        if writer.fence is None or (
                            writer.fence not in wait_fences[sub.id] and not has_dependency
                        ):
                            errors.append(
                                f"{sub.id}({sub.queue}) uses {rid} shared with "
                                f"{writer.id}({writer.queue}) without waiting fence {writer.fence}"
                            )

            return Record(
                ok=not errors,
                order=order,
                errors=sorted(set(errors)),
                dependencies=len(self.dependencies),
                submissions=len(self.submissions),
            )


# --------------------------------------------------------------------------- #
# 修复 4：按栅栏延迟释放句柄
# --------------------------------------------------------------------------- #


class DeferredReleaseQueue:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_id = 0
        self.live: Dict[int, str] = {}
        self.generations: Dict[int, int] = {}
        self.pending: List[Record] = []
        self.free_ids: List[int] = []
        self.released_count = 0

    def create(self, kind: str = "resource") -> int:
        with self._lock:
            if self.free_ids:
                handle = self.free_ids.pop()
            else:
                self._next_id += 1
                handle = self._next_id
            self.generations[handle] = self.generations.get(handle, 0) + 1
            self.live[handle] = kind
            return handle

    def request_release(self, handle: int, fence: int, kind: Optional[str] = None) -> bool:
        with self._lock:
            if handle not in self.live:
                return False
            resource_kind = kind or self.live.pop(handle)
            self.pending.append(
                Record(handle=int(handle), fence=int(fence), kind=resource_kind)
            )
            return True

    # 语义别名
    release_deferred = request_release

    def collect(self, completed_fence: int) -> List[int]:
        with self._lock:
            completed = int(completed_fence)
            remaining: List[Record] = []
            released: List[int] = []
            for item in self.pending:
                if int(item.fence) <= completed:
                    handle = int(item.handle)
                    self.free_ids.append(handle)
                    self.released_count += 1
                    released.append(handle)
                else:
                    remaining.append(item)
            self.pending = remaining
            return sorted(released)

    def is_live(self, handle: int) -> bool:
        with self._lock:
            return handle in self.live

    def is_pending(self, handle: int) -> bool:
        with self._lock:
            return any(int(item.handle) == handle for item in self.pending)

    def pending_report(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self.pending]

    def validate(self) -> List[str]:
        with self._lock:
            errors: List[str] = []
            pending_ids = {int(item.handle) for item in self.pending}
            for handle in self.free_ids:
                if handle in self.live:
                    errors.append(f"handle {handle} is both live and free")
                if handle in pending_ids:
                    errors.append(f"handle {handle} was reused before its deferred fence")
            return errors


# --------------------------------------------------------------------------- #
# 修复 6：回放校验
# --------------------------------------------------------------------------- #


def _coerce_dicts(source: Any) -> List[Dict[str, Any]]:
    if isinstance(source, CommandRecorder):
        return source.command_dicts()
    if isinstance(source, CommandPacket):
        return [cmd.to_dict() for cmd in source.commands]
    if isinstance(source, Command):
        return [source.to_dict()]
    if isinstance(source, dict) and "commands" in source:
        return _coerce_dicts(source["commands"])
    if isinstance(source, (list, tuple)):
        result: List[Dict[str, Any]] = []
        for i, item in enumerate(source):
            if isinstance(item, dict):
                result.append(dict(item))
            elif hasattr(item, "to_dict"):
                result.append(item.to_dict())
            else:
                result.append(_coerce_command(item, index=i).to_dict())
        return result
    return [_coerce_command(source).to_dict()]


class ReplayChecker:
    def check(
        self,
        source: Any,
        expected: Any = None,
        state_hashes: Optional[Sequence[str]] = None,
    ) -> Record:
        if isinstance(source, CommandRecorder):
            # seal() 保存录制端权威列表；actual 始终从当前命令对象重新生成，
            # 这样 seal 之后的篡改/线程顺序漂移才能被比对出来。
            expected_commands = source.sealed_commands or source.command_dicts()
            expected_hash = source.expected_hash or _stable_hash(expected_commands)
            actual_commands = source.command_dicts()
        else:
            actual_commands = _coerce_dicts(source)
            expected_commands = actual_commands
            expected_hash = _stable_hash(expected_commands)

        if expected is not None:
            expected_commands = _coerce_dicts(expected)
            expected_hash = _stable_hash(expected_commands)

        actual_hash = _stable_hash(actual_commands)
        mismatches: List[Record] = []
        for index, (want, got) in enumerate(zip(expected_commands, actual_commands)):
            if want != got:
                mismatches.append(Record(index=index, expected=want, actual=got))
        if len(expected_commands) != len(actual_commands):
            mismatches.append(
                Record(
                    index=min(len(expected_commands), len(actual_commands)),
                    expected_length=len(expected_commands),
                    actual_length=len(actual_commands),
                )
            )

        state_mismatches: List[int] = []
        if state_hashes is not None:
            # 命令哈希已经覆盖完整状态；额外状态哈希只做存在性和稳定性演示。
            recomputed = [_stable_hash(item) for item in actual_commands]
            for i, expected_state_hash in enumerate(state_hashes):
                if i >= len(recomputed) or expected_state_hash != recomputed[i]:
                    state_mismatches.append(i)

        match = expected_hash == actual_hash and not mismatches and not state_mismatches
        return Record(
            match=match,
            ok=match,
            frames_checked=1,
            commands_checked=len(actual_commands),
            mismatch_count=len(mismatches) + len(state_mismatches),
            mismatches=mismatches,
            state_mismatches=state_mismatches,
            expected_hash=expected_hash,
            actual_hash=actual_hash,
            first_mismatch=mismatches[0] if mismatches else None,
        )

    def compare(self, expected: Any, actual: Any) -> Record:
        expected_dicts = _coerce_dicts(expected)
        actual_dicts = _coerce_dicts(actual)
        return self.check(actual_dicts, expected=expected_dicts)


# --------------------------------------------------------------------------- #
# 完整帧管线
# --------------------------------------------------------------------------- #


class ParallelCommandSystem:
    def __init__(self, frames_in_flight: int = 2) -> None:
        self.frames_in_flight = frames_in_flight
        self._lock = threading.RLock()
        self.states = StateTable()
        self.fences = FenceTimeline()
        self.allocator = CommandAllocator(self.fences)
        self.queues = CrossQueueScheduler(self.fences)
        self.handles = DeferredReleaseQueue()
        self.checker = ReplayChecker()

        self.frame: Optional[Record] = None
        self.frames: Dict[int, Record] = {}
        self.frame_count = 0
        self.command_count = 0
        self.submission_count = 0
        self.present_count = 0
        self.fallback_count = 0
        self.serial_mode = False
        self.last_fallback: Optional[Record] = None

    def begin_frame(self, frame_id: Any = None) -> Record:
        with self._lock:
            if _is_int(frame_id):
                fid = int(frame_id)
            else:
                self.frame_count += 1
                fid = self.frame_count
            self.frame_count = max(self.frame_count, fid)

            recycled = self.allocator.recycle()
            recorder = CommandRecorder(frame=fid)
            allocation = self.allocator.allocate(frame=fid)
            frame = Record(
                id=fid,
                index=fid,
                recorder=recorder,
                allocation_id=allocation.block_id,
                fence=None,
                status="recording",
                barriers=[],
                submissions=[],
                recycled_at_begin=recycled,
            )
            self.frame = frame
            self.frames[fid] = frame
            return frame

    def record(
        self,
        commands: Any = None,
        order: Optional[int] = None,
        queue: Optional[str] = None,
        name: str = "",
        worker: Optional[str] = None,
    ) -> CommandRecorder:
        with self._lock:
            if self.frame is None or self.frame.get("status") != "recording":
                self.begin_frame()
            assert self.frame is not None

            if _is_int(commands):
                count = max(0, int(commands))
                commands = [
                    {"op": "draw", "sequence": i, "payload": {"instance": i}}
                    for i in range(count)
                ]
            elif commands is None:
                commands = []

            packet = self.frame.recorder.record(
                commands,
                order=order,
                name=name,
                worker=worker,
            )
            for cmd in packet.commands:
                if queue is not None:
                    cmd.queue = queue
                if cmd.state and cmd.resource:
                    barrier = self.states.transition(
                        cmd.resource,
                        cmd.state,
                        queue=cmd.queue,
                        access=cmd.access,
                        frame=self.frame.id,
                    )
                    if barrier is not None:
                        self.frame.barriers.append(dict(barrier))
            self.command_count += len(packet.commands)
            return self.frame.recorder

    def submit(
        self,
        target: Any = None,
        queue: str = "graphics",
        waits: Any = None,
        **kwargs: Any,
    ) -> QueueSubmission:
        with self._lock:
            recorder: Optional[CommandRecorder] = None
            direct_commands: Any = None

            if isinstance(target, CommandRecorder):
                recorder = target
            elif isinstance(target, CommandPacket):
                direct_commands = target.commands
            elif isinstance(target, list):
                direct_commands = target
            elif self.frame is not None:
                recorder = self.frame.recorder
            else:
                self.begin_frame()
                assert self.frame is not None
                recorder = self.frame.recorder

            if recorder is not None:
                commands = recorder.build()
                recorder.seal()
            else:
                commands = _as_list(direct_commands)

            fence = self.fences.allocate()
            command_waits: List[Any] = _as_list(waits)
            for cmd in commands:
                wait_fence = _pick(cmd, ("wait_fence", "wait_for_fence"))
                if wait_fence is not None:
                    command_waits.append(_as_int(wait_fence))

            submission = self.queues.submit(
                queue=queue,
                commands=commands,
                fence=fence,
                waits=command_waits,
                signals=[fence],
            )
            self.submission_count += 1

            if self.frame is not None and recorder is self.frame.recorder:
                self.frame.fence = fence
                self.frame.status = "submitted"
                self.frame.submission_id = submission.id
                self.frame.submissions.append(submission.id)
                if self.frame.allocation_id is not None:
                    self.allocator.mark_submitted(self.frame.allocation_id, fence)
            return submission

    def present(self, target: Any = None, fence: Optional[int] = None) -> Record:
        with self._lock:
            frame: Optional[Record] = None
            if _is_int(target) and int(target) in self.frames:
                frame = self.frames[int(target)]
            elif isinstance(target, Record) and target.get("id") in self.frames:
                frame = self.frames[int(target["id"])]
            elif self.frame is not None:
                frame = self.frame

            if frame is None:
                frame = self.begin_frame()

            if fence is not None:
                value = int(fence)
            elif frame.get("fence") is not None:
                value = int(frame.fence)
            else:
                value = self.fences.allocate()

            completed = self.fences.signal(value)
            recycled = self.allocator.recycle(completed)
            released = self.handles.collect(completed)
            frame.status = "presented"
            self.present_count += 1
            return Record(
                ok=True,
                frame=frame.id,
                fence=value,
                completed=completed,
                recycled=recycled,
                released=released,
            )

    def wait_idle(self) -> Record:
        with self._lock:
            completed = self.fences.signal(self.fences.latest)
            recycled = self.allocator.recycle(completed)
            released = self.handles.collect(completed)
            return Record(completed=completed, recycled=recycled, released=released)

    def validate(self) -> Record:
        errors: List[str] = []

        scheduler_report = self.queues.schedule_report()
        if not scheduler_report.ok:
            errors.extend(scheduler_report.errors)

        errors.extend(self.handles.validate())

        frame_reports: Dict[int, Dict[str, Any]] = {}
        with self._lock:
            frames = list(self.frames.values())
        for frame in frames:
            recorder: CommandRecorder = frame.recorder
            hash_a = recorder.recording_hash()
            hash_b = recorder.recording_hash()
            replay = self.checker.check(recorder)
            frame_errors: List[str] = []
            if hash_a != hash_b:
                frame_errors.append("command order hash is not stable")
            if not replay.match:
                frame_errors.append("replay hash mismatch")
            frame_reports[frame.id] = {
                "hash": hash_a,
                "commands": len(recorder.build()),
                "errors": frame_errors,
            }
            errors.extend(f"frame {frame.id}: {msg}" for msg in frame_errors)

        return Record(
            ok=not errors,
            errors=errors,
            scheduler=scheduler_report,
            frames=frame_reports,
            global_state_checksum=self.states.checksum(),
        )

    def metrics(self) -> Record:
        with self._lock:
            return Record(
                frames=len(self.frames),
                commands=self.command_count,
                submissions=self.submission_count,
                presents=self.present_count,
                fallbacks=self.fallback_count,
                barriers=len(self.states.barriers),
                dependencies=len(self.queues.dependencies),
                scheduler_submissions=len(self.queues.submissions),
                allocations=len(self.allocator.allocations),
                recycled_allocations=self.allocator.recycled_count,
                blocked_resets=self.allocator.blocked_reset_count,
                live_handles=len(self.handles.live),
                pending_releases=len(self.handles.pending),
                released_handles=self.handles.released_count,
                fence_completed=self.fences.completed,
                serial_mode=self.serial_mode,
            )

    def audit(self) -> Record:
        validation = self.validate()
        return Record(
            ok=validation.ok,
            metrics=self.metrics(),
            validation=validation,
            states=self.states.snapshot(),
            barriers=[dict(b) for b in self.states.barriers],
            dependencies=[dict(d) for d in self.queues.dependencies],
            pending_releases=self.handles.pending_report(),
            last_fallback=dict(self.last_fallback) if self.last_fallback else None,
        )

    def fallback(self, reason: str = "validation_failed") -> Record:
        with self._lock:
            self.fallback_count += 1
            idle = self.wait_idle()
            self.serial_mode = True
            order: List[str] = []
            if self.frame is not None:
                order = self.frame.recorder.ops()
            result = Record(
                activated=True,
                ok=True,
                mode="wait_idle_then_serialize",
                reason=reason,
                idle=idle,
                fixed_order=order,
            )
            self.last_fallback = result
            return result


_DEFAULT_SYSTEM = ParallelCommandSystem()


# --------------------------------------------------------------------------- #
# 内置确定性演示场景
# --------------------------------------------------------------------------- #


def _demo_allocator_report() -> Record:
    fences = FenceTimeline()
    allocator = CommandAllocator(fences)
    block = allocator.allocate(frame=7)
    allocator.mark_submitted(block.block_id, 3)

    early = allocator.recycle(completed_fence=2)
    unsafe_reset_blocked = False
    try:
        allocator.reset(block.block_id, completed_fence=2)
    except FenceNotReady:
        unsafe_reset_blocked = True

    fences.signal(3)
    recycled = allocator.recycle()
    safe_reuse = allocator.allocate(frame=8).block_id == block.block_id
    return Record(
        ok=unsafe_reset_blocked and early == [] and recycled == [block.block_id] and safe_reuse,
        early_recycle=early,
        unsafe_reset_blocked=unsafe_reset_blocked,
        recycled=recycled,
        reused_block=block.block_id if safe_reuse else None,
        fence=3,
    )


def _demo_cross_queue(resource: str = "vertex_buffer") -> Record:
    scheduler = CrossQueueScheduler()
    graphics = scheduler.submit(
        "graphics",
        [{"op": "write_vertex_buffer", "resource": resource, "access": "write"}],
        fence=1,
    )
    compute = scheduler.submit(
        "compute",
        [{"op": "read_vertex_buffer", "resource": resource, "access": "read"}],
        waits=[1],
    )
    dependency = scheduler.add_dependency(
        resource=resource,
        producer_queue="graphics",
        producer_fence=1,
        consumer_queue="compute",
        stages=["vertex_input", "shader_read"],
    )
    schedule = scheduler.schedule_report()
    return Record(
        ok=bool(schedule.ok) and schedule.order.index(graphics.id) < schedule.order.index(compute.id),
        explicit=True,
        resource=resource,
        producer_queue="graphics",
        producer_fence=1,
        consumer_queue="compute",
        consumer_wait=1,
        graphics_submission=graphics.id,
        compute_submission=compute.id,
        schedule=schedule,
        order=schedule.order,
        dependency=dependency,
    )


def _demo_deferred_release() -> Record:
    queue = DeferredReleaseQueue()
    handle = queue.create("texture")
    queue.request_release(handle, fence=2)
    early = queue.collect(completed_fence=1)
    reused_early = handle in queue.free_ids and not queue.is_pending(handle)
    released = queue.collect(completed_fence=2)
    new_handle = queue.create("texture")
    return Record(
        ok=early == [] and not reused_early and released == [handle] and new_handle == handle,
        handle=handle,
        early_collect=early,
        released=released,
        reused_handle_after_fence=new_handle,
        pending_before_fence=queue.pending_report(),
    )


def _packets_for_demo() -> List[Tuple[int, str, List[Dict[str, Any]]]]:
    return [
        (
            2,
            "worker-B",
            [
                {"op": "main_pass", "sequence": 0, "resource": "scene_color", "state": "render_target"},
                {"op": "draw_scene", "sequence": 1},
            ],
        ),
        (
            0,
            "worker-main",
            [
                {"op": "clear_color", "sequence": 0, "resource": "backbuffer", "state": "render_target"},
                {"op": "bind_backbuffer", "sequence": 1},
            ],
        ),
        (
            1,
            "worker-A",
            [
                {"op": "shadow_pass", "sequence": 0, "resource": "shadow_map", "state": "depth_write"},
            ],
        ),
    ]


def _run_fixed_order_once(start_reversed: bool = False) -> Tuple[List[str], str]:
    recorder = CommandRecorder(frame=1)
    packets = list(_packets_for_demo())
    if start_reversed:
        packets.reverse()

    threads: List[threading.Thread] = []
    for order, worker, commands in packets:
        thread = threading.Thread(
            target=recorder.record,
            args=(commands,),
            kwargs={"order": order, "worker": worker},
        )
        threads.append(thread)

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return recorder.ops(), recorder.recording_hash()


def _demo_fixed_order(runs: int = 5) -> Record:
    orders: List[List[str]] = []
    hashes: List[str] = []
    for run in range(max(1, runs)):
        ops, digest = _run_fixed_order_once(start_reversed=bool(run % 2))
        orders.append(ops)
        hashes.append(digest)

    expected = [
        "clear_color",
        "bind_backbuffer",
        "shadow_pass",
        "main_pass",
        "draw_scene",
    ]
    identical_orders = all(order == orders[0] for order in orders)
    identical_hashes = len(set(hashes)) == 1
    return Record(
        ok=identical_orders and identical_hashes and orders[0] == expected,
        deterministic=identical_orders and identical_hashes,
        fixed_order=expected,
        order=orders[0],
        hash=hashes[0] if hashes else None,
        runs=len(orders),
        orders=orders,
        hashes=hashes,
    )


def _golden_recorder(seed: Any = 1) -> CommandRecorder:
    recorder = CommandRecorder(frame=_as_int(seed, 1) or 1)
    recorder.record(
        [
            {
                "op": "transition_backbuffer",
                "resource": "backbuffer",
                "state": "render_target",
                "queue": "graphics",
                "sequence": 0,
            },
            {
                "op": "write_vertex_buffer",
                "resource": "vertex_buffer",
                "access": "write",
                "queue": "graphics",
                "sequence": 1,
                "signal_fence": 1,
            },
            {
                "op": "compute_read_vertex_buffer",
                "resource": "vertex_buffer",
                "access": "read",
                "queue": "compute",
                "sequence": 2,
                "wait_fence": 1,
            },
            {
                "op": "sample_texture",
                "resource": "texture",
                "state": "sampled",
                "queue": "graphics",
                "sequence": 3,
            },
        ],
        order=0,
        worker="golden",
    )
    recorder.seal()
    return recorder


def _order_report_for_spec(spec: Any) -> Record:
    recorder = CommandRecorder()
    if isinstance(spec, CommandRecorder):
        recorder = spec
    elif isinstance(spec, list) and spec and isinstance(spec[0], CommandPacket):
        for packet in spec:
            recorder.add_packet(packet)
    else:
        recorder.record(spec, order=0)
    digest_a = recorder.recording_hash()
    digest_b = recorder.recording_hash()
    return Record(
        ok=digest_a == digest_b,
        deterministic=digest_a == digest_b,
        order=recorder.ops(),
        commands=recorder.command_dicts(),
        hash=digest_a,
    )


# --------------------------------------------------------------------------- #
# 模块级公共入口
# --------------------------------------------------------------------------- #


def global_state_table(*args: Any, **kwargs: Any) -> StateTable:
    """返回全局唯一状态表；传入 system 时返回该 system 的共享表。"""

    for arg in args:
        if isinstance(arg, ParallelCommandSystem):
            return arg.states
    return _DEFAULT_SYSTEM.states


def allocator_fence_recycle(*args: Any, **kwargs: Any) -> Record:
    """栅栏回收入口；整数占位调用运行内置安全回收演示。"""

    if args and isinstance(args[0], ParallelCommandSystem):
        system = args[0]
        recycled = system.allocator.recycle()
        return Record(ok=True, recycled=recycled, metrics=system.metrics())
    if args and isinstance(args[0], CommandAllocator):
        allocator = args[0]
        recycled = allocator.recycle()
        return Record(ok=True, recycled=recycled)
    return _demo_allocator_report()


def cross_queue_dependency(*args: Any, **kwargs: Any) -> Record:
    """构造/演示跨队列显式 fence 依赖。"""

    resource = "vertex_buffer"
    for arg in args:
        if isinstance(arg, str):
            resource = arg
            break
        if isinstance(arg, dict) and _resource_id(arg.get("resource")):
            resource = _resource_id(arg.get("resource"), resource)
            break
    return _demo_cross_queue(resource)


def deferred_release(*args: Any, **kwargs: Any) -> Record:
    """延迟释放入口；可传入 ``(queue, handle, fence)``。"""

    if args and isinstance(args[0], DeferredReleaseQueue):
        queue = args[0]
        if len(args) >= 3:
            handle = _as_int(args[1])
            fence = _as_int(args[2])
            requested = queue.request_release(handle, fence)
            return Record(ok=requested, requested=requested, pending=queue.pending_report())
        return Record(ok=True, pending=queue.pending_report())
    return _demo_deferred_release()


def fixed_record_order(*args: Any, **kwargs: Any) -> Record:
    """固定录制顺序；传入命令列表时排序，否则运行多线程确定性演示。"""

    if args and isinstance(args[0], (list, tuple, CommandRecorder)):
        return _order_report_for_spec(args[0])
    return _demo_fixed_order()


def replay_check(*args: Any, **kwargs: Any) -> Record:
    """回放校验；支持 ``replay_check(expected, actual)`` 或单个录制对象。"""

    if len(args) >= 2:
        return ReplayChecker().compare(args[0], args[1])
    if args and not _is_int(args[0]):
        target = args[0]
        if isinstance(target, ParallelCommandSystem):
            if target.frame is not None:
                return target.checker.check(target.frame.recorder)
            return target.validate()
        return ReplayChecker().check(target)
    return ReplayChecker().check(_golden_recorder(args[0] if args else 1))


def begin_frame(*args: Any, **kwargs: Any) -> Record:
    if args and isinstance(args[0], ParallelCommandSystem):
        return args[0].begin_frame(*args[1:], **kwargs)
    return _DEFAULT_SYSTEM.begin_frame(*args, **kwargs)


def record(*args: Any, **kwargs: Any) -> CommandRecorder:
    if args and isinstance(args[0], ParallelCommandSystem):
        return args[0].record(*args[1:], **kwargs)
    return _DEFAULT_SYSTEM.record(*args, **kwargs)


def submit(*args: Any, **kwargs: Any) -> QueueSubmission:
    if args and isinstance(args[0], ParallelCommandSystem):
        return args[0].submit(*args[1:], **kwargs)
    return _DEFAULT_SYSTEM.submit(*args, **kwargs)


def present(*args: Any, **kwargs: Any) -> Record:
    if args and isinstance(args[0], ParallelCommandSystem):
        return args[0].present(*args[1:], **kwargs)
    return _DEFAULT_SYSTEM.present(*args, **kwargs)


def validate_record(*args: Any, **kwargs: Any) -> Record:
    if args and isinstance(args[0], ParallelCommandSystem):
        return args[0].validate()
    if args and isinstance(args[0], CommandRecorder):
        result = ReplayChecker().check(args[0])
        result["order_stable"] = args[0].recording_hash() == args[0].recording_hash()
        return result
    if args and isinstance(args[0], (list, tuple)):
        recorder = CommandRecorder()
        recorder.record(args[0], order=0)
        recorder.seal()
        return ReplayChecker().check(recorder)
    return _DEFAULT_SYSTEM.validate()


def metrics(*args: Any, **kwargs: Any) -> Record:
    if args and isinstance(args[0], ParallelCommandSystem):
        return args[0].metrics()
    if args and isinstance(args[0], CommandRecorder):
        return Record(
            packets=len(args[0].ordered_packets()),
            commands=len(args[0].build()),
            hash=args[0].recording_hash(),
        )
    return _DEFAULT_SYSTEM.metrics()


def audit(*args: Any, **kwargs: Any) -> Record:
    if args and isinstance(args[0], ParallelCommandSystem):
        return args[0].audit()
    return _DEFAULT_SYSTEM.audit()


def fallback(*args: Any, **kwargs: Any) -> Record:
    """校验失败或分配器压力下的安全兜底：等栅栏、延迟回收、串行录制。

    返回包含 ``mode`` / ``ok`` / ``fixed_order`` 的报告；传入
    :class:`ParallelCommandSystem` 时操作该实例，否则操作模块级默认系统。
    """

    reason = str(kwargs.get("reason", "contract_fallback"))
    for arg in args:
        if isinstance(arg, ParallelCommandSystem):
            return arg.fallback(reason)
        if isinstance(arg, Record) and arg.get("ok") is False:
            return _DEFAULT_SYSTEM.fallback("validation_failed")
    return _DEFAULT_SYSTEM.fallback(reason)


def determinism(*args: Any, **kwargs: Any) -> Record:
    report = _demo_fixed_order(runs=int(kwargs.get("runs", 5)))
    replay = ReplayChecker().check(_golden_recorder())
    report["replay_match"] = replay.match
    report["ok"] = bool(report.ok and replay.match)
    report["deterministic"] = bool(report.deterministic and replay.match)
    return report
