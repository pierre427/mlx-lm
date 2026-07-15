# Copyright © 2026 Apple Inc.

"""Synthetic MLX probe for a per-radix-path recurrent-state checkpoint cap.

This is deliberately a policy probe, not a production cache implementation.
mlx-lm's request cache is an entry trie rather than SGLang's GPU radix cache,
so the useful first question is whether a pruning rule can bound redundant
device recurrent state without invalidating branch points or recovery state.

The probe models a radix tree whose nodes may own:

* an MLX recurrent-state checkpoint (eligible for device pruning),
* Full KV state (never touched),
* a host backup (never touched), or
* a lock/fork/leaf role that protects its recurrent checkpoint.

For every over-cap root-to-leaf path, the shallowest unprotected recurrent
checkpoint is removed. The deepest checkpoint on every path is protected as
the tail checkpoint. A cap is therefore soft when protected checkpoints alone
exceed it; correctness wins over the memory target.
"""

import argparse
import json
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx


@dataclass(eq=False)
class RadixNode:
    name: str
    recurrent_state: Optional[mx.array] = None
    full_kv: Optional[mx.array] = None
    host_backup: Optional[bytes] = None
    locked: bool = False
    parent: Optional["RadixNode"] = field(default=None, repr=False)
    children: list["RadixNode"] = field(default_factory=list, repr=False)

    def add(self, child: "RadixNode") -> "RadixNode":
        child.parent = self
        self.children.append(child)
        return child

    @property
    def is_fork(self) -> bool:
        return len(self.children) > 1

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def recurrent_nbytes(self) -> int:
        return 0 if self.recurrent_state is None else self.recurrent_state.nbytes


def _walk(root: RadixNode) -> list[RadixNode]:
    out = []
    stack = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(reversed(node.children))
    return out


def _path(leaf: RadixNode) -> list[RadixNode]:
    out = []
    node = leaf
    while node is not None:
        out.append(node)
        node = node.parent
    return list(reversed(out))


def _checkpoint_nodes(path: list[RadixNode]) -> list[RadixNode]:
    return [node for node in path if node.recurrent_state is not None]


def cap_recurrent_checkpoints(
    root: RadixNode, cap: Optional[int]
) -> dict[str, object]:
    """Prune recurrent checkpoints until every safely reducible path fits.

    Forks, locks, leaves, Full-KV nodes, and each path's deepest recurrent
    checkpoint are protected. Host backups are orthogonal: pruning device state
    must not remove them. If protected checkpoints exceed ``cap``, the path is
    reported in ``soft_overflow_paths`` rather than pruned unsafely.
    """
    if cap is None:
        return {"pruned": [], "soft_overflow_paths": []}
    if cap < 1:
        raise ValueError("cap must be at least 1 or None")

    nodes = _walk(root)
    leaves = [node for node in nodes if node.is_leaf]
    tail_checkpoints = {
        checkpoints[-1]
        for leaf in leaves
        if (checkpoints := _checkpoint_nodes(_path(leaf)))
    }

    def protected(node: RadixNode) -> bool:
        return (
            node in tail_checkpoints
            or node.is_fork
            or node.is_leaf
            or node.locked
            or node.full_kv is not None
        )

    pruned = []
    while True:
        candidates = []
        violating = []
        for leaf in leaves:
            path = _path(leaf)
            checkpoints = _checkpoint_nodes(path)
            if len(checkpoints) <= cap:
                continue
            violating.append(leaf.name)
            for depth, node in enumerate(path):
                if node.recurrent_state is not None and not protected(node):
                    candidates.append((depth, node.name, node))
        if not violating or not candidates:
            break
        _, _, victim = min(candidates, key=lambda item: (item[0], item[1]))
        victim.recurrent_state = None
        pruned.append(victim.name)

    soft_overflow = [
        leaf.name
        for leaf in leaves
        if len(_checkpoint_nodes(_path(leaf))) > cap
    ]
    return {"pruned": pruned, "soft_overflow_paths": soft_overflow}


def _state(kib: int = 16) -> mx.array:
    return mx.zeros((kib * 256,), dtype=mx.float32)


def build_probe_tree() -> RadixNode:
    root = RadixNode("root")
    shared_0 = root.add(RadixNode("shared_0", recurrent_state=_state()))
    shared_1 = shared_0.add(RadixNode("shared_1", recurrent_state=_state()))
    fork = shared_1.add(RadixNode("fork", recurrent_state=_state()))

    left_0 = fork.add(RadixNode("left_0", recurrent_state=_state()))
    left_1 = left_0.add(
        RadixNode("left_locked", recurrent_state=_state(), locked=True)
    )
    left_1.add(
        RadixNode(
            "left_leaf",
            recurrent_state=_state(),
            full_kv=mx.ones((2048,), dtype=mx.float32),
        )
    )

    right_0 = fork.add(
        RadixNode(
            "right_host_backed",
            recurrent_state=_state(),
            host_backup=b"host-recovery-checkpoint",
        )
    )
    right_1 = right_0.add(RadixNode("right_1", recurrent_state=_state()))
    right_1.add(RadixNode("right_leaf", recurrent_state=_state()))
    return root


def run_probe(cap: Optional[int]) -> dict[str, object]:
    root = build_probe_tree()
    nodes = _walk(root)
    mx.eval(
        *[node.recurrent_state for node in nodes if node.recurrent_state is not None],
        *[node.full_kv for node in nodes if node.full_kv is not None],
    )
    before = sum(node.recurrent_nbytes for node in nodes)
    full_kv_before = {
        node.name: mx.array(node.full_kv)
        for node in nodes
        if node.full_kv is not None
    }
    host_before = {
        node.name: node.host_backup for node in nodes if node.host_backup is not None
    }

    result = cap_recurrent_checkpoints(root, cap)
    after = sum(node.recurrent_nbytes for node in nodes)

    protected_names = {"fork", "left_locked", "left_leaf", "right_leaf"}
    assert all(
        next(node for node in nodes if node.name == name).recurrent_state is not None
        for name in protected_names
    )
    assert all(
        mx.array_equal(next(n for n in nodes if n.name == name).full_kv, value)
        for name, value in full_kv_before.items()
    )
    assert all(
        next(n for n in nodes if n.name == name).host_backup == value
        for name, value in host_before.items()
    )
    if cap is None:
        assert result["pruned"] == [] and after == before
    else:
        assert after <= before
        for leaf in (node for node in nodes if node.is_leaf):
            count = len(_checkpoint_nodes(_path(leaf)))
            if leaf.name not in result["soft_overflow_paths"]:
                assert count <= cap

    return {
        "cap": cap,
        "recurrent_device_bytes_before": before,
        "recurrent_device_bytes_after": after,
        "reduction_percent": round(100.0 * (before - after) / before, 1),
        "full_kv_bytes_preserved": sum(v.nbytes for v in full_kv_before.values()),
        "host_backups_preserved": len(host_before),
        **result,
        "path_checkpoint_counts": {
            leaf.name: len(_checkpoint_nodes(_path(leaf)))
            for leaf in nodes
            if leaf.is_leaf
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=3)
    parser.add_argument("--unlimited", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_probe(None if args.unlimited else args.cap), indent=2))
