# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic planning primitives for cooperative checkpoint loading."""

from __future__ import annotations

import json
import time
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from heapq import merge as merge_sorted
from types import MappingProxyType
from typing import TypeAlias

WIRE_VERSION = 1
EXECUTION_PLAN_WIRE_VERSION = 1
COORDINATION_WORLD_MAPPING_VERSION = "leader-rank-node-membership-v2"
_UINT64_MAX = (1 << 64) - 1
_BYTE_RANGE_KEYS = frozenset({"length", "offset", "source_rank"})
_FQN_DEMAND_KEYS = frozenset({"fqn", "ranges"})

NodeId: TypeAlias = str | int
SourceConsumerBytesByNode: TypeAlias = Mapping[NodeId, Mapping[int, int]]


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_u64(name: str, value: object, *, minimum: int = 0) -> int:
    parsed = _require_int(name, value, minimum=minimum)
    if parsed > _UINT64_MAX:
        raise ValueError(f"{name} must fit in an unsigned 64-bit integer")
    return parsed


def _require_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_fqn(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _require_node_id(name: str, value: object) -> NodeId:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer or non-empty string")
    if isinstance(value, int):
        return _require_int(name, value)
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{name} must be a non-negative integer or non-empty string")


def _node_id_sort_key(node_id: NodeId) -> tuple[int, int, str]:
    if isinstance(node_id, int):
        return (0, node_id, "")
    return (1, 0, node_id)


def _canonical_node_ids(node_ids: Iterable[NodeId]) -> tuple[NodeId, ...]:
    validated = tuple(_require_node_id("node_id", node_id) for node_id in node_ids)
    result = tuple(sorted(validated, key=_node_id_sort_key))
    if not result:
        raise ValueError("at least one node is required")
    if len(result) != len(set(result)):
        raise ValueError("node IDs must be unique")
    return result


def _require_mapping(value: object, name: str = "payload") -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _require_sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value


def _require_keys(
    payload: Mapping[str, object], expected: set[str] | frozenset[str]
) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"wire payload keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _checked_end(offset: int, length: int) -> int:
    end = offset + length
    if end > _UINT64_MAX:
        raise ValueError(f"byte range overflows u64: offset={offset}, length={length}")
    return end


def _byte_range_fields_from_dict(value: object) -> tuple[int, int, int]:
    if (
        type(value) is dict
        and len(value) == len(_BYTE_RANGE_KEYS)
        and all(type(key) is str for key in value)
        and "source_rank" in value
        and "offset" in value
        and "length" in value
    ):
        payload = value
    else:
        payload = _require_mapping(value)
        _require_keys(payload, _BYTE_RANGE_KEYS)
    source_rank = _require_int("source_rank", payload["source_rank"])
    offset = _require_u64("offset", payload["offset"])
    length = _require_u64("length", payload["length"], minimum=1)
    _checked_end(offset, length)
    return source_rank, offset, length


@dataclass(frozen=True)
class NodeMembership:
    """Stable node identity and its explicit, possibly non-contiguous ranks."""

    node_id: NodeId
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        node_id = _require_node_id("node_id", self.node_id)
        ranks = tuple(sorted(_require_int("rank", rank) for rank in self.ranks))
        if not ranks:
            raise ValueError("a node must contain at least one rank")
        if len(ranks) != len(set(ranks)):
            raise ValueError("a rank may occur only once within a node")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "ranks", ranks)

    @property
    def leader_rank(self) -> int:
        return self.ranks[0]

    def to_dict(self) -> dict[str, object]:
        return {"node_id": self.node_id, "ranks": list(self.ranks)}

    @classmethod
    def from_dict(cls, value: object) -> NodeMembership:
        payload = _require_mapping(value)
        _require_keys(payload, {"node_id", "ranks"})
        return cls(
            node_id=_require_node_id("node_id", payload["node_id"]),
            ranks=tuple(
                _require_int("rank", rank)
                for rank in _require_sequence(payload["ranks"], "ranks")
            ),
        )


def _partition_nodes(
    node_ids: tuple[NodeId, ...], world_count: int
) -> tuple[tuple[NodeId, ...], ...]:
    quotient, remainder = divmod(len(node_ids), world_count)
    worlds: list[tuple[NodeId, ...]] = []
    start = 0
    for world_id in range(world_count):
        size = quotient + (1 if world_id < remainder else 0)
        worlds.append(node_ids[start : start + size])
        start += size
    return tuple(worlds)


@dataclass(frozen=True)
class CoordinationWorld:
    """One deterministic coordination group over explicit node identities."""

    global_node_ids: tuple[NodeId, ...]
    node_ids: tuple[NodeId, ...]
    local_node_id: NodeId
    world_count: int
    world_id: int
    rendezvous_id: str

    def __post_init__(self) -> None:
        global_node_ids = _canonical_node_ids(self.global_node_ids)
        node_ids = _canonical_node_ids(self.node_ids)
        local_node_id = _require_node_id("local_node_id", self.local_node_id)
        world_count = _require_int("world_count", self.world_count, minimum=1)
        world_id = _require_int("world_id", self.world_id)
        _require_str("rendezvous_id", self.rendezvous_id)
        if world_count > len(global_node_ids):
            raise ValueError("world_count cannot exceed the number of nodes")
        if world_id >= world_count:
            raise ValueError("world_id must be smaller than world_count")
        if local_node_id not in node_ids:
            raise ValueError("local_node_id is not in this coordination world")
        expected = _partition_nodes(global_node_ids, world_count)[world_id]
        if node_ids != expected:
            raise ValueError("coordination-world nodes do not match canonical mapping")
        object.__setattr__(self, "global_node_ids", global_node_ids)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "local_node_id", local_node_id)

    @property
    def global_num_nodes(self) -> int:
        return len(self.global_node_ids)

    @property
    def global_node_index(self) -> int:
        return self.global_node_ids.index(self.local_node_id)

    @property
    def world_num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def world_node_index(self) -> int:
        return self.node_ids.index(self.local_node_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "global_node_ids": list(self.global_node_ids),
            "local_node_id": self.local_node_id,
            "node_ids": list(self.node_ids),
            "rendezvous_id": self.rendezvous_id,
            "world_count": self.world_count,
            "world_id": self.world_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> CoordinationWorld:
        payload = _require_mapping(value)
        _require_keys(
            payload,
            {
                "global_node_ids",
                "local_node_id",
                "node_ids",
                "rendezvous_id",
                "world_count",
                "world_id",
            },
        )
        return cls(
            global_node_ids=tuple(
                _require_node_id("node_id", node_id)
                for node_id in _require_sequence(
                    payload["global_node_ids"], "global_node_ids"
                )
            ),
            node_ids=tuple(
                _require_node_id("node_id", node_id)
                for node_id in _require_sequence(payload["node_ids"], "node_ids")
            ),
            local_node_id=_require_node_id("local_node_id", payload["local_node_id"]),
            world_count=_require_int("world_count", payload["world_count"], minimum=1),
            world_id=_require_int("world_id", payload["world_id"]),
            rendezvous_id=_require_str("rendezvous_id", payload["rendezvous_id"]),
        )


@dataclass(frozen=True)
class RankTopology:
    """The current rank's view of explicit global rank-to-node membership."""

    global_rank: int
    nodes: tuple[NodeMembership, ...]
    coordination_world_count: int
    job_id: str

    def __post_init__(self) -> None:
        global_rank = _require_int("global_rank", self.global_rank)
        nodes = tuple(
            sorted(self.nodes, key=lambda node: _node_id_sort_key(node.node_id))
        )
        if not nodes:
            raise ValueError("topology must contain at least one node")
        node_ids = tuple(node.node_id for node in nodes)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node IDs must be unique")
        ranks = tuple(rank for node in nodes for rank in node.ranks)
        if len(ranks) != len(set(ranks)):
            raise ValueError("every global rank must appear exactly once")
        if global_rank not in ranks:
            raise ValueError("global_rank is not present in node membership")
        world_count = _require_int(
            "coordination_world_count", self.coordination_world_count, minimum=1
        )
        if world_count > len(nodes):
            raise ValueError("coordination_world_count cannot exceed node count")
        _require_str("job_id", self.job_id)
        object.__setattr__(self, "global_rank", global_rank)
        object.__setattr__(self, "nodes", nodes)

    @property
    def world_size(self) -> int:
        return sum(len(node.ranks) for node in self.nodes)

    @property
    def global_num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def node(self) -> NodeMembership:
        return next(node for node in self.nodes if self.global_rank in node.ranks)

    @property
    def node_id(self) -> NodeId:
        return self.node.node_id

    @property
    def node_ranks(self) -> tuple[int, ...]:
        return self.node.ranks

    @property
    def node_leader_rank(self) -> int:
        return self.node.leader_rank

    @property
    def local_rank(self) -> int:
        return self.node.ranks.index(self.global_rank)

    @property
    def ranks_per_node(self) -> int:
        return len(self.node.ranks)

    @property
    def global_node_index(self) -> int:
        return self.nodes.index(self.node)

    @property
    def world_id(self) -> int:
        return self.coordination_world.world_id

    @property
    def world_node_index(self) -> int:
        return self.coordination_world.world_node_index

    @property
    def world_num_nodes(self) -> int:
        return self.coordination_world.world_num_nodes

    @property
    def rendezvous_id(self) -> str:
        return self.coordination_world.rendezvous_id

    @property
    def coordination_world(self) -> CoordinationWorld:
        global_node_ids = tuple(node.node_id for node in self.nodes)
        worlds = _partition_nodes(global_node_ids, self.coordination_world_count)
        world_id = next(
            index for index, node_ids in enumerate(worlds) if self.node_id in node_ids
        )
        return CoordinationWorld(
            global_node_ids=global_node_ids,
            node_ids=worlds[world_id],
            local_node_id=self.node_id,
            world_count=self.coordination_world_count,
            world_id=world_id,
            rendezvous_id=(
                f"{self.job_id}__coord_{COORDINATION_WORLD_MAPPING_VERSION}"
                f"__r{self.coordination_world_count}__w{world_id}"
            ),
        )

    @property
    def is_node_leader(self) -> bool:
        return self.global_rank == self.node_leader_rank

    @property
    def is_world_leader(self) -> bool:
        return self.is_node_leader and self.world_node_index == 0

    @property
    def num_local_followers(self) -> int:
        return len(self.node_ranks) - 1

    def to_dict(self) -> dict[str, object]:
        return {
            "coordination_world_count": self.coordination_world_count,
            "global_rank": self.global_rank,
            "job_id": self.job_id,
            "nodes": [node.to_dict() for node in self.nodes],
        }

    @classmethod
    def from_dict(cls, value: object) -> RankTopology:
        payload = _require_mapping(value)
        _require_keys(
            payload,
            {"coordination_world_count", "global_rank", "job_id", "nodes"},
        )
        return cls(
            global_rank=_require_int("global_rank", payload["global_rank"]),
            nodes=tuple(
                NodeMembership.from_dict(node)
                for node in _require_sequence(payload["nodes"], "nodes")
            ),
            coordination_world_count=_require_int(
                "coordination_world_count",
                payload["coordination_world_count"],
                minimum=1,
            ),
            job_id=_require_str("job_id", payload["job_id"]),
        )


@dataclass(frozen=True, order=True)
class ByteRange:
    source_rank: int
    offset: int
    length: int

    def __post_init__(self) -> None:
        _require_int("source_rank", self.source_rank)
        _require_u64("offset", self.offset)
        _require_u64("length", self.length, minimum=1)
        _checked_end(self.offset, self.length)

    @property
    def end(self) -> int:
        return self.offset + self.length

    def to_dict(self) -> dict[str, object]:
        return {
            "length": self.length,
            "offset": self.offset,
            "source_rank": self.source_rank,
        }

    @classmethod
    def from_dict(cls, value: object) -> ByteRange:
        source_rank, offset, length = _byte_range_fields_from_dict(value)
        return cls(source_rank=source_rank, offset=offset, length=length)


@dataclass(frozen=True)
class FqnDemand:
    fqn: str
    ranges: tuple[ByteRange, ...]

    def __post_init__(self) -> None:
        _require_fqn("fqn", self.fqn)
        canonical = union_byte_ranges(tuple(self.ranges))
        if not canonical:
            raise ValueError("an FQN demand must contain at least one byte range")
        object.__setattr__(self, "ranges", canonical)

    @property
    def total_bytes(self) -> int:
        return sum(byte_range.length for byte_range in self.ranges)

    def to_dict(self) -> dict[str, object]:
        return {
            "fqn": self.fqn,
            "ranges": [byte_range.to_dict() for byte_range in self.ranges],
        }

    @classmethod
    def from_dict(cls, value: object) -> FqnDemand:
        payload = _require_mapping(value)
        _require_keys(payload, _FQN_DEMAND_KEYS)
        raw_ranges = _require_sequence(payload["ranges"], "ranges")
        return cls(
            fqn=_require_fqn("fqn", payload["fqn"]),
            ranges=tuple(ByteRange.from_dict(item) for item in raw_ranges),
        )


@dataclass(frozen=True, slots=True)
class FqnDemandWireMergeResult:
    demands: tuple[FqnDemand, ...]
    input_demand_count: int
    input_range_count: int
    decode_ns: int
    union_ns: int
    finalize_ns: int


@dataclass(frozen=True)
class FileAssignment:
    node_ids: tuple[NodeId, ...]
    node_source_ranks: tuple[tuple[int, ...], ...]
    node_bytes: tuple[int, ...]
    _source_rank_to_node: Mapping[int, NodeId] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _node_to_index: Mapping[NodeId, int] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not (
            len(self.node_ids) == len(self.node_source_ranks) == len(self.node_bytes)
        ):
            raise ValueError("assignment node arrays must have matching lengths")
        if not self.node_ids:
            raise ValueError("assignment must contain at least one node")
        entries = sorted(
            zip(self.node_ids, self.node_source_ranks, self.node_bytes),
            key=lambda item: _node_id_sort_key(_require_node_id("node_id", item[0])),
        )
        node_ids = tuple(_require_node_id("node_id", item[0]) for item in entries)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node IDs must be unique")
        node_source_ranks = tuple(
            tuple(sorted(_require_int("source_rank", rank) for rank in item[1]))
            for item in entries
        )
        node_bytes = tuple(_require_u64("node_bytes", item[2]) for item in entries)
        flattened = [rank for ranks in node_source_ranks for rank in ranks]
        if len(flattened) != len(set(flattened)):
            raise ValueError("a source rank may be assigned to only one node")
        source_rank_to_node = {
            source_rank: node_id
            for node_id, source_ranks in zip(node_ids, node_source_ranks)
            for source_rank in source_ranks
        }
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "node_source_ranks", node_source_ranks)
        object.__setattr__(self, "node_bytes", node_bytes)
        object.__setattr__(
            self,
            "_source_rank_to_node",
            MappingProxyType(source_rank_to_node),
        )
        object.__setattr__(
            self,
            "_node_to_index",
            MappingProxyType(
                {node_id: index for index, node_id in enumerate(node_ids)}
            ),
        )

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def source_rank_to_node(self) -> Mapping[int, NodeId]:
        return self._source_rank_to_node

    def owner_for(self, source_rank: int) -> NodeId:
        try:
            return self.source_rank_to_node[source_rank]
        except KeyError as error:
            raise ValueError(
                f"source rank {source_rank} has no assigned owner"
            ) from error

    def node_index_for(self, node_id: NodeId) -> int:
        try:
            return self._node_to_index[node_id]
        except KeyError as error:
            raise ValueError(f"node {node_id!r} is not in the assignment") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "node_bytes": list(self.node_bytes),
            "node_ids": list(self.node_ids),
            "node_source_ranks": [list(ranks) for ranks in self.node_source_ranks],
        }

    @classmethod
    def from_dict(cls, value: object) -> FileAssignment:
        payload = _require_mapping(value)
        _require_keys(payload, {"node_bytes", "node_ids", "node_source_ranks"})
        return cls(
            node_ids=tuple(
                _require_node_id("node_id", node_id)
                for node_id in _require_sequence(payload["node_ids"], "node_ids")
            ),
            node_source_ranks=tuple(
                tuple(
                    _require_int("source_rank", source_rank)
                    for source_rank in _require_sequence(
                        node, "node_source_ranks entry"
                    )
                )
                for node in _require_sequence(
                    payload["node_source_ranks"], "node_source_ranks"
                )
            ),
            node_bytes=tuple(
                _require_u64("node_bytes entry", byte_count)
                for byte_count in _require_sequence(payload["node_bytes"], "node_bytes")
            ),
        )


@dataclass(frozen=True, slots=True)
class AssignmentLocalityStats:
    """Predicted consumer traffic for one source-file assignment."""

    node_ids: tuple[NodeId, ...]
    node_total_consumer_bytes: tuple[int, ...]
    node_local_consumer_bytes: tuple[int, ...]
    node_remote_consumer_bytes: tuple[int, ...]
    total_consumer_bytes: int
    local_consumer_bytes: int
    remote_consumer_bytes: int


@dataclass(frozen=True, slots=True)
class SourceAssignmentResult:
    """A balanced assignment and its predicted locality bounds."""

    assignment: FileAssignment
    baseline_locality: AssignmentLocalityStats
    chosen_locality: AssignmentLocalityStats
    theoretical_max_local_consumer_bytes: int


@dataclass(frozen=True, slots=True)
class _CooperativePlanningResult:
    assignment_result: SourceAssignmentResult
    batches: tuple[PipelineBatch, ...]
    works: tuple[BatchNodeWork, ...]
    assignment_ns: int = field(compare=False, repr=False)
    batching_ns: int = field(compare=False, repr=False)
    work_ns: int = field(compare=False, repr=False)


@dataclass(frozen=True)
class PipelineBatch:
    batch_index: int
    demands: tuple[FqnDemand, ...]

    def __post_init__(self) -> None:
        _require_int("batch_index", self.batch_index)
        demands = tuple(sorted(self.demands, key=_demand_key))
        if not demands:
            raise ValueError("a pipeline batch must contain at least one FQN")
        fqns = [demand.fqn for demand in demands]
        if len(fqns) != len(set(fqns)):
            raise ValueError("an FQN may occur only once in a pipeline batch")
        object.__setattr__(self, "demands", demands)

    @property
    def total_bytes(self) -> int:
        return sum(demand.total_bytes for demand in self.demands)

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_index": self.batch_index,
            "demands": [demand.to_dict() for demand in self.demands],
        }

    @classmethod
    def from_dict(cls, value: object) -> PipelineBatch:
        payload = _require_mapping(value)
        _require_keys(payload, {"batch_index", "demands"})
        return cls(
            batch_index=_require_int("batch_index", payload["batch_index"]),
            demands=tuple(
                FqnDemand.from_dict(item)
                for item in _require_sequence(payload["demands"], "demands")
            ),
        )


@dataclass(frozen=True)
class BatchNodeWork:
    batch_index: int
    node_id: NodeId
    fqn_names: tuple[str, ...]
    exact_ranges: tuple[ByteRange, ...]
    download_ranges: tuple[ByteRange, ...]

    def __post_init__(self) -> None:
        _require_int("batch_index", self.batch_index)
        node_id = _require_node_id("node_id", self.node_id)
        names = tuple(sorted({_require_fqn("fqn", name) for name in self.fqn_names}))
        exact = union_byte_ranges(self.exact_ranges)
        downloads = union_byte_ranges(self.download_ranges)
        missing = subtract_byte_ranges(exact, downloads)
        if missing:
            raise ValueError(f"download ranges do not cover exact ranges: {missing!r}")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "fqn_names", names)
        object.__setattr__(self, "exact_ranges", exact)
        object.__setattr__(self, "download_ranges", downloads)

    @property
    def exact_bytes(self) -> int:
        return sum(byte_range.length for byte_range in self.exact_ranges)

    @property
    def download_bytes(self) -> int:
        return sum(byte_range.length for byte_range in self.download_ranges)

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_index": self.batch_index,
            "download_ranges": [item.to_dict() for item in self.download_ranges],
            "exact_ranges": [item.to_dict() for item in self.exact_ranges],
            "fqn_names": list(self.fqn_names),
            "node_id": self.node_id,
        }

    @classmethod
    def _from_canonical(
        cls,
        *,
        batch_index: int,
        node_id: NodeId,
        fqn_names: tuple[str, ...],
        exact_ranges: tuple[ByteRange, ...],
        download_ranges: tuple[ByteRange, ...],
    ) -> BatchNodeWork:
        instance = object.__new__(cls)
        object.__setattr__(instance, "batch_index", batch_index)
        object.__setattr__(instance, "node_id", node_id)
        object.__setattr__(instance, "fqn_names", fqn_names)
        object.__setattr__(instance, "exact_ranges", exact_ranges)
        object.__setattr__(instance, "download_ranges", download_ranges)
        return instance

    @classmethod
    def from_dict(cls, value: object) -> BatchNodeWork:
        payload = _require_mapping(value)
        _require_keys(
            payload,
            {
                "batch_index",
                "download_ranges",
                "exact_ranges",
                "fqn_names",
                "node_id",
            },
        )
        return cls(
            batch_index=_require_int("batch_index", payload["batch_index"]),
            node_id=_require_node_id("node_id", payload["node_id"]),
            fqn_names=tuple(
                _require_fqn("fqn", name)
                for name in _require_sequence(payload["fqn_names"], "fqn_names")
            ),
            exact_ranges=_ranges_from_wire(payload["exact_ranges"], "exact_ranges"),
            download_ranges=_ranges_from_wire(
                payload["download_ranges"], "download_ranges"
            ),
        )


@dataclass(frozen=True, slots=True)
class ProjectedSourceSchedule:
    """One source file's ordered download schedule within one batch."""

    batch_index: int
    source_rank: int
    owner_node_id: NodeId
    ranges: tuple[ByteRange, ...]
    starts: tuple[int, ...] = field(init=False)
    cumulative_bytes: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        _require_int("batch_index", self.batch_index)
        _require_int("source_rank", self.source_rank)
        owner_node_id = _require_node_id("owner_node_id", self.owner_node_id)
        ranges = tuple(self.ranges)
        _validate_source_schedule_ranges(self.source_rank, ranges)
        cumulative = 0
        cumulative_bytes: list[int] = []
        for byte_range in ranges:
            cumulative = _require_u64(
                "source schedule cumulative bytes",
                cumulative + byte_range.length,
            )
            cumulative_bytes.append(cumulative)
        object.__setattr__(self, "owner_node_id", owner_node_id)
        object.__setattr__(self, "ranges", ranges)
        object.__setattr__(
            self,
            "starts",
            tuple(byte_range.offset for byte_range in ranges),
        )
        object.__setattr__(self, "cumulative_bytes", tuple(cumulative_bytes))

    @property
    def download_bytes(self) -> int:
        return self.cumulative_bytes[-1] if self.cumulative_bytes else 0


@dataclass(frozen=True, slots=True)
class ProjectedBatchDownload:
    """The local node leader's complete download descriptor for one batch."""

    batch_index: int
    node_id: NodeId
    download_ranges: tuple[ByteRange, ...]

    def __post_init__(self) -> None:
        _require_int("batch_index", self.batch_index)
        node_id = _require_node_id("node_id", self.node_id)
        download_ranges = tuple(self.download_ranges)
        _validate_canonical_download_ranges(download_ranges)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "download_ranges", download_ranges)

    @property
    def download_bytes(self) -> int:
        return sum(byte_range.length for byte_range in self.download_ranges)


@dataclass(frozen=True, slots=True)
class ProjectedExecutionPlan:
    """An immutable rank-local projection of the compact execution plan."""

    node_ids: tuple[NodeId, ...]
    batch_indices: tuple[int, ...]
    local_target_indices_by_batch: tuple[tuple[int, ...], ...]
    source_owners: Mapping[int, NodeId]
    source_schedules: Mapping[tuple[int, int], ProjectedSourceSchedule]
    active_node_ids_by_batch: tuple[tuple[NodeId, ...], ...]
    local_downloads: tuple[ProjectedBatchDownload, ...]

    def __post_init__(self) -> None:
        node_ids = _ordered_node_ids(self.node_ids, "projected node_ids")
        batch_indices = _validated_projected_batch_indices(self.batch_indices)
        target_indices = _validated_projected_target_indices(
            self.local_target_indices_by_batch,
        )
        source_owners, source_schedules = _validated_projected_sources(
            self.source_owners,
            self.source_schedules,
            node_ids,
            len(batch_indices),
        )
        active_node_ids_by_batch = _validated_projected_active_nodes(
            self.active_node_ids_by_batch,
            node_ids,
        )
        local_downloads = _validated_projected_local_downloads(
            self.local_downloads,
            batch_indices,
        )
        _validate_projected_batch_array_lengths(
            batch_indices,
            target_indices,
            active_node_ids_by_batch,
            local_downloads,
        )
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "batch_indices", batch_indices)
        object.__setattr__(self, "local_target_indices_by_batch", target_indices)
        object.__setattr__(
            self,
            "source_owners",
            MappingProxyType(source_owners),
        )
        object.__setattr__(
            self,
            "source_schedules",
            MappingProxyType(source_schedules),
        )
        object.__setattr__(
            self,
            "active_node_ids_by_batch",
            active_node_ids_by_batch,
        )
        object.__setattr__(self, "local_downloads", local_downloads)

    def owner_for(self, source_rank: int) -> NodeId:
        try:
            return self.source_owners[source_rank]
        except KeyError as error:
            raise ValueError(
                f"source rank {source_rank} has no projected owner"
            ) from error

    def schedule_for(
        self,
        batch_index: int,
        source_rank: int,
    ) -> ProjectedSourceSchedule:
        try:
            return self.source_schedules[(batch_index, source_rank)]
        except KeyError as error:
            raise ValueError(
                f"source rank {source_rank} has no projected schedule in "
                f"batch {batch_index}"
            ) from error


def _validated_projected_batch_indices(
    batch_indices: Iterable[int],
) -> tuple[int, ...]:
    result = tuple(
        _require_int("batch_index", batch_index) for batch_index in batch_indices
    )
    if result != tuple(range(len(result))):
        raise ValueError("projected batch indices must be contiguous and zero-based")
    return result


def _validated_projected_target_indices(
    target_indices_by_batch: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], ...]:
    result = tuple(
        tuple(
            _require_int("local target index", target_index) for target_index in indices
        )
        for indices in target_indices_by_batch
    )
    flattened = tuple(target_index for indices in result for target_index in indices)
    if len(flattened) != len(set(flattened)):
        raise ValueError("projected local target indices must be unique")
    return result


def _validated_projected_sources(
    raw_source_owners: Mapping[int, NodeId],
    raw_source_schedules: Mapping[tuple[int, int], ProjectedSourceSchedule],
    node_ids: tuple[NodeId, ...],
    batch_count: int,
) -> tuple[
    dict[int, NodeId],
    dict[tuple[int, int], ProjectedSourceSchedule],
]:
    source_owners = {
        _require_int("source_rank", source_rank): _require_node_id(
            "owner_node_id", owner_node_id
        )
        for source_rank, owner_node_id in raw_source_owners.items()
    }
    if any(owner not in node_ids for owner in source_owners.values()):
        raise ValueError("projected source owner is not in the topology")
    source_schedules = dict(raw_source_schedules)
    for key, schedule in source_schedules.items():
        if key != (schedule.batch_index, schedule.source_rank):
            raise ValueError("projected source schedule key is invalid")
        if schedule.batch_index >= batch_count:
            raise ValueError("projected source schedule batch is out of range")
        if source_owners.get(schedule.source_rank) != schedule.owner_node_id:
            raise ValueError("projected source schedule owner differs")
    return source_owners, source_schedules


def _validated_projected_active_nodes(
    raw_active_nodes: Iterable[Iterable[NodeId]],
    node_ids: tuple[NodeId, ...],
) -> tuple[tuple[NodeId, ...], ...]:
    result = tuple(tuple(nodes) for nodes in raw_active_nodes)
    for active_node_ids in result:
        canonical = tuple(node_id for node_id in node_ids if node_id in active_node_ids)
        if canonical != active_node_ids:
            raise ValueError("active nodes must follow topology order and be unique")
    return result


def _validated_projected_local_downloads(
    raw_downloads: Iterable[ProjectedBatchDownload],
    batch_indices: tuple[int, ...],
) -> tuple[ProjectedBatchDownload, ...]:
    downloads = tuple(raw_downloads)
    if tuple(download.batch_index for download in downloads) != batch_indices:
        raise ValueError("local downloads must follow batch order")
    if downloads and len({item.node_id for item in downloads}) != 1:
        raise ValueError("local downloads must belong to one node")
    return downloads


def _validate_projected_batch_array_lengths(
    batch_indices: Sequence[int],
    target_indices: Sequence[Sequence[int]],
    active_nodes: Sequence[Sequence[NodeId]],
    local_downloads: Sequence[ProjectedBatchDownload],
) -> None:
    if not (
        len(target_indices)
        == len(active_nodes)
        == len(local_downloads)
        == len(batch_indices)
    ):
        raise ValueError("projected batch arrays must have matching lengths")


def execution_plan_to_wire(
    assignment: FileAssignment,
    batches: Sequence[PipelineBatch],
    works: Sequence[BatchNodeWork],
) -> dict[str, object]:
    """Encode a validated execution plan without repeating global demands."""

    planned, ordered_works = _validate_execution_plan_equivalence(
        assignment,
        batches,
        works,
    )
    return _execution_plan_to_wire_from_validated(
        assignment,
        planned,
        ordered_works,
    )


def _execution_plan_to_wire_from_validated(
    assignment: FileAssignment,
    batches: Sequence[PipelineBatch],
    works: Sequence[BatchNodeWork],
) -> dict[str, object]:
    """Encode outputs already validated by the cooperative planning pipeline."""

    planned = tuple(batches)
    ordered_works = tuple(works)
    fqn_batches = sorted(
        (demand.fqn, batch.batch_index) for batch in planned for demand in batch.demands
    )
    ranges_by_source_and_batch: dict[tuple[int, int], tuple[ByteRange, ...]] = {}
    for work in ordered_works:
        ranges_by_source: dict[int, list[ByteRange]] = defaultdict(list)
        for byte_range in work.download_ranges:
            ranges_by_source[byte_range.source_rank].append(byte_range)
        for source_rank, source_ranges in ranges_by_source.items():
            ranges_by_source_and_batch[(source_rank, work.batch_index)] = tuple(
                source_ranges
            )

    sources: list[object] = []
    for source_rank in sorted(assignment.source_rank_to_node):
        schedules = [
            [
                batch_index,
                [
                    [byte_range.offset, byte_range.length]
                    for byte_range in ranges_by_source_and_batch[
                        (source_rank, batch_index)
                    ]
                ],
            ]
            for batch_index in range(len(planned))
            if (source_rank, batch_index) in ranges_by_source_and_batch
        ]
        if not schedules:
            raise ValueError(
                f"source rank {source_rank} has no final download schedule"
            )
        sources.append([source_rank, assignment.owner_for(source_rank), schedules])
    return {
        "batch_count": len(planned),
        "fqn_batches": [list(item) for item in fqn_batches],
        "node_ids": list(assignment.node_ids),
        "sources": sources,
        "version": EXECUTION_PLAN_WIRE_VERSION,
    }


def project_execution_plan_wire(
    value: object,
    *,
    expected_node_ids: Sequence[NodeId],
    local_node_id: NodeId,
    local_targets: Sequence[object],
    node_capacities: Mapping[NodeId, int],
) -> ProjectedExecutionPlan:
    """Strictly decode and project a compact execution plan for one rank."""

    payload = _strict_wire_mapping(value, "execution plan")
    wire_nodes, local_node, capacities, batch_count = _execution_plan_wire_header(
        payload,
        expected_node_ids,
        local_node_id,
        node_capacities,
    )
    fqn_to_batch = _fqn_batches_from_execution_wire(
        payload["fqn_batches"],
        batch_count,
    )
    source_owners, source_schedules, bytes_by_batch_and_node = (
        _sources_from_execution_wire(
            payload["sources"],
            wire_nodes,
            batch_count,
        )
    )
    _validate_execution_plan_capacities(bytes_by_batch_and_node, capacities)
    target_indices_by_batch = _project_local_target_indices(
        local_targets,
        batch_count,
        fqn_to_batch,
        source_owners,
        source_schedules,
    )
    active_node_ids_by_batch = _active_nodes_by_batch(
        wire_nodes,
        batch_count,
        bytes_by_batch_and_node,
    )
    local_downloads = _local_downloads_by_batch(
        local_node,
        batch_count,
        source_schedules,
    )
    return ProjectedExecutionPlan(
        node_ids=wire_nodes,
        batch_indices=tuple(range(batch_count)),
        local_target_indices_by_batch=tuple(
            tuple(indices) for indices in target_indices_by_batch
        ),
        source_owners=source_owners,
        source_schedules=source_schedules,
        active_node_ids_by_batch=active_node_ids_by_batch,
        local_downloads=local_downloads,
    )


def _execution_plan_wire_header(
    payload: Mapping[str, object],
    expected_node_ids: Sequence[NodeId],
    local_node_id: NodeId,
    node_capacities: Mapping[NodeId, int],
) -> tuple[tuple[NodeId, ...], NodeId, Mapping[NodeId, int], int]:
    _require_keys(
        payload,
        {"batch_count", "fqn_batches", "node_ids", "sources", "version"},
    )
    version = _require_int("version", payload["version"], minimum=1)
    if version != EXECUTION_PLAN_WIRE_VERSION:
        raise ValueError(f"unsupported execution-plan wire version {version!r}")
    expected_nodes = _ordered_node_ids(expected_node_ids, "expected_node_ids")
    wire_nodes = _ordered_node_ids(
        (
            _require_node_id("node_id", node_id)
            for node_id in _strict_wire_list(payload["node_ids"], "node_ids")
        ),
        "node_ids",
    )
    if wire_nodes != expected_nodes:
        raise ValueError(
            "execution-plan node IDs/order do not match the local topology"
        )
    local_node = _require_node_id("local_node_id", local_node_id)
    if local_node not in wire_nodes:
        raise ValueError("local_node_id is not in the execution-plan topology")
    return (
        wire_nodes,
        local_node,
        _validated_node_capacities(node_capacities, wire_nodes),
        _require_u64("batch_count", payload["batch_count"]),
    )


def _fqn_batches_from_execution_wire(
    value: object,
    batch_count: int,
) -> dict[str, int]:
    raw_fqn_batches = _strict_wire_list(value, "fqn_batches")
    if batch_count > len(raw_fqn_batches) and batch_count != 0:
        raise ValueError("every execution-plan batch must contain an FQN")
    fqn_to_batch: dict[str, int] = {}
    previous_fqn: str | None = None
    observed_batches: set[int] = set()
    for raw_entry in raw_fqn_batches:
        fqn, batch_index = _fqn_batch_from_wire(raw_entry, batch_count)
        if previous_fqn is not None and fqn <= previous_fqn:
            raise ValueError("fqn_batches must be strictly sorted by unique FQN")
        previous_fqn = fqn
        fqn_to_batch[fqn] = batch_index
        observed_batches.add(batch_index)
    if observed_batches != set(range(batch_count)):
        raise ValueError("fqn_batches do not cover every execution-plan batch")
    return fqn_to_batch


def _fqn_batch_from_wire(value: object, batch_count: int) -> tuple[str, int]:
    entry = _strict_wire_list(value, "fqn_batches entry")
    if len(entry) != 2:
        raise ValueError("each fqn_batches entry must contain two values")
    fqn = _require_fqn("fqn", entry[0])
    batch_index = _require_int("batch_index", entry[1])
    if batch_index >= batch_count:
        raise ValueError("FQN batch index is outside batch_count")
    return fqn, batch_index


def _sources_from_execution_wire(
    value: object,
    node_ids: tuple[NodeId, ...],
    batch_count: int,
) -> tuple[
    dict[int, NodeId],
    dict[tuple[int, int], ProjectedSourceSchedule],
    Counter[tuple[int, NodeId]],
]:
    source_owners: dict[int, NodeId] = {}
    source_schedules: dict[tuple[int, int], ProjectedSourceSchedule] = {}
    bytes_by_batch_and_node: Counter[tuple[int, NodeId]] = Counter()
    previous_source_rank = -1
    for raw_source in _strict_wire_list(value, "sources"):
        source_rank, owner_node_id, schedules = _source_from_execution_wire(
            raw_source,
            node_ids,
            batch_count,
        )
        if source_rank <= previous_source_rank:
            raise ValueError("sources must be strictly source-rank sorted and unique")
        previous_source_rank = source_rank
        source_owners[source_rank] = owner_node_id
        for schedule in schedules:
            source_schedules[(schedule.batch_index, source_rank)] = schedule
            key = (schedule.batch_index, owner_node_id)
            bytes_by_batch_and_node[key] = _require_u64(
                "batch node download bytes",
                bytes_by_batch_and_node[key] + schedule.download_bytes,
            )
    scheduled_batches = {batch_index for batch_index, _ in source_schedules}
    if scheduled_batches != set(range(batch_count)):
        raise ValueError("source schedules do not cover every execution-plan batch")
    _validate_cross_batch_source_schedules(source_schedules.values())
    return source_owners, source_schedules, bytes_by_batch_and_node


def _source_from_execution_wire(
    value: object,
    node_ids: tuple[NodeId, ...],
    batch_count: int,
) -> tuple[int, NodeId, tuple[ProjectedSourceSchedule, ...]]:
    source = _strict_wire_list(value, "sources entry")
    if len(source) != 3:
        raise ValueError("each sources entry must contain three values")
    source_rank = _require_int("source_rank", source[0])
    owner_node_id = _require_node_id("owner_node_id", source[1])
    if owner_node_id not in node_ids:
        raise ValueError("source owner is not in the execution-plan topology")
    raw_schedules = _strict_wire_list(source[2], "source schedules")
    if not raw_schedules:
        raise ValueError("each source must contain at least one batch schedule")
    schedules = tuple(
        _source_schedule_from_execution_wire(
            raw_schedule,
            source_rank,
            owner_node_id,
            batch_count,
        )
        for raw_schedule in raw_schedules
    )
    if tuple(item.batch_index for item in schedules) != tuple(
        sorted({item.batch_index for item in schedules})
    ):
        raise ValueError("source schedules must be strictly batch-sorted and unique")
    return source_rank, owner_node_id, schedules


def _source_schedule_from_execution_wire(
    value: object,
    source_rank: int,
    owner_node_id: NodeId,
    batch_count: int,
) -> ProjectedSourceSchedule:
    schedule = _strict_wire_list(value, "source schedule")
    if len(schedule) != 2:
        raise ValueError("each source schedule must contain two values")
    batch_index = _require_int("batch_index", schedule[0])
    if batch_index >= batch_count:
        raise ValueError("source schedule batch is outside batch_count")
    raw_ranges = _strict_wire_list(schedule[1], "download ranges")
    if not raw_ranges:
        raise ValueError("a source schedule must contain a download range")
    return ProjectedSourceSchedule(
        batch_index=batch_index,
        source_rank=source_rank,
        owner_node_id=owner_node_id,
        ranges=_source_ranges_from_compact_wire(source_rank, raw_ranges),
    )


def _validate_execution_plan_capacities(
    bytes_by_batch_and_node: Mapping[tuple[int, NodeId], int],
    capacities: Mapping[NodeId, int],
) -> None:
    for (batch_index, node_id), byte_count in bytes_by_batch_and_node.items():
        if byte_count > capacities[node_id]:
            raise ValueError(
                f"batch {batch_index} needs {byte_count} bytes on node "
                f"{node_id!r}, whose capacity is {capacities[node_id]} bytes"
            )


def _project_local_target_indices(
    local_targets: Sequence[object],
    batch_count: int,
    fqn_to_batch: Mapping[str, int],
    source_owners: Mapping[int, NodeId],
    source_schedules: Mapping[tuple[int, int], ProjectedSourceSchedule],
) -> tuple[tuple[int, ...], ...]:
    target_indices_by_batch: list[list[int]] = [[] for _ in range(batch_count)]
    for target_index, target in enumerate(local_targets):
        projected = _project_one_local_target(
            target,
            fqn_to_batch,
            source_owners,
            source_schedules,
        )
        if projected is not None:
            target_indices_by_batch[projected].append(target_index)
    return tuple(tuple(indices) for indices in target_indices_by_batch)


def _project_one_local_target(
    target: object,
    fqn_to_batch: Mapping[str, int],
    source_owners: Mapping[int, NodeId],
    source_schedules: Mapping[tuple[int, int], ProjectedSourceSchedule],
) -> int | None:
    pattern = _local_target_field(target, "source_pattern")
    dense_nbytes = _require_u64(
        "local target dense_nbytes",
        _local_target_field(pattern, "dense_nbytes"),
    )
    if dense_nbytes == 0:
        return None
    fqn = _require_fqn(
        "local target target_fqn",
        _local_target_field(target, "target_fqn"),
    )
    if fqn not in fqn_to_batch:
        raise ValueError(f"local target FQN {fqn!r} is absent from the execution plan")
    batch_index = fqn_to_batch[fqn]
    source_rank = _require_int(
        "local target source_rank",
        _local_target_field(target, "source_rank"),
    )
    if source_rank not in source_owners:
        raise ValueError(
            f"local target source rank {source_rank} has no execution-plan owner"
        )
    schedule = _local_target_schedule(
        batch_index,
        source_rank,
        fqn,
        source_schedules,
    )
    missing = subtract_byte_ranges(
        _local_target_source_ranges(target, source_rank),
        schedule.ranges,
    )
    if missing:
        raise ValueError(
            f"local target source ranges are not covered in FQN {fqn!r}'s "
            f"batch {batch_index}: {missing!r}"
        )
    return batch_index


def _local_target_schedule(
    batch_index: int,
    source_rank: int,
    fqn: str,
    source_schedules: Mapping[tuple[int, int], ProjectedSourceSchedule],
) -> ProjectedSourceSchedule:
    schedule = source_schedules.get((batch_index, source_rank))
    if schedule is not None:
        return schedule
    other_batches = sorted(
        scheduled_batch
        for scheduled_batch, scheduled_source in source_schedules
        if scheduled_source == source_rank
    )
    raise ValueError(
        f"local target source rank {source_rank} is not scheduled in "
        f"FQN {fqn!r}'s batch {batch_index}; scheduled batches are {other_batches}"
    )


def _active_nodes_by_batch(
    node_ids: tuple[NodeId, ...],
    batch_count: int,
    bytes_by_batch_and_node: Mapping[tuple[int, NodeId], int],
) -> tuple[tuple[NodeId, ...], ...]:
    return tuple(
        tuple(
            node_id
            for node_id in node_ids
            if bytes_by_batch_and_node.get((batch_index, node_id), 0) > 0
        )
        for batch_index in range(batch_count)
    )


def _local_downloads_by_batch(
    local_node_id: NodeId,
    batch_count: int,
    source_schedules: Mapping[tuple[int, int], ProjectedSourceSchedule],
) -> tuple[ProjectedBatchDownload, ...]:
    return tuple(
        ProjectedBatchDownload(
            batch_index=batch_index,
            node_id=local_node_id,
            download_ranges=tuple(
                sorted(
                    byte_range
                    for (scheduled_batch, _), schedule in source_schedules.items()
                    if scheduled_batch == batch_index
                    and schedule.owner_node_id == local_node_id
                    for byte_range in schedule.ranges
                )
            ),
        )
        for batch_index in range(batch_count)
    )


def _validate_execution_plan_equivalence(
    assignment: FileAssignment,
    batches: Sequence[PipelineBatch],
    works: Sequence[BatchNodeWork],
) -> tuple[tuple[PipelineBatch, ...], tuple[BatchNodeWork, ...]]:
    planned = tuple(sorted(batches, key=lambda batch: batch.batch_index))
    if tuple(batch.batch_index for batch in planned) != tuple(range(len(planned))):
        raise ValueError("pipeline batch indices must be contiguous and zero-based")
    work_by_key = _index_execution_works(works, assignment, len(planned))
    expected_keys = {
        (batch.batch_index, node_id)
        for batch in planned
        for node_id in assignment.node_ids
    }
    if set(work_by_key) != expected_keys:
        raise ValueError("batch works do not cover every batch/node pair exactly once")

    expected_exact_by_key: dict[tuple[int, NodeId], list[ByteRange]] = {
        key: [] for key in expected_keys
    }
    seen_fqns: set[str] = set()
    demanded_source_ranks: set[int] = set()
    for batch in planned:
        for demand in batch.demands:
            if demand.fqn in seen_fqns:
                raise ValueError(
                    f"FQN {demand.fqn!r} occurs in multiple pipeline batches"
                )
            seen_fqns.add(demand.fqn)
            for byte_range in demand.ranges:
                demanded_source_ranks.add(byte_range.source_rank)
                expected_exact_by_key[
                    (
                        batch.batch_index,
                        assignment.owner_for(byte_range.source_rank),
                    )
                ].append(byte_range)
    if demanded_source_ranks != set(assignment.source_rank_to_node):
        raise ValueError(
            "file assignment does not cover exactly the planned source ranks"
        )

    for batch in planned:
        expected_fqns = {demand.fqn for demand in batch.demands}
        canonical_fqns = work_by_key[
            (batch.batch_index, assignment.node_ids[0])
        ].fqn_names
        if len(canonical_fqns) != len(expected_fqns) or any(
            fqn not in expected_fqns for fqn in canonical_fqns
        ):
            raise ValueError("batch work FQNs differ from its pipeline batch")
        for node_id in assignment.node_ids:
            _validate_execution_work(
                work_by_key[(batch.batch_index, node_id)],
                node_id,
                assignment,
                canonical_fqns,
                union_byte_ranges(expected_exact_by_key[(batch.batch_index, node_id)]),
            )
    ordered_works = tuple(
        work_by_key[(batch.batch_index, node_id)]
        for batch in planned
        for node_id in assignment.node_ids
    )
    _validate_disjoint_batch_downloads(ordered_works)
    return planned, ordered_works


def _index_execution_works(
    works: Sequence[BatchNodeWork],
    assignment: FileAssignment,
    batch_count: int,
) -> dict[tuple[int, NodeId], BatchNodeWork]:
    result: dict[tuple[int, NodeId], BatchNodeWork] = {}
    for work in works:
        if work.batch_index >= batch_count:
            raise ValueError("batch work index is outside the pipeline plan")
        if work.node_id not in assignment.node_ids:
            raise ValueError("batch work node is outside the file assignment")
        key = (work.batch_index, work.node_id)
        if key in result:
            raise ValueError(f"duplicate batch/node work for {key!r}")
        result[key] = work
    return result


def _validate_execution_work(
    work: BatchNodeWork,
    node_id: NodeId,
    assignment: FileAssignment,
    expected_fqns: tuple[str, ...],
    expected_exact: tuple[ByteRange, ...],
) -> None:
    if work.fqn_names is not expected_fqns and work.fqn_names != expected_fqns:
        raise ValueError("batch work FQNs differ from its pipeline batch")
    if work.exact_ranges != expected_exact:
        raise ValueError("batch work exact ranges differ from the pipeline plan")
    if any(
        assignment.owner_for(byte_range.source_rank) != node_id
        for byte_range in work.download_ranges
    ):
        raise ValueError("batch work downloads a source owned by another node")


def _strict_wire_mapping(value: object, name: str) -> Mapping[str, object]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return value


def _strict_wire_list(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return value


def _ordered_node_ids(node_ids: Iterable[NodeId], name: str) -> tuple[NodeId, ...]:
    result = tuple(_require_node_id("node_id", node_id) for node_id in node_ids)
    if not result:
        raise ValueError(f"{name} must contain at least one node")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique node IDs")
    return result


def _validated_node_capacities(
    node_capacities: Mapping[NodeId, int],
    node_ids: tuple[NodeId, ...],
) -> Mapping[NodeId, int]:
    if not isinstance(node_capacities, Mapping):
        raise ValueError("node_capacities must be a mapping")
    capacities: dict[NodeId, int] = {}
    for raw_node_id, raw_capacity in node_capacities.items():
        node_id = _require_node_id("node capacity node_id", raw_node_id)
        capacities[node_id] = _require_u64("node capacity", raw_capacity)
    if set(capacities) != set(node_ids):
        raise ValueError("node capacities must cover exactly the topology nodes")
    return MappingProxyType(capacities)


def _source_ranges_from_compact_wire(
    source_rank: int,
    raw_ranges: Sequence[object],
) -> tuple[ByteRange, ...]:
    ranges: list[ByteRange] = []
    for raw_range in raw_ranges:
        compact_range = _strict_wire_list(raw_range, "download range")
        if len(compact_range) != 2:
            raise ValueError("each download range must contain two values")
        offset = _require_u64("offset", compact_range[0])
        length = _require_u64("length", compact_range[1], minimum=1)
        _checked_end(offset, length)
        ranges.append(ByteRange(source_rank, offset, length))
    result = tuple(ranges)
    _validate_source_schedule_ranges(source_rank, result)
    return result


def _validate_source_schedule_ranges(
    source_rank: int,
    ranges: Sequence[ByteRange],
) -> None:
    if not ranges:
        raise ValueError("a source schedule must contain at least one range")
    previous_end = -1
    for byte_range in ranges:
        if not isinstance(byte_range, ByteRange):
            raise ValueError("source schedule ranges must be ByteRange values")
        if byte_range.source_rank != source_rank:
            raise ValueError("source schedule contains the wrong source rank")
        if byte_range.offset <= previous_end:
            raise ValueError(
                "source schedule ranges must be strictly ordered, disjoint, "
                "and non-adjacent"
            )
        previous_end = byte_range.end


def _validate_canonical_download_ranges(ranges: Sequence[ByteRange]) -> None:
    if tuple(ranges) != tuple(sorted(ranges)):
        raise ValueError("download ranges must follow source-rank/offset order")
    previous_by_source: dict[int, int] = {}
    for byte_range in ranges:
        if not isinstance(byte_range, ByteRange):
            raise ValueError("download ranges must be ByteRange values")
        previous_end = previous_by_source.get(byte_range.source_rank, -1)
        if byte_range.offset <= previous_end:
            raise ValueError(
                "download ranges must be disjoint and non-adjacent per source"
            )
        previous_by_source[byte_range.source_rank] = byte_range.end


def _validate_cross_batch_source_schedules(
    schedules: Iterable[ProjectedSourceSchedule],
) -> None:
    tagged_by_source: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for schedule in schedules:
        tagged_by_source[schedule.source_rank].extend(
            (byte_range.offset, byte_range.end, schedule.batch_index)
            for byte_range in schedule.ranges
        )
    for source_rank, tagged in tagged_by_source.items():
        active_end = -1
        active_batch = -1
        for offset, end, batch_index in sorted(tagged):
            if offset < active_end and batch_index != active_batch:
                raise ValueError(
                    f"source rank {source_rank} download ranges overlap across "
                    "pipeline batches"
                )
            if end > active_end:
                active_end = end
                active_batch = batch_index


def _local_target_field(target: object, name: str) -> object:
    try:
        return getattr(target, name)
    except AttributeError as error:
        raise ValueError(f"local target is missing {name}") from error


def _local_target_source_ranges(
    target: object,
    source_rank: int,
) -> tuple[ByteRange, ...]:
    pattern = _local_target_field(target, "source_pattern")
    try:
        raw_ranges = pattern.iter_ranges()
    except AttributeError as error:
        raise ValueError("local target source_pattern has no iter_ranges") from error
    ranges: list[ByteRange] = []
    for raw_range in raw_ranges:
        offset = _require_u64(
            "local target source offset",
            _local_target_field(raw_range, "offset"),
        )
        length = _require_u64(
            "local target source length",
            _local_target_field(raw_range, "length"),
            minimum=1,
        )
        _checked_end(offset, length)
        ranges.append(ByteRange(source_rank, offset, length))
    return union_byte_ranges(ranges)


@dataclass(frozen=True)
class ChunkPiece:
    chunk_slot: int
    offset_in_chunk: int
    length: int

    def __post_init__(self) -> None:
        _require_int("chunk_slot", self.chunk_slot)
        _require_u64("offset_in_chunk", self.offset_in_chunk)
        _require_u64("length", self.length, minimum=1)

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_slot": self.chunk_slot,
            "length": self.length,
            "offset_in_chunk": self.offset_in_chunk,
        }

    @classmethod
    def from_dict(cls, value: object) -> ChunkPiece:
        payload = _require_mapping(value)
        _require_keys(payload, {"chunk_slot", "length", "offset_in_chunk"})
        return cls(
            chunk_slot=_require_int("chunk_slot", payload["chunk_slot"]),
            offset_in_chunk=_require_u64("offset_in_chunk", payload["offset_in_chunk"]),
            length=_require_u64("length", payload["length"], minimum=1),
        )


PlanningRecord: TypeAlias = (
    NodeMembership
    | RankTopology
    | CoordinationWorld
    | ByteRange
    | FqnDemand
    | FileAssignment
    | PipelineBatch
    | BatchNodeWork
    | ChunkPiece
)


def _ranges_from_wire(value: object, name: str) -> tuple[ByteRange, ...]:
    return tuple(ByteRange.from_dict(item) for item in _require_sequence(value, name))


def _record_type_and_payload(record: PlanningRecord) -> tuple[str, dict[str, object]]:
    record_types: tuple[tuple[type[object], str], ...] = (
        (NodeMembership, "node_membership"),
        (RankTopology, "rank_topology"),
        (CoordinationWorld, "coordination_world"),
        (ByteRange, "byte_range"),
        (FqnDemand, "fqn_demand"),
        (FileAssignment, "file_assignment"),
        (PipelineBatch, "pipeline_batch"),
        (BatchNodeWork, "batch_node_work"),
        (ChunkPiece, "chunk_piece"),
    )
    for record_class, wire_name in record_types:
        if isinstance(record, record_class):
            return wire_name, record.to_dict()
    raise TypeError(f"unsupported planning record: {type(record).__name__}")


def planning_record_to_json(record: PlanningRecord) -> str:
    record_type, payload = _record_type_and_payload(record)
    envelope = {"payload": payload, "type": record_type, "version": WIRE_VERSION}
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"))


def planning_record_from_json(encoded: str | bytes) -> PlanningRecord:
    try:
        envelope = _require_mapping(json.loads(encoded), "wire envelope")
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("invalid planning JSON") from error
    _require_keys(envelope, {"payload", "type", "version"})
    version = _require_int("version", envelope["version"], minimum=1)
    if version != WIRE_VERSION:
        raise ValueError(f"unsupported planning wire version {envelope['version']!r}")
    record_type = _require_str("type", envelope["type"])
    decoders = {
        "batch_node_work": BatchNodeWork.from_dict,
        "byte_range": ByteRange.from_dict,
        "chunk_piece": ChunkPiece.from_dict,
        "coordination_world": CoordinationWorld.from_dict,
        "file_assignment": FileAssignment.from_dict,
        "fqn_demand": FqnDemand.from_dict,
        "node_membership": NodeMembership.from_dict,
        "pipeline_batch": PipelineBatch.from_dict,
        "rank_topology": RankTopology.from_dict,
    }
    try:
        decoder = decoders[record_type]
    except KeyError as error:
        raise ValueError(f"unsupported planning record type {record_type!r}") from error
    return decoder(envelope["payload"])


def union_byte_ranges(ranges: Iterable[ByteRange]) -> tuple[ByteRange, ...]:
    ordered = sorted(ranges)
    merged: list[ByteRange] = []
    for byte_range in ordered:
        if not merged or byte_range.source_rank != merged[-1].source_rank:
            merged.append(byte_range)
            continue
        previous = merged[-1]
        if byte_range.offset > previous.end:
            merged.append(byte_range)
            continue
        merged[-1] = ByteRange(
            source_rank=previous.source_rank,
            offset=previous.offset,
            length=max(previous.end, byte_range.end) - previous.offset,
        )
    return tuple(merged)


def consolidate_byte_ranges(
    ranges: Iterable[ByteRange], gap_bytes: int
) -> tuple[ByteRange, ...]:
    gap = _require_u64("gap_bytes", gap_bytes)
    ordered = union_byte_ranges(ranges)
    consolidated: list[ByteRange] = []
    for byte_range in ordered:
        if not consolidated or byte_range.source_rank != consolidated[-1].source_rank:
            consolidated.append(byte_range)
            continue
        previous = consolidated[-1]
        if byte_range.offset > previous.end + gap:
            consolidated.append(byte_range)
            continue
        consolidated[-1] = ByteRange(
            source_rank=previous.source_rank,
            offset=previous.offset,
            length=max(previous.end, byte_range.end) - previous.offset,
        )
    return tuple(consolidated)


def _ranges_by_source(ranges: Iterable[ByteRange]) -> dict[int, list[ByteRange]]:
    by_source: dict[int, list[ByteRange]] = defaultdict(list)
    for byte_range in union_byte_ranges(ranges):
        by_source[byte_range.source_rank].append(byte_range)
    return by_source


def _subtract_source_ranges(
    required: Sequence[ByteRange], covered: Sequence[ByteRange]
) -> list[ByteRange]:
    missing: list[ByteRange] = []
    covered_index = 0
    for required_range in required:
        cursor = required_range.offset
        while covered_index < len(covered) and covered[covered_index].end <= cursor:
            covered_index += 1
        scan_index = covered_index
        while (
            scan_index < len(covered)
            and covered[scan_index].offset < required_range.end
        ):
            item = covered[scan_index]
            if item.offset > cursor:
                missing.append(
                    ByteRange(required_range.source_rank, cursor, item.offset - cursor)
                )
            cursor = max(cursor, item.end)
            if cursor >= required_range.end:
                break
            scan_index += 1
        if cursor < required_range.end:
            missing.append(
                ByteRange(
                    required_range.source_rank,
                    cursor,
                    required_range.end - cursor,
                )
            )
    return missing


def subtract_byte_ranges(
    required: Iterable[ByteRange], covered: Iterable[ByteRange]
) -> tuple[ByteRange, ...]:
    required_by_source = _ranges_by_source(required)
    covered_by_source = _ranges_by_source(covered)
    missing: list[ByteRange] = []
    for source_rank in sorted(required_by_source):
        missing.extend(
            _subtract_source_ranges(
                required_by_source[source_rank], covered_by_source.get(source_rank, ())
            )
        )
    return tuple(missing)


def validate_exact_coverage(
    expected: Iterable[ByteRange], actual: Iterable[ByteRange]
) -> None:
    expected_ranges = union_byte_ranges(expected)
    actual_ranges = union_byte_ranges(actual)
    missing = subtract_byte_ranges(expected_ranges, actual_ranges)
    unexpected = subtract_byte_ranges(actual_ranges, expected_ranges)
    if missing or unexpected:
        raise ValueError(
            f"byte coverage differs: missing={missing!r}, unexpected={unexpected!r}"
        )


def _demand_key(demand: FqnDemand) -> tuple[int, int, str]:
    first = min(demand.ranges)
    return (first.source_rank, first.offset, demand.fqn)


def merge_fqn_demands(demands: Iterable[FqnDemand]) -> tuple[FqnDemand, ...]:
    ranges_by_fqn: dict[str, list[ByteRange]] = defaultdict(list)
    for demand in demands:
        ranges_by_fqn[demand.fqn].extend(demand.ranges)
    merged = [
        FqnDemand(fqn=fqn, ranges=tuple(ranges))
        for fqn, ranges in ranges_by_fqn.items()
    ]
    return tuple(sorted(merged, key=_demand_key))


_RawInterval: TypeAlias = tuple[int, int]
_RawIntervalsBySource: TypeAlias = dict[int, list[_RawInterval]]
_RawIntervalsByFqn: TypeAlias = dict[str, _RawIntervalsBySource]


@dataclass(frozen=True, slots=True)
class _RawFqnDemandWireMergeResult:
    ranges_by_fqn: _RawIntervalsByFqn
    input_demand_count: int
    input_range_count: int
    decode_ns: int
    union_ns: int


@dataclass(frozen=True, slots=True)
class _CanonicalFqnDemandWireMergeResult:
    payload: list[object]
    input_demand_count: int
    input_range_count: int
    output_demand_count: int
    output_range_count: int
    decode_ns: int
    union_ns: int
    finalize_ns: int


def merge_fqn_demand_wire_payloads(
    payloads: Iterable[object],
) -> FqnDemandWireMergeResult:
    """Validate and exactly union decoded rank-demand payloads with bounded state."""

    raw_result = _merge_fqn_demand_wire_payloads_to_raw(payloads)
    return _materialize_fqn_demand_wire_merge_result(raw_result)


def _merge_canonical_fqn_demand_wire_payloads(
    payloads: Iterable[object],
) -> FqnDemandWireMergeResult:
    raw_result = _merge_fqn_demand_wire_payloads_to_raw(
        payloads,
        require_canonical=True,
    )
    return _materialize_fqn_demand_wire_merge_result(raw_result)


def _materialize_fqn_demand_wire_merge_result(
    raw_result: _RawFqnDemandWireMergeResult,
) -> FqnDemandWireMergeResult:
    finalize_started_ns = time.perf_counter_ns()
    merged = _materialize_fqn_demands_from_validated_raw(raw_result.ranges_by_fqn)
    finalize_ns = time.perf_counter_ns() - finalize_started_ns
    return FqnDemandWireMergeResult(
        demands=merged,
        input_demand_count=raw_result.input_demand_count,
        input_range_count=raw_result.input_range_count,
        decode_ns=raw_result.decode_ns,
        union_ns=raw_result.union_ns,
        finalize_ns=finalize_ns,
    )


def _merge_fqn_demand_wire_payloads_to_canonical_wire(
    payloads: Iterable[object],
) -> _CanonicalFqnDemandWireMergeResult:
    raw_result = _merge_fqn_demand_wire_payloads_to_raw(payloads)
    return _canonical_fqn_demand_wire_merge_result(raw_result)


def _canonical_fqn_demand_wire_merge_result(
    raw_result: _RawFqnDemandWireMergeResult,
) -> _CanonicalFqnDemandWireMergeResult:
    finalize_started_ns = time.perf_counter_ns()
    payload = _fqn_demands_to_wire_from_validated_raw(raw_result.ranges_by_fqn)
    finalize_ns = time.perf_counter_ns() - finalize_started_ns
    return _CanonicalFqnDemandWireMergeResult(
        payload=payload,
        input_demand_count=raw_result.input_demand_count,
        input_range_count=raw_result.input_range_count,
        output_demand_count=len(raw_result.ranges_by_fqn),
        output_range_count=sum(
            len(intervals)
            for by_source in raw_result.ranges_by_fqn.values()
            for intervals in by_source.values()
        ),
        decode_ns=raw_result.decode_ns,
        union_ns=raw_result.union_ns,
        finalize_ns=finalize_ns,
    )


def _merge_fqn_demand_wire_payloads_to_raw(
    payloads: Iterable[object],
    *,
    require_canonical: bool = False,
) -> _RawFqnDemandWireMergeResult:
    """Validate and union demand payloads without materializing public objects."""

    ranges_by_fqn: _RawIntervalsByFqn = {}
    input_demand_count = 0
    input_range_count = 0
    decode_ns = 0
    union_ns = 0
    merge_payload = (
        _merge_canonical_fqn_demand_wire_payload
        if require_canonical
        else _merge_fqn_demand_wire_payload
    )
    payload_iterator = iter(payloads)
    while True:
        decode_started_ns = time.perf_counter_ns()
        try:
            raw_payload = next(payload_iterator)
        except StopIteration:
            break
        decode_ns += time.perf_counter_ns() - decode_started_ns

        union_started_ns = time.perf_counter_ns()
        demand_count, range_count = merge_payload(
            raw_payload,
            ranges_by_fqn,
        )
        input_demand_count += demand_count
        input_range_count += range_count
        del raw_payload
        union_ns += time.perf_counter_ns() - union_started_ns

    return _RawFqnDemandWireMergeResult(
        ranges_by_fqn=ranges_by_fqn,
        input_demand_count=input_demand_count,
        input_range_count=input_range_count,
        decode_ns=decode_ns,
        union_ns=union_ns,
    )


def _ordered_validated_raw_demands(
    ranges_by_fqn: _RawIntervalsByFqn,
) -> list[tuple[str, _RawIntervalsBySource]]:
    def key(item: tuple[str, _RawIntervalsBySource]) -> tuple[int, int, str]:
        fqn, by_source = item
        first_source_rank = min(by_source)
        return (first_source_rank, by_source[first_source_rank][0][0], fqn)

    return sorted(ranges_by_fqn.items(), key=key)


def _materialize_fqn_demands_from_validated_raw(
    ranges_by_fqn: _RawIntervalsByFqn,
) -> tuple[FqnDemand, ...]:
    """Construct objects from fully validated, canonical wire intervals."""

    demands: list[FqnDemand] = []
    new_instance = object.__new__
    set_attribute = object.__setattr__
    for fqn, by_source in _ordered_validated_raw_demands(ranges_by_fqn):
        ranges: list[ByteRange] = []
        append_range = ranges.append
        for source_rank, intervals in sorted(by_source.items()):
            for offset, end in intervals:
                byte_range = new_instance(ByteRange)
                set_attribute(byte_range, "source_rank", source_rank)
                set_attribute(byte_range, "offset", offset)
                set_attribute(byte_range, "length", end - offset)
                append_range(byte_range)
        demand = new_instance(FqnDemand)
        set_attribute(demand, "fqn", fqn)
        set_attribute(demand, "ranges", tuple(ranges))
        demands.append(demand)
    return tuple(demands)


def _fqn_demands_to_wire_from_validated_raw(
    ranges_by_fqn: _RawIntervalsByFqn,
) -> list[object]:
    return [
        {
            "fqn": fqn,
            "ranges": [
                {
                    "length": end - offset,
                    "offset": offset,
                    "source_rank": source_rank,
                }
                for source_rank, intervals in sorted(by_source.items())
                for offset, end in intervals
            ],
        }
        for fqn, by_source in _ordered_validated_raw_demands(ranges_by_fqn)
    ]


def _merge_canonical_fqn_demand_wire_payload(
    raw_payload: object,
    ranges_by_fqn: _RawIntervalsByFqn,
) -> tuple[int, int]:
    if type(raw_payload) is not list:
        raise ValueError("canonical rank byte demands must be a list")
    input_range_count = 0
    seen_fqns: set[str] = set()
    previous_demand_key: tuple[int, int, str] | None = None
    decode_range = _canonical_byte_range_interval
    for raw_demand in raw_payload:
        fqn, raw_ranges = _canonical_fqn_demand_fields(raw_demand)
        if fqn in seen_fqns:
            raise ValueError("canonical rank byte demands contain a duplicate FQN")
        seen_fqns.add(fqn)

        accumulated_by_source = ranges_by_fqn.setdefault(fqn, {})
        active_source_rank = -1
        active_end = -1
        incoming: list[_RawInterval] = []
        for range_index, raw_range in enumerate(raw_ranges):
            source_rank, offset, end = decode_range(raw_range)
            if range_index == 0:
                demand_key = (source_rank, offset, fqn)
                if (
                    previous_demand_key is not None
                    and demand_key <= previous_demand_key
                ):
                    raise ValueError("canonical rank byte demands are not ordered")
                previous_demand_key = demand_key
            if source_rank < active_source_rank or (
                source_rank == active_source_rank and offset <= active_end
            ):
                raise ValueError("canonical rank byte demand ranges are not ordered")
            if source_rank != active_source_rank:
                if incoming:
                    accumulated_by_source[active_source_rank] = (
                        _merge_canonical_raw_intervals(
                            accumulated_by_source.get(active_source_rank),
                            incoming,
                        )
                    )
                active_source_rank = source_rank
                incoming = []
            incoming.append((offset, end))
            active_end = end
        accumulated_by_source[active_source_rank] = _merge_canonical_raw_intervals(
            accumulated_by_source.get(active_source_rank),
            incoming,
        )
        input_range_count += len(raw_ranges)
    return len(raw_payload), input_range_count


def _canonical_fqn_demand_fields(
    raw_demand: object,
) -> tuple[str, list[object]]:
    if (
        type(raw_demand) is not dict
        or len(raw_demand) != len(_FQN_DEMAND_KEYS)
        or "fqn" not in raw_demand
        or "ranges" not in raw_demand
    ):
        raise ValueError("canonical rank byte demand has invalid fields")
    fqn = raw_demand["fqn"]
    if type(fqn) is not str:
        raise ValueError("canonical rank byte demand FQN must be a string")
    raw_ranges = raw_demand["ranges"]
    if type(raw_ranges) is not list or not raw_ranges:
        raise ValueError("canonical rank byte demand ranges must be a non-empty list")
    return fqn, raw_ranges


def _canonical_byte_range_interval(raw_range: object) -> tuple[int, int, int]:
    if (
        type(raw_range) is not dict
        or len(raw_range) != len(_BYTE_RANGE_KEYS)
        or "source_rank" not in raw_range
        or "offset" not in raw_range
        or "length" not in raw_range
    ):
        raise ValueError("canonical rank byte range has invalid fields")
    source_rank = raw_range["source_rank"]
    offset = raw_range["offset"]
    length = raw_range["length"]
    if type(source_rank) is not int or source_rank < 0:
        raise ValueError("canonical rank byte range source rank must be non-negative")
    if type(offset) is not int or not 0 <= offset <= _UINT64_MAX:
        raise ValueError("canonical rank byte range offset must fit in u64")
    if type(length) is not int or not 1 <= length <= _UINT64_MAX:
        raise ValueError("canonical rank byte range length must fit in u64")
    end = offset + length
    if end > _UINT64_MAX:
        raise ValueError("canonical rank byte range end must fit in u64")
    return source_rank, offset, end


def _merge_fqn_demand_wire_payload(
    raw_payload: object,
    ranges_by_fqn: _RawIntervalsByFqn,
) -> tuple[int, int]:
    if not isinstance(raw_payload, list):
        raise ValueError("rank byte demands must be a list")
    input_range_count = 0
    for raw_demand in raw_payload:
        payload = _require_mapping(raw_demand)
        _require_keys(payload, _FQN_DEMAND_KEYS)
        fqn = _require_fqn("fqn", payload["fqn"])
        raw_ranges = _require_sequence(payload["ranges"], "ranges")
        if not raw_ranges:
            raise ValueError("an FQN demand must contain at least one byte range")

        incoming_by_source: _RawIntervalsBySource = defaultdict(list)
        unordered_sources: set[int] = set()
        for raw_range in raw_ranges:
            source_rank, offset, length = _byte_range_fields_from_dict(raw_range)
            end = offset + length
            incoming = incoming_by_source[source_rank]
            if source_rank in unordered_sources or not incoming:
                incoming.append((offset, end))
                continue
            previous_offset, previous_end = incoming[-1]
            if offset < previous_offset:
                unordered_sources.add(source_rank)
                incoming.append((offset, end))
            elif offset > previous_end:
                incoming.append((offset, end))
            elif end > previous_end:
                incoming[-1] = (previous_offset, end)
        input_range_count += len(raw_ranges)

        accumulated_by_source = ranges_by_fqn.setdefault(fqn, {})
        for source_rank, incoming in incoming_by_source.items():
            if source_rank in unordered_sources:
                _canonicalize_raw_intervals(incoming)
            accumulated_by_source[source_rank] = _merge_canonical_raw_intervals(
                accumulated_by_source.get(source_rank),
                incoming,
            )
    return len(raw_payload), input_range_count


def _canonicalize_raw_intervals(intervals: list[_RawInterval]) -> None:
    intervals.sort()
    write_index = 0
    for read_index in range(len(intervals)):
        offset, end = intervals[read_index]
        if write_index == 0 or offset > intervals[write_index - 1][1]:
            intervals[write_index] = (offset, end)
            write_index += 1
            continue
        previous_offset, previous_end = intervals[write_index - 1]
        if end > previous_end:
            intervals[write_index - 1] = (previous_offset, end)
    del intervals[write_index:]


def _merge_canonical_raw_intervals(
    accumulated: list[_RawInterval] | None,
    incoming: list[_RawInterval],
) -> list[_RawInterval]:
    if not accumulated:
        return incoming
    if accumulated[-1][1] < incoming[0][0]:
        accumulated.extend(incoming)
        return accumulated
    if incoming[-1][1] < accumulated[0][0]:
        incoming.extend(accumulated)
        return incoming
    if (
        accumulated[0][0] <= incoming[0][0]
        and accumulated[-1][1] >= incoming[-1][1]
        and _raw_intervals_cover(accumulated, incoming)
    ):
        return accumulated
    if (
        incoming[0][0] <= accumulated[0][0]
        and incoming[-1][1] >= accumulated[-1][1]
        and _raw_intervals_cover(incoming, accumulated)
    ):
        return incoming

    merged: list[_RawInterval] = []
    accumulated_index = 0
    incoming_index = 0
    while accumulated_index < len(accumulated) or incoming_index < len(incoming):
        if incoming_index >= len(incoming) or (
            accumulated_index < len(accumulated)
            and accumulated[accumulated_index] <= incoming[incoming_index]
        ):
            offset, end = accumulated[accumulated_index]
            accumulated_index += 1
        else:
            offset, end = incoming[incoming_index]
            incoming_index += 1
        if not merged or offset > merged[-1][1]:
            merged.append((offset, end))
            continue
        previous_offset, previous_end = merged[-1]
        if end > previous_end:
            merged[-1] = (previous_offset, end)
    return merged


def _raw_intervals_cover(
    covering: Sequence[_RawInterval],
    required: Sequence[_RawInterval],
) -> bool:
    covering_index = 0
    for required_offset, required_end in required:
        while (
            covering_index < len(covering)
            and covering[covering_index][1] < required_offset
        ):
            covering_index += 1
        if covering_index >= len(covering):
            return False
        covering_offset, covering_end = covering[covering_index]
        if covering_offset > required_offset or covering_end < required_end:
            return False
    return True


def _find(parent: list[int], item: int) -> int:
    root = item
    while parent[root] != root:
        root = parent[root]
    while parent[item] != item:
        next_item = parent[item]
        parent[item] = root
        item = next_item
    return root


def _union(parent: list[int], sizes: list[int], left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root == right_root:
        return
    if sizes[left_root] < sizes[right_root]:
        left_root, right_root = right_root, left_root
    parent[right_root] = left_root
    sizes[left_root] += sizes[right_root]


def _connect_overlapping_demands(demands: Sequence[FqnDemand]) -> list[int]:
    parent = list(range(len(demands)))
    sizes = [1] * len(demands)
    tagged_by_source: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for item_index, demand in enumerate(demands):
        for byte_range in demand.ranges:
            tagged_by_source[byte_range.source_rank].append(
                (byte_range.offset, byte_range.length, item_index)
            )

    # Preserve the prior ``(ByteRange, index)`` ordering exactly: union-by-size
    # tie-breaking makes even independent source scans order-sensitive.
    for source_rank in sorted(tagged_by_source):
        tagged = tagged_by_source[source_rank]
        tagged.sort()
        active_end = -1
        active_index = -1
        for offset, length, item_index in tagged:
            end = offset + length
            if offset >= active_end:
                active_end = end
                active_index = item_index
                continue
            _union(parent, sizes, active_index, item_index)
            if end > active_end:
                active_end = end
                active_index = item_index
    return parent


def overlap_connected_fqn_groups(
    demands: Iterable[FqnDemand],
) -> tuple[tuple[FqnDemand, ...], ...]:
    merged = merge_fqn_demands(demands)
    return _overlap_connected_fqn_groups_from_merged(merged)


def _overlap_connected_fqn_groups_from_merged(
    merged: Sequence[FqnDemand],
) -> tuple[tuple[FqnDemand, ...], ...]:
    parent = _connect_overlapping_demands(merged)
    by_root: dict[int, list[FqnDemand]] = defaultdict(list)
    for index, demand in enumerate(merged):
        by_root[_find(parent, index)].append(demand)
    components = [
        tuple(sorted(component, key=_demand_key)) for component in by_root.values()
    ]
    return tuple(sorted(components, key=lambda component: _demand_key(component[0])))


def _demanded_source_loads(
    demands: Iterable[FqnDemand], gap_bytes: int
) -> dict[int, int]:
    ranges = (byte_range for demand in demands for byte_range in demand.ranges)
    consolidated = consolidate_byte_ranges(ranges, gap_bytes)
    loads: Counter[int] = Counter()
    for byte_range in consolidated:
        loads[byte_range.source_rank] += byte_range.length
    return dict(loads)


def _validated_source_consumer_bytes_by_node(
    source_consumer_bytes_by_node: SourceConsumerBytesByNode,
    node_ids: tuple[NodeId, ...],
    source_ranks: frozenset[int],
) -> tuple[Mapping[int, int], ...]:
    observed_node_ids = {
        _require_node_id("node_id", node_id)
        for node_id in source_consumer_bytes_by_node
    }
    if observed_node_ids != set(node_ids):
        raise ValueError(
            "source consumer bytes must contain exactly the assignment nodes"
        )
    result: list[Mapping[int, int]] = []
    for node_id in node_ids:
        raw_source_bytes = source_consumer_bytes_by_node[node_id]
        if not isinstance(raw_source_bytes, Mapping):
            raise ValueError("source consumer bytes for each node must be a mapping")
        source_bytes: dict[int, int] = {}
        for source_rank, byte_count in raw_source_bytes.items():
            rank = _require_int("source_rank", source_rank)
            if rank not in source_ranks:
                raise ValueError(
                    f"source consumer bytes contain unknown source rank {rank}"
                )
            source_bytes[rank] = _require_u64("consumer_bytes", byte_count)
        result.append(MappingProxyType(source_bytes))
    return tuple(result)


def _build_source_assignment(
    *,
    nodes: tuple[NodeId, ...],
    ordered_source_ranks: Sequence[int],
    source_loads: Mapping[int, int],
    source_consumer_bytes: Sequence[Mapping[int, int]] | None = None,
) -> FileAssignment:
    node_sources: list[list[int]] = [[] for _ in nodes]
    node_bytes = [0] * len(nodes)
    for source_rank in ordered_source_ranks:
        minimum_bytes = min(node_bytes)
        candidates = [
            index
            for index, assigned_bytes in enumerate(node_bytes)
            if assigned_bytes == minimum_bytes
        ]
        if source_consumer_bytes is None:
            node_index = candidates[0]
        else:
            node_index = min(
                candidates,
                key=lambda index: (
                    -source_consumer_bytes[index].get(source_rank, 0),
                    index,
                ),
            )
        node_sources[node_index].append(source_rank)
        node_bytes[node_index] += source_loads[source_rank]
    return FileAssignment(
        node_ids=nodes,
        node_source_ranks=tuple(tuple(ranks) for ranks in node_sources),
        node_bytes=tuple(node_bytes),
    )


def assignment_locality_stats(
    assignment: FileAssignment,
    source_consumer_bytes_by_node: SourceConsumerBytesByNode,
) -> AssignmentLocalityStats:
    """Return predicted local and remote consumer bytes for an assignment."""

    source_ranks = frozenset(assignment.source_rank_to_node)
    source_consumer_bytes = _validated_source_consumer_bytes_by_node(
        source_consumer_bytes_by_node,
        assignment.node_ids,
        source_ranks,
    )
    return _assignment_locality_stats(assignment, source_consumer_bytes)


def _assignment_locality_stats(
    assignment: FileAssignment,
    source_consumer_bytes: Sequence[Mapping[int, int]],
) -> AssignmentLocalityStats:
    node_total_bytes = tuple(
        sum(source_bytes.values()) for source_bytes in source_consumer_bytes
    )
    node_local_bytes = tuple(
        sum(
            byte_count
            for source_rank, byte_count in source_bytes.items()
            if assignment.owner_for(source_rank) == node_id
        )
        for node_id, source_bytes in zip(
            assignment.node_ids,
            source_consumer_bytes,
            strict=True,
        )
    )
    node_remote_bytes = tuple(
        total - local
        for total, local in zip(node_total_bytes, node_local_bytes, strict=True)
    )
    total_bytes = sum(node_total_bytes)
    local_bytes = sum(node_local_bytes)
    return AssignmentLocalityStats(
        node_ids=assignment.node_ids,
        node_total_consumer_bytes=node_total_bytes,
        node_local_consumer_bytes=node_local_bytes,
        node_remote_consumer_bytes=node_remote_bytes,
        total_consumer_bytes=total_bytes,
        local_consumer_bytes=local_bytes,
        remote_consumer_bytes=total_bytes - local_bytes,
    )


def _source_assignment_inputs(
    demands: Iterable[FqnDemand],
    node_ids: Sequence[NodeId],
    consolidate_gap_bytes: int,
) -> tuple[
    tuple[NodeId, ...],
    int,
    tuple[FqnDemand, ...],
    Mapping[int, int],
    tuple[int, ...],
]:
    nodes = _canonical_node_ids(
        _require_node_id("node_id", node_id) for node_id in node_ids
    )
    gap = _require_u64("consolidate_gap_bytes", consolidate_gap_bytes)
    merged = merge_fqn_demands(demands)
    source_loads = _demanded_source_loads(merged, gap)
    source_range_counts: Counter[int] = Counter()
    all_ranges = consolidate_byte_ranges(
        (byte_range for demand in merged for byte_range in demand.ranges), gap
    )
    for byte_range in all_ranges:
        source_range_counts[byte_range.source_rank] += 1
    ordered = sorted(
        source_loads,
        key=lambda rank: (-source_loads[rank], -source_range_counts[rank], rank),
    )
    return nodes, gap, merged, source_loads, tuple(ordered)


def _canonical_raw_intervals_with_gap(
    intervals: Iterable[_RawInterval],
    gap_bytes: int,
) -> tuple[_RawInterval, ...]:
    ordered = sorted(intervals)
    consolidated: list[_RawInterval] = []
    for offset, end in ordered:
        if not consolidated or offset > consolidated[-1][1] + gap_bytes:
            consolidated.append((offset, end))
            continue
        previous_offset, previous_end = consolidated[-1]
        if end > previous_end:
            consolidated[-1] = (previous_offset, end)
    return tuple(consolidated)


def _merge_consolidated_raw_intervals(
    left: Sequence[_RawInterval],
    right: Sequence[_RawInterval],
    gap_bytes: int,
) -> tuple[_RawInterval, ...]:
    if not left:
        return tuple(right)
    if not right:
        return tuple(left)
    if left[-1][1] + gap_bytes < right[0][0]:
        return (*left, *right)
    if right[-1][1] + gap_bytes < left[0][0]:
        return (*right, *left)

    merged: list[_RawInterval] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) or right_index < len(right):
        if right_index >= len(right) or (
            left_index < len(left) and left[left_index] <= right[right_index]
        ):
            offset, end = left[left_index]
            left_index += 1
        else:
            offset, end = right[right_index]
            right_index += 1
        if not merged or offset > merged[-1][1] + gap_bytes:
            merged.append((offset, end))
            continue
        previous_offset, previous_end = merged[-1]
        if end > previous_end:
            merged[-1] = (previous_offset, end)
    return tuple(merged)


def _insert_consolidated_raw_interval(
    intervals: Sequence[_RawInterval],
    interval_bytes: int,
    incoming: _RawInterval,
    gap_bytes: int,
) -> tuple[tuple[_RawInterval, ...], int]:
    if not intervals:
        return (incoming,), incoming[1] - incoming[0]
    offset, end = incoming
    insertion = bisect_left(intervals, (offset, -1))
    first = insertion
    if insertion > 0 and offset <= intervals[insertion - 1][1] + gap_bytes:
        first -= 1
        offset = intervals[first][0]
        end = max(end, intervals[first][1])
    last = insertion
    while last < len(intervals) and intervals[last][0] <= end + gap_bytes:
        end = max(end, intervals[last][1])
        last += 1
    if first == insertion and last == insertion:
        return (
            (*intervals[:insertion], incoming, *intervals[insertion:]),
            interval_bytes + incoming[1] - incoming[0],
        )
    if (
        first == insertion - 1
        and last == insertion
        and (offset, end) == intervals[first]
    ):
        return tuple(intervals), interval_bytes
    replaced_bytes = _raw_interval_bytes(intervals[first:last])
    return (
        (*intervals[:first], (offset, end), *intervals[last:]),
        interval_bytes - replaced_bytes + end - offset,
    )


def _raw_interval_bytes(intervals: Iterable[_RawInterval]) -> int:
    return sum(end - offset for offset, end in intervals)


class _MutableConsolidatedIntervals:
    __slots__ = ("_bytes_by_source", "_gap_bytes", "_ranges_by_source", "total_bytes")

    def __init__(
        self,
        ranges_by_source: Mapping[int, Sequence[_RawInterval]],
        gap_bytes: int,
    ) -> None:
        self._gap_bytes = gap_bytes
        self._ranges_by_source = {
            source_rank: tuple(intervals)
            for source_rank, intervals in ranges_by_source.items()
            if intervals
        }
        self._bytes_by_source = {
            source_rank: _raw_interval_bytes(intervals)
            for source_rank, intervals in self._ranges_by_source.items()
        }
        self.total_bytes = sum(self._bytes_by_source.values())

    def preview_add(
        self,
        incoming_by_source: Mapping[int, Sequence[_RawInterval]],
        budget_bytes: int,
    ) -> (
        tuple[
            int,
            tuple[tuple[int, tuple[_RawInterval, ...], int], ...],
        ]
        | None
    ):
        if budget_bytes > 0 and self.total_bytes > budget_bytes:
            return None
        total_bytes = self.total_bytes
        updates: list[tuple[int, tuple[_RawInterval, ...], int]] = []
        for source_rank, incoming in incoming_by_source.items():
            current = self._ranges_by_source.get(source_rank, ())
            current_bytes = self._bytes_by_source.get(source_rank, 0)
            if len(incoming) == 1:
                merged, merged_bytes = _insert_consolidated_raw_interval(
                    current,
                    current_bytes,
                    incoming[0],
                    self._gap_bytes,
                )
            else:
                merged = _merge_consolidated_raw_intervals(
                    current,
                    incoming,
                    self._gap_bytes,
                )
                merged_bytes = _raw_interval_bytes(merged)
            total_bytes += merged_bytes - current_bytes
            if budget_bytes > 0 and total_bytes > budget_bytes:
                return None
            updates.append((source_rank, merged, merged_bytes))
        return total_bytes, tuple(updates)

    def commit(
        self,
        preview: tuple[
            int,
            tuple[tuple[int, tuple[_RawInterval, ...], int], ...],
        ],
    ) -> None:
        self.total_bytes = preview[0]
        for source_rank, merged, merged_bytes in preview[1]:
            self._ranges_by_source[source_rank] = merged
            self._bytes_by_source[source_rank] = merged_bytes

    def to_byte_ranges(self) -> tuple[ByteRange, ...]:
        return tuple(
            ByteRange(source_rank, offset, end - offset)
            for source_rank in sorted(self._ranges_by_source)
            for offset, end in self._ranges_by_source[source_rank]
        )


class _DisjointExactIntervals:
    """Exact fragments from distinct, non-overlapping demand components."""

    __slots__ = ("_ranges",)

    def __init__(
        self,
        ranges: Sequence[ByteRange],
    ) -> None:
        self._ranges = list(ranges)

    def add(
        self,
        ranges: Sequence[ByteRange],
    ) -> None:
        self._ranges.extend(ranges)

    def to_byte_ranges(self) -> tuple[ByteRange, ...]:
        self._ranges.sort(
            key=lambda byte_range: (
                byte_range.source_rank,
                byte_range.offset,
                byte_range.length,
            )
        )
        result: list[ByteRange] = []
        for byte_range in self._ranges:
            if (
                not result
                or byte_range.source_rank != result[-1].source_rank
                or byte_range.offset > result[-1].end
            ):
                result.append(byte_range)
                continue
            previous = result[-1]
            result[-1] = ByteRange(
                previous.source_rank,
                previous.offset,
                max(previous.end, byte_range.end) - previous.offset,
            )
        return tuple(result)


class _FusedBatchState:
    """Linear-size batch state retaining exact intervals for final work emission.

    The nonzero-gap path keeps one additional linear list of references to the
    canonical ``ByteRange`` objects; the zero-gap path reuses download state.
    """

    __slots__ = ("demands", "download_by_node", "exact_by_node")

    def __init__(
        self,
        component: Sequence[FqnDemand],
        exact_by_node: Sequence[Sequence[ByteRange]],
        download_by_node: Sequence[Mapping[int, Sequence[_RawInterval]]],
        gap_bytes: int,
    ) -> None:
        self.demands = list(component)
        self.download_by_node = tuple(
            _MutableConsolidatedIntervals(node_ranges, gap_bytes)
            for node_ranges in download_by_node
        )
        self.exact_by_node = (
            None
            if gap_bytes == 0
            else tuple(
                _DisjointExactIntervals(node_ranges) for node_ranges in exact_by_node
            )
        )

    def try_add(
        self,
        component: Sequence[FqnDemand],
        exact_by_node: Sequence[Sequence[ByteRange]],
        download_by_node: Sequence[Mapping[int, Sequence[_RawInterval]]],
        budget_bytes: int,
    ) -> bool:
        previews: list[
            tuple[int, tuple[tuple[int, tuple[_RawInterval, ...], int], ...]]
        ] = []
        for accumulated, incoming in zip(
            self.download_by_node,
            download_by_node,
            strict=True,
        ):
            preview = accumulated.preview_add(incoming, budget_bytes)
            if preview is None:
                return False
            previews.append(preview)
        for accumulated, preview in zip(
            self.download_by_node,
            previews,
            strict=True,
        ):
            accumulated.commit(preview)
        if self.exact_by_node is not None:
            for accumulated, incoming in zip(
                self.exact_by_node,
                exact_by_node,
                strict=True,
            ):
                accumulated.add(incoming)
        self.demands.extend(component)
        return True


def _source_loads_from_merged_demands(
    merged_demands: Sequence[FqnDemand],
    gap_bytes: int,
) -> tuple[dict[int, int], dict[int, int]]:
    intervals_by_source: dict[int, list[_RawInterval]] = defaultdict(list)
    for demand in merged_demands:
        for byte_range in demand.ranges:
            intervals_by_source[byte_range.source_rank].append(
                (byte_range.offset, byte_range.end)
            )
    source_loads: dict[int, int] = {}
    source_range_counts: dict[int, int] = {}
    for source_rank in sorted(intervals_by_source):
        consolidated = _canonical_raw_intervals_with_gap(
            intervals_by_source[source_rank],
            gap_bytes,
        )
        source_loads[source_rank] = _raw_interval_bytes(consolidated)
        source_range_counts[source_rank] = len(consolidated)
    return source_loads, source_range_counts


def _component_intervals_by_node(
    component: Sequence[FqnDemand],
    source_node_indices: Mapping[int, int],
    node_count: int,
    gap_bytes: int,
) -> tuple[
    tuple[tuple[ByteRange, ...], ...],
    tuple[Mapping[int, Sequence[_RawInterval]], ...],
]:
    ranges_by_node: list[list[ByteRange]] = [[] for _ in range(node_count)]
    for demand in component:
        for byte_range in demand.ranges:
            ranges_by_node[source_node_indices[byte_range.source_rank]].append(
                byte_range
            )

    exact_by_node: list[tuple[ByteRange, ...]] = []
    download_by_node: list[Mapping[int, Sequence[_RawInterval]]] = []
    for node_ranges in ranges_by_node:
        exact_ranges = (
            tuple(node_ranges)
            if len(component) == 1
            else union_byte_ranges(node_ranges)
        )
        raw_exact_by_source: dict[int, list[_RawInterval]] = defaultdict(list)
        for byte_range in exact_ranges:
            raw_exact_by_source[byte_range.source_rank].append(
                (byte_range.offset, byte_range.end)
            )
        download_ranges: dict[int, tuple[_RawInterval, ...]] = {}
        for source_rank, exact in raw_exact_by_source.items():
            download_ranges[source_rank] = (
                tuple(exact)
                if gap_bytes == 0
                else _canonical_raw_intervals_with_gap(exact, gap_bytes)
            )
        exact_by_node.append(exact_ranges)
        download_by_node.append(download_ranges)
    return tuple(exact_by_node), tuple(download_by_node)


def _build_fused_batch_node_works(
    batch_states: Sequence[_FusedBatchState],
    node_ids: Sequence[NodeId],
    gap_bytes: int,
) -> tuple[BatchNodeWork, ...]:
    exact_works: list[BatchNodeWork] = []
    for batch_index, batch_state in enumerate(batch_states):
        fqn_names = tuple(sorted(demand.fqn for demand in batch_state.demands))
        exact_by_node = (
            batch_state.download_by_node
            if batch_state.exact_by_node is None
            else batch_state.exact_by_node
        )
        for node_id, exact_ranges in zip(
            node_ids,
            exact_by_node,
            strict=True,
        ):
            canonical_exact_ranges = exact_ranges.to_byte_ranges()
            exact_works.append(
                BatchNodeWork._from_canonical(
                    batch_index=batch_index,
                    node_id=node_id,
                    fqn_names=fqn_names,
                    exact_ranges=canonical_exact_ranges,
                    download_ranges=canonical_exact_ranges,
                )
            )
    return _build_disjoint_download_works_from_canonical(exact_works, gap_bytes)


def _build_disjoint_download_works_from_canonical(
    works: Sequence[BatchNodeWork],
    gap_bytes: int,
) -> tuple[BatchNodeWork, ...]:
    """Consolidate exact ranges whose cross-batch coverage is already disjoint."""

    if gap_bytes == 0:
        return tuple(works)
    tagged_by_source: dict[
        int,
        list[tuple[int, int, int]],
    ] = defaultdict(list)
    for work_index, work in enumerate(works):
        for byte_range in work.exact_ranges:
            tagged_by_source[byte_range.source_rank].append(
                (
                    byte_range.offset,
                    byte_range.end,
                    work_index,
                )
            )

    raw_downloads_by_work: list[list[tuple[int, int, int]]] = [[] for _ in works]
    for source_rank in sorted(tagged_by_source):
        tagged = tagged_by_source[source_rank]
        tagged.sort()
        previous_work_index = -1
        for offset, end, work_index in tagged:
            downloads = raw_downloads_by_work[work_index]
            if (
                previous_work_index == work_index
                and offset <= downloads[-1][2] + gap_bytes
            ):
                previous_source, previous_offset, previous_end = downloads[-1]
                downloads[-1] = (
                    previous_source,
                    previous_offset,
                    max(previous_end, end),
                )
            else:
                downloads.append((source_rank, offset, end))
            previous_work_index = work_index

    return tuple(
        BatchNodeWork._from_canonical(
            batch_index=work.batch_index,
            node_id=work.node_id,
            fqn_names=work.fqn_names,
            exact_ranges=work.exact_ranges,
            download_ranges=tuple(
                ByteRange(source_rank, offset, end - offset)
                for source_rank, offset, end in raw_downloads
            ),
        )
        for work, raw_downloads in zip(
            works,
            raw_downloads_by_work,
            strict=True,
        )
    )


def _rebuild_batch_node_works_from_canonical(
    works: Sequence[BatchNodeWork],
    consolidate_gap_bytes: int,
) -> tuple[BatchNodeWork, ...]:
    """Rebuild download ranges while retaining trusted canonical exact ranges."""

    gap = _require_u64("consolidate_gap_bytes", consolidate_gap_bytes)
    if gap == 0:
        return tuple(
            BatchNodeWork._from_canonical(
                batch_index=work.batch_index,
                node_id=work.node_id,
                fqn_names=work.fqn_names,
                exact_ranges=work.exact_ranges,
                download_ranges=work.exact_ranges,
            )
            for work in works
        )
    blockers: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for work in works:
        for byte_range in work.exact_ranges:
            blockers[byte_range.source_rank].append(
                (byte_range.offset, byte_range.end, work.batch_index)
            )
    for items in blockers.values():
        items.sort()
    blocker_starts = {
        source_rank: tuple(item[0] for item in items)
        for source_rank, items in blockers.items()
    }

    return tuple(
        BatchNodeWork._from_canonical(
            batch_index=work.batch_index,
            node_id=work.node_id,
            fqn_names=work.fqn_names,
            exact_ranges=work.exact_ranges,
            download_ranges=_consolidate_without_foreign_ranges(
                work.exact_ranges,
                work.batch_index,
                blockers,
                blocker_starts,
                gap,
            ),
        )
        for work in works
    )


def _plan_cooperative_resharding_from_merged_demands(
    merged_demands: tuple[FqnDemand, ...],
    node_ids: Sequence[NodeId],
    batch_budget_bytes: int,
    consolidate_gap_bytes: int,
    *,
    source_consumer_bytes_by_node: SourceConsumerBytesByNode,
) -> _CooperativePlanningResult:
    """Build the cooperative plan from canonical merged demand records."""

    budget = _require_u64("batch_budget_bytes", batch_budget_bytes)
    gap = _require_u64("consolidate_gap_bytes", consolidate_gap_bytes)
    assignment_started_ns = time.perf_counter_ns()
    nodes = _canonical_node_ids(node_ids)
    source_loads, source_range_counts = _source_loads_from_merged_demands(
        merged_demands,
        gap,
    )
    ordered_source_ranks = tuple(
        sorted(
            source_loads,
            key=lambda rank: (
                -source_loads[rank],
                -source_range_counts[rank],
                rank,
            ),
        )
    )
    baseline = _build_source_assignment(
        nodes=nodes,
        ordered_source_ranks=ordered_source_ranks,
        source_loads=source_loads,
    )
    source_consumer_bytes = _validated_source_consumer_bytes_by_node(
        source_consumer_bytes_by_node,
        nodes,
        frozenset(source_loads),
    )
    affinity_assignment = _build_source_assignment(
        nodes=nodes,
        ordered_source_ranks=ordered_source_ranks,
        source_loads=source_loads,
        source_consumer_bytes=source_consumer_bytes,
    )
    baseline_locality = _assignment_locality_stats(baseline, source_consumer_bytes)
    affinity_locality = _assignment_locality_stats(
        affinity_assignment,
        source_consumer_bytes,
    )
    if affinity_locality.local_consumer_bytes > baseline_locality.local_consumer_bytes:
        assignment = affinity_assignment
        chosen_locality = affinity_locality
    else:
        assignment = baseline
        chosen_locality = baseline_locality
    assignment_result = SourceAssignmentResult(
        assignment=assignment,
        baseline_locality=baseline_locality,
        chosen_locality=chosen_locality,
        theoretical_max_local_consumer_bytes=sum(
            max(
                source_bytes.get(source_rank, 0)
                for source_bytes in source_consumer_bytes
            )
            for source_rank in ordered_source_ranks
        ),
    )
    assignment_ns = time.perf_counter_ns() - assignment_started_ns

    batching_started_ns = time.perf_counter_ns()
    source_node_indices = {
        source_rank: node_index
        for node_index, source_ranks in enumerate(assignment.node_source_ranks)
        for source_rank in source_ranks
    }
    batch_states: list[_FusedBatchState] = []
    for component in _overlap_connected_fqn_groups_from_merged(merged_demands):
        exact_by_node, download_by_node = _component_intervals_by_node(
            component,
            source_node_indices,
            assignment.num_nodes,
            gap,
        )
        component_is_oversized = budget > 0 and any(
            sum(_raw_interval_bytes(ranges) for ranges in node_ranges.values()) > budget
            for node_ranges in download_by_node
        )
        if not component_is_oversized:
            for batch in batch_states:
                if batch.try_add(
                    component,
                    exact_by_node,
                    download_by_node,
                    budget,
                ):
                    break
            else:
                batch_states.append(
                    _FusedBatchState(
                        component,
                        exact_by_node,
                        download_by_node,
                        gap,
                    )
                )
            continue
        batch_states.append(
            _FusedBatchState(
                component,
                exact_by_node,
                download_by_node,
                gap,
            )
        )

    batches = tuple(
        PipelineBatch(batch_index, tuple(batch.demands))
        for batch_index, batch in enumerate(batch_states)
    )
    batching_ns = time.perf_counter_ns() - batching_started_ns
    work_started_ns = time.perf_counter_ns()
    works = _build_fused_batch_node_works(
        batch_states,
        assignment.node_ids,
        gap,
    )
    work_ns = time.perf_counter_ns() - work_started_ns
    return _CooperativePlanningResult(
        assignment_result=assignment_result,
        batches=batches,
        works=works,
        assignment_ns=assignment_ns,
        batching_ns=batching_ns,
        work_ns=work_ns,
    )


def assign_sources_to_nodes_with_stats(
    demands: Iterable[FqnDemand],
    node_ids: Sequence[NodeId],
    consolidate_gap_bytes: int = 0,
    *,
    source_consumer_bytes_by_node: SourceConsumerBytesByNode,
) -> SourceAssignmentResult:
    """Assign sources with locality preference and report its traffic bounds."""

    nodes, gap, merged, source_loads, ordered = _source_assignment_inputs(
        demands,
        node_ids,
        consolidate_gap_bytes,
    )
    baseline = _build_source_assignment(
        nodes=nodes,
        ordered_source_ranks=ordered,
        source_loads=source_loads,
    )
    source_consumer_bytes = _validated_source_consumer_bytes_by_node(
        source_consumer_bytes_by_node,
        nodes,
        frozenset(source_loads),
    )
    affinity_assignment = _build_source_assignment(
        nodes=nodes,
        ordered_source_ranks=ordered,
        source_loads=source_loads,
        source_consumer_bytes=source_consumer_bytes,
    )
    baseline_locality = _assignment_locality_stats(
        baseline,
        source_consumer_bytes,
    )
    affinity_locality = _assignment_locality_stats(
        affinity_assignment,
        source_consumer_bytes,
    )
    if affinity_locality.local_consumer_bytes > baseline_locality.local_consumer_bytes:
        assignment = affinity_assignment
        chosen_locality = affinity_locality
    else:
        assignment = baseline
        chosen_locality = baseline_locality
    validate_assignment_coverage(merged, assignment, gap)
    theoretical_max_local_bytes = sum(
        max(source_bytes.get(source_rank, 0) for source_bytes in source_consumer_bytes)
        for source_rank in ordered
    )
    return SourceAssignmentResult(
        assignment=assignment,
        baseline_locality=baseline_locality,
        chosen_locality=chosen_locality,
        theoretical_max_local_consumer_bytes=theoretical_max_local_bytes,
    )


def assign_sources_to_nodes(
    demands: Iterable[FqnDemand],
    node_ids: Sequence[NodeId],
    consolidate_gap_bytes: int = 0,
    *,
    source_consumer_bytes_by_node: SourceConsumerBytesByNode | None = None,
) -> FileAssignment:
    if source_consumer_bytes_by_node is not None:
        return assign_sources_to_nodes_with_stats(
            demands,
            node_ids,
            consolidate_gap_bytes,
            source_consumer_bytes_by_node=source_consumer_bytes_by_node,
        ).assignment
    nodes, gap, merged, source_loads, ordered = _source_assignment_inputs(
        demands,
        node_ids,
        consolidate_gap_bytes,
    )
    assignment = _build_source_assignment(
        nodes=nodes,
        ordered_source_ranks=ordered,
        source_loads=source_loads,
    )
    validate_assignment_coverage(merged, assignment, gap)
    return assignment


def validate_assignment_coverage(
    demands: Iterable[FqnDemand],
    assignment: FileAssignment,
    consolidate_gap_bytes: int = 0,
) -> None:
    expected_loads = _demanded_source_loads(
        merge_fqn_demands(demands), consolidate_gap_bytes
    )
    if set(expected_loads) != set(assignment.source_rank_to_node):
        raise ValueError(
            "file assignment does not cover exactly the demanded source ranks"
        )
    observed_loads = [0] * assignment.num_nodes
    for source_rank, byte_count in expected_loads.items():
        node_index = assignment.node_index_for(assignment.owner_for(source_rank))
        observed_loads[node_index] += byte_count
    if tuple(observed_loads) != assignment.node_bytes:
        raise ValueError(
            f"file assignment byte totals differ: expected={observed_loads!r}, "
            f"actual={assignment.node_bytes!r}"
        )


def plan_pipeline_batches(
    demands: Iterable[FqnDemand],
    batch_budget_bytes: int,
    assignment: FileAssignment | None = None,
    consolidate_gap_bytes: int = 0,
) -> tuple[PipelineBatch, ...]:
    budget = _require_u64("batch_budget_bytes", batch_budget_bytes)
    gap = _require_u64("consolidate_gap_bytes", consolidate_gap_bytes)
    merged = merge_fqn_demands(demands)
    if assignment is not None:
        demanded_sources = {
            byte_range.source_rank for demand in merged for byte_range in demand.ranges
        }
        if demanded_sources != set(assignment.source_rank_to_node):
            raise ValueError(
                "file assignment does not cover exactly the demanded source ranks"
            )
    batch_demands: list[list[FqnDemand]] = []
    batch_node_ranges: list[tuple[tuple[ByteRange, ...], ...]] = []
    for component in overlap_connected_fqn_groups(merged):
        component_node_ranges = _component_node_download_ranges(
            component,
            assignment,
            gap,
        )
        component_is_oversized = budget > 0 and any(
            _range_bytes(ranges) > budget for ranges in component_node_ranges
        )
        if component_is_oversized:
            batch_demands.append(list(component))
            batch_node_ranges.append(component_node_ranges)
            continue
        placement = _first_batch_with_capacity(
            batch_node_ranges,
            component_node_ranges,
            budget,
            gap,
        )
        if placement is None:
            batch_demands.append(list(component))
            batch_node_ranges.append(component_node_ranges)
            continue
        batch_index, combined_node_ranges = placement
        batch_demands[batch_index].extend(component)
        batch_node_ranges[batch_index] = combined_node_ranges
    batches = tuple(
        PipelineBatch(batch_index, tuple(batch_demands_for_index))
        for batch_index, batch_demands_for_index in enumerate(batch_demands)
    )
    validate_batch_coverage(merged, batches)
    return batches


def _component_node_download_ranges(
    component: Sequence[FqnDemand],
    assignment: FileAssignment | None,
    gap_bytes: int,
) -> tuple[tuple[ByteRange, ...], ...]:
    ranges_by_node: list[list[ByteRange]] = [
        [] for _ in range(1 if assignment is None else assignment.num_nodes)
    ]
    for demand in component:
        for byte_range in demand.ranges:
            node_index = (
                0
                if assignment is None
                else assignment.node_index_for(
                    assignment.owner_for(byte_range.source_rank)
                )
            )
            ranges_by_node[node_index].append(byte_range)
    return tuple(
        consolidate_byte_ranges(node_ranges, gap_bytes)
        for node_ranges in ranges_by_node
    )


def _range_bytes(ranges: Sequence[ByteRange]) -> int:
    return sum(byte_range.length for byte_range in ranges)


def _merge_consolidated_byte_ranges(
    left: Sequence[ByteRange],
    right: Sequence[ByteRange],
    gap_bytes: int,
) -> tuple[ByteRange, ...]:
    combined: list[ByteRange] = []
    for byte_range in merge_sorted(left, right):
        if not combined or byte_range.source_rank != combined[-1].source_rank:
            combined.append(byte_range)
            continue
        previous = combined[-1]
        if byte_range.offset > previous.end + gap_bytes:
            combined.append(byte_range)
            continue
        combined[-1] = ByteRange(
            source_rank=previous.source_rank,
            offset=previous.offset,
            length=max(previous.end, byte_range.end) - previous.offset,
        )
    return tuple(combined)


def _merge_node_download_ranges(
    left: Sequence[Sequence[ByteRange]],
    right: Sequence[Sequence[ByteRange]],
    gap_bytes: int,
) -> tuple[tuple[ByteRange, ...], ...]:
    if len(left) != len(right):
        raise ValueError("node range arrays must have matching lengths")
    return tuple(
        _merge_consolidated_byte_ranges(left_ranges, right_ranges, gap_bytes)
        for left_ranges, right_ranges in zip(left, right, strict=True)
    )


def _first_batch_with_capacity(
    batch_node_ranges: Sequence[Sequence[Sequence[ByteRange]]],
    component_node_ranges: Sequence[Sequence[ByteRange]],
    budget: int,
    gap_bytes: int,
) -> tuple[int, tuple[tuple[ByteRange, ...], ...]] | None:
    for batch_index, node_ranges in enumerate(batch_node_ranges):
        combined = _merge_node_download_ranges(
            node_ranges,
            component_node_ranges,
            gap_bytes,
        )
        if budget == 0 or all(_range_bytes(ranges) <= budget for ranges in combined):
            return batch_index, combined
    return None


def validate_batch_coverage(
    demands: Iterable[FqnDemand],
    batches: Iterable[PipelineBatch],
) -> None:
    expected = merge_fqn_demands(demands)
    planned = tuple(batches)
    if tuple(batch.batch_index for batch in planned) != tuple(range(len(planned))):
        raise ValueError("pipeline batch indices must be contiguous and zero-based")
    by_fqn: dict[str, tuple[int, FqnDemand]] = {}
    for batch in planned:
        for demand in batch.demands:
            if demand.fqn in by_fqn:
                raise ValueError(f"FQN {demand.fqn!r} occurs in multiple batches")
            by_fqn[demand.fqn] = (batch.batch_index, demand)
    if {item.fqn: item for item in expected} != {
        fqn: item for fqn, (_, item) in by_fqn.items()
    }:
        raise ValueError("pipeline batches do not exactly cover the FQN demands")
    for component in overlap_connected_fqn_groups(expected):
        component_batches = {by_fqn[demand.fqn][0] for demand in component}
        if len(component_batches) != 1:
            raise ValueError("overlap-connected FQNs must stay in one batch")


def build_batch_node_work(
    batch: PipelineBatch,
    assignment: FileAssignment,
    node_id: NodeId,
    consolidate_gap_bytes: int,
) -> BatchNodeWork:
    node_index = assignment.node_index_for(_require_node_id("node_id", node_id))
    owner = assignment.node_ids[node_index]
    gap = _require_u64("consolidate_gap_bytes", consolidate_gap_bytes)
    exact = union_byte_ranges(
        byte_range
        for demand in batch.demands
        for byte_range in demand.ranges
        if assignment.owner_for(byte_range.source_rank) == owner
    )
    return BatchNodeWork(
        batch_index=batch.batch_index,
        node_id=owner,
        fqn_names=tuple(demand.fqn for demand in batch.demands),
        exact_ranges=exact,
        download_ranges=consolidate_byte_ranges(exact, gap),
    )


def build_batch_node_works(
    batches: Iterable[PipelineBatch],
    assignment: FileAssignment,
    consolidate_gap_bytes: int,
) -> tuple[BatchNodeWork, ...]:
    planned = tuple(batches)
    if tuple(batch.batch_index for batch in planned) != tuple(range(len(planned))):
        raise ValueError("pipeline batch indices must be contiguous and zero-based")
    exact_works = tuple(
        work
        for batch in planned
        for work in _build_exact_batch_node_works(batch, assignment)
    )
    works = _build_disjoint_download_works(exact_works, consolidate_gap_bytes)
    expected = (
        byte_range
        for batch in planned
        for demand in batch.demands
        for byte_range in demand.ranges
    )
    actual = (item for work in works for item in work.exact_ranges)
    validate_exact_coverage(expected, actual)
    _validate_disjoint_batch_downloads(works)
    return works


def _build_exact_batch_node_works(
    batch: PipelineBatch,
    assignment: FileAssignment,
) -> tuple[BatchNodeWork, ...]:
    ranges_by_node: dict[NodeId, list[ByteRange]] = {
        node_id: [] for node_id in assignment.node_ids
    }
    for demand in batch.demands:
        for byte_range in demand.ranges:
            ranges_by_node[assignment.owner_for(byte_range.source_rank)].append(
                byte_range
            )
    fqn_names = tuple(demand.fqn for demand in batch.demands)
    return tuple(
        BatchNodeWork(
            batch_index=batch.batch_index,
            node_id=node_id,
            fqn_names=fqn_names,
            exact_ranges=tuple(ranges_by_node[node_id]),
            download_ranges=tuple(ranges_by_node[node_id]),
        )
        for node_id in assignment.node_ids
    )


def _build_disjoint_download_works(
    works: Sequence[BatchNodeWork], gap_bytes: int
) -> tuple[BatchNodeWork, ...]:
    gap = _require_u64("consolidate_gap_bytes", gap_bytes)
    blockers: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for work in works:
        for byte_range in work.exact_ranges:
            blockers[byte_range.source_rank].append(
                (byte_range.offset, byte_range.end, work.batch_index)
            )
    for items in blockers.values():
        items.sort()
    blocker_starts = {
        source_rank: tuple(item[0] for item in items)
        for source_rank, items in blockers.items()
    }
    return tuple(
        BatchNodeWork(
            batch_index=work.batch_index,
            node_id=work.node_id,
            fqn_names=work.fqn_names,
            exact_ranges=work.exact_ranges,
            download_ranges=_consolidate_without_foreign_ranges(
                work.exact_ranges,
                work.batch_index,
                blockers,
                blocker_starts,
                gap,
            ),
        )
        for work in works
    )


def _consolidate_without_foreign_ranges(
    ranges: Sequence[ByteRange],
    batch_index: int,
    blockers: Mapping[int, Sequence[tuple[int, int, int]]],
    blocker_starts: Mapping[int, Sequence[int]],
    gap_bytes: int,
) -> tuple[ByteRange, ...]:
    consolidated: list[ByteRange] = []
    for byte_range in ranges:
        if not consolidated or byte_range.source_rank != consolidated[-1].source_rank:
            consolidated.append(byte_range)
            continue
        previous = consolidated[-1]
        gap_is_small = byte_range.offset <= previous.end + gap_bytes
        source_blockers = blockers[byte_range.source_rank]
        insertion = bisect_left(
            blocker_starts[byte_range.source_rank],
            byte_range.offset,
        )
        preceding = source_blockers[insertion - 1] if insertion else None
        has_foreign_range = (
            preceding is not None
            and preceding[1] > previous.end
            and preceding[2] != batch_index
        )
        if not gap_is_small or has_foreign_range:
            consolidated.append(byte_range)
            continue
        consolidated[-1] = ByteRange(
            previous.source_rank,
            previous.offset,
            max(previous.end, byte_range.end) - previous.offset,
        )
    return tuple(consolidated)


def _validate_disjoint_batch_downloads(works: Iterable[BatchNodeWork]) -> None:
    active: tuple[ByteRange, int] | None = None
    tagged = sorted(
        (byte_range, work.batch_index)
        for work in works
        for byte_range in work.download_ranges
    )
    for byte_range, batch_index in tagged:
        if active is None or active[0].source_rank != byte_range.source_rank:
            active = (byte_range, batch_index)
            continue
        active_range, active_batch = active
        if byte_range.offset < active_range.end and batch_index != active_batch:
            raise ValueError(
                "consolidated download ranges overlap across pipeline batches"
            )
        if byte_range.end > active_range.end:
            active = (byte_range, batch_index)


def chunks_for_bytes(byte_count: int, chunk_bytes: int) -> int:
    size = _require_u64("byte_count", byte_count)
    chunk = _require_u64("chunk_bytes", chunk_bytes, minimum=1)
    return (size + chunk - 1) // chunk


def chunks_for_batches(batch_bytes: Iterable[int], chunk_bytes: int) -> int:
    chunk = _require_u64("chunk_bytes", chunk_bytes, minimum=1)
    return sum(chunks_for_bytes(byte_count, chunk) for byte_count in batch_bytes)


def split_flat_range_into_chunks(
    flat_offset: int,
    length: int,
    chunk_bytes: int,
) -> tuple[ChunkPiece, ...]:
    offset = _require_u64("flat_offset", flat_offset)
    remaining = _require_u64("length", length)
    chunk = _require_u64("chunk_bytes", chunk_bytes, minimum=1)
    _checked_end(offset, remaining)
    pieces: list[ChunkPiece] = []
    position = offset
    while remaining:
        offset_in_chunk = position % chunk
        piece_length = min(remaining, chunk - offset_in_chunk)
        pieces.append(ChunkPiece(position // chunk, offset_in_chunk, piece_length))
        position += piece_length
        remaining -= piece_length
    return tuple(pieces)
