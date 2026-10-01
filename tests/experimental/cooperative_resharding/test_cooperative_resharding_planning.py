# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
from collections import Counter
from dataclasses import dataclass
from unittest import mock

import pytest
from torch_checkpointing.experimental.cooperative_resharding.planning import (
    _connect_overlapping_demands,
    _execution_plan_to_wire_from_validated,
    _merge_canonical_fqn_demand_wire_payloads,
    _merge_fqn_demand_wire_payloads_to_canonical_wire,
    _plan_cooperative_resharding_from_merged_demands,
    _rebuild_batch_node_works_from_canonical,
    assign_sources_to_nodes,
    assign_sources_to_nodes_with_stats,
    assignment_locality_stats,
    BatchNodeWork,
    build_batch_node_works,
    ByteRange,
    ChunkPiece,
    consolidate_byte_ranges,
    execution_plan_to_wire,
    FileAssignment,
    FqnDemand,
    merge_fqn_demand_wire_payloads,
    merge_fqn_demands,
    NodeId,
    NodeMembership,
    overlap_connected_fqn_groups,
    PipelineBatch,
    plan_pipeline_batches,
    planning_record_from_json,
    planning_record_to_json,
    project_execution_plan_wire,
    ProjectedBatchDownload,
    ProjectedSourceSchedule,
    RankTopology,
    split_flat_range_into_chunks,
    subtract_byte_ranges,
    union_byte_ranges,
    validate_assignment_coverage,
    validate_batch_coverage,
    validate_exact_coverage,
)


def _demand(fqn: str, *ranges: tuple[int, int, int]) -> FqnDemand:
    return FqnDemand(
        fqn=fqn,
        ranges=tuple(
            ByteRange(source_rank=source_rank, offset=offset, length=length)
            for source_rank, offset, length in ranges
        ),
    )


def _reference_connect_overlapping_demands(
    demands: tuple[FqnDemand, ...],
) -> list[int]:
    parent = list(range(len(demands)))
    sizes = [1] * len(demands)

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            next_item = parent[item]
            parent[item] = root
            item = next_item
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    tagged = sorted(
        (byte_range, index)
        for index, demand in enumerate(demands)
        for byte_range in demand.ranges
    )
    active: tuple[ByteRange, int] | None = None
    for byte_range, item_index in tagged:
        if active is None or active[0].source_rank != byte_range.source_rank:
            active = (byte_range, item_index)
            continue
        active_range, active_index = active
        if byte_range.offset >= active_range.end:
            active = (byte_range, item_index)
            continue
        union(active_index, item_index)
        if byte_range.end > active_range.end:
            active = (byte_range, item_index)
    return parent


def _wire_demand(fqn: str, *ranges: tuple[int, int, int]) -> dict[str, object]:
    return {
        "fqn": fqn,
        "ranges": [
            {"source_rank": source_rank, "offset": offset, "length": length}
            for source_rank, offset, length in ranges
        ],
    }


_LARGE_SCALE_DEMAND_COUNT = 1_024
_LARGE_SCALE_RANGE_COUNT = 262_144
_LARGE_SCALE_SOURCE_COUNT = 256
_LARGE_SCALE_NODE_COUNT = 8


def _large_synthetic_demands() -> tuple[FqnDemand, ...]:
    range_length = 4 * 1024 * 1024
    cluster_stride = 256 * 1024 * 1024
    cluster_count = 16
    ranges_by_fqn: list[list[ByteRange]] = [
        [] for _ in range(_LARGE_SCALE_DEMAND_COUNT)
    ]
    for index in range(_LARGE_SCALE_RANGE_COUNT):
        source_rank = (index // _LARGE_SCALE_DEMAND_COUNT) % _LARGE_SCALE_SOURCE_COUNT
        source_index = index % _LARGE_SCALE_DEMAND_COUNT
        cluster_index = source_index % cluster_count
        cluster_slot = source_index // cluster_count
        ranges_by_fqn[index % _LARGE_SCALE_DEMAND_COUNT].append(
            ByteRange(
                source_rank,
                cluster_index * cluster_stride + cluster_slot * range_length,
                range_length,
            )
        )
    return tuple(
        FqnDemand(f"tensor.{index:04d}", tuple(ranges))
        for index, ranges in enumerate(ranges_by_fqn)
    )


@dataclass(frozen=True)
class _LocalRange:
    offset: int
    length: int


@dataclass(frozen=True)
class _LocalPattern:
    ranges: tuple[_LocalRange, ...]

    @property
    def dense_nbytes(self) -> int:
        return sum(item.length for item in self.ranges)

    def iter_ranges(self) -> object:
        return iter(self.ranges)


@dataclass(frozen=True)
class _LocalTarget:
    target_fqn: str
    source_fqn: str
    source_rank: int
    source_pattern: _LocalPattern


def _local_target(
    fqn: str,
    source_rank: int,
    *ranges: tuple[int, int],
    source_fqn: str | None = None,
) -> _LocalTarget:
    return _LocalTarget(
        target_fqn=fqn,
        source_fqn=fqn if source_fqn is None else source_fqn,
        source_rank=source_rank,
        source_pattern=_LocalPattern(
            tuple(_LocalRange(offset, length) for offset, length in ranges)
        ),
    )


def _topology() -> RankTopology:
    return RankTopology(
        global_rank=11,
        nodes=(
            NodeMembership("node-d", (42,)),
            NodeMembership("node-b", (11, 5)),
            NodeMembership("node-a", (7, 3)),
            NodeMembership("node-c", (21, 19)),
        ),
        coordination_world_count=3,
        job_id="load-17",
    )


def test_explicit_topology_supports_noncontiguous_uneven_nodes() -> None:
    topology = _topology()

    assert topology.world_size == 7
    assert topology.global_num_nodes == 4
    assert topology.node_id == "node-b"
    assert topology.node_ranks == (5, 11)
    assert topology.node_leader_rank == 5
    assert topology.local_rank == 1
    assert topology.global_node_index == 1
    assert topology.coordination_world.node_ids == ("node-a", "node-b")
    assert topology.coordination_world.world_id == 0
    assert topology.coordination_world.world_node_index == 1
    assert (
        topology.rendezvous_id
        == "load-17__coord_leader-rank-node-membership-v2__r3__w0"
    )
    assert not topology.is_node_leader
    assert not topology.is_world_leader


def test_topology_is_canonical_and_rejects_duplicate_rank_membership() -> None:
    topology = _topology()
    reordered = RankTopology(
        global_rank=11,
        nodes=tuple(reversed(topology.nodes)),
        coordination_world_count=3,
        job_id="load-17",
    )

    assert topology == reordered
    assert planning_record_to_json(topology) == planning_record_to_json(reordered)
    with pytest.raises(ValueError, match="exactly once"):
        RankTopology(
            global_rank=3,
            nodes=(NodeMembership("a", (3, 9)), NodeMembership("b", (3, 11))),
            coordination_world_count=1,
            job_id="job",
        )


def test_union_subtraction_and_adjacency_are_exact() -> None:
    ranges = (
        ByteRange(1, 5, 5),
        ByteRange(0, 20, 5),
        ByteRange(0, 0, 10),
        ByteRange(0, 8, 12),
        ByteRange(1, 0, 5),
    )

    assert union_byte_ranges(ranges) == (
        ByteRange(0, 0, 25),
        ByteRange(1, 0, 10),
    )
    required = (ByteRange(0, 0, 100), ByteRange(0, 200, 100))
    covered = (
        ByteRange(0, 20, 30),
        ByteRange(0, 60, 50),
        ByteRange(0, 250, 25),
    )
    assert subtract_byte_ranges(required, covered) == (
        ByteRange(0, 0, 20),
        ByteRange(0, 50, 10),
        ByteRange(0, 200, 50),
        ByteRange(0, 275, 25),
    )
    validate_exact_coverage(
        (ByteRange(2, 0, 20),),
        (ByteRange(2, 0, 5), ByteRange(2, 5, 15)),
    )
    with pytest.raises(ValueError, match="unexpected"):
        validate_exact_coverage((ByteRange(2, 0, 20),), (ByteRange(2, 0, 21),))


def test_consolidation_bridges_only_configured_gaps() -> None:
    ranges = (
        ByteRange(0, 0, 10),
        ByteRange(0, 15, 5),
        ByteRange(0, 26, 4),
        ByteRange(1, 0, 100),
    )

    assert consolidate_byte_ranges(ranges, gap_bytes=5) == (
        ByteRange(0, 0, 20),
        ByteRange(0, 26, 4),
        ByteRange(1, 0, 100),
    )


def test_overlap_components_are_transitive_but_not_adjacent() -> None:
    demands = (
        _demand("a", (0, 0, 10)),
        _demand("b", (0, 8, 12)),
        _demand("c", (0, 18, 12)),
        _demand("adjacent", (0, 30, 5)),
        _demand("other-source", (1, 0, 100)),
    )

    groups = overlap_connected_fqn_groups(reversed(demands))

    assert tuple(tuple(item.fqn for item in group) for group in groups) == (
        ("a", "b", "c"),
        ("adjacent",),
        ("other-source",),
    )


def test_primitive_overlap_connect_preserves_union_roots() -> None:
    randomizer = random.Random(20260824)
    source_ranks = (0, 1, 7, 2_048, 1_000_003)
    for _ in range(1_000):
        raw_demands = tuple(
            _demand(
                f"tensor.{randomizer.randrange(max(1, index // 4 + 1))}",
                *(
                    (
                        randomizer.choice(source_ranks),
                        randomizer.randrange(24) * 8,
                        randomizer.randrange(1, 33),
                    )
                    for _ in range(randomizer.randrange(1, 13))
                ),
            )
            for index in range(randomizer.randrange(45))
        )
        demands = merge_fqn_demands(raw_demands)

        assert _connect_overlapping_demands(
            demands
        ) == _reference_connect_overlapping_demands(demands)


def test_wire_merge_matches_object_merge_for_randomized_payloads() -> None:
    randomizer = random.Random(20260821)
    for _ in range(100):
        payloads: list[list[object]] = []
        for _ in range(randomizer.randint(1, 6)):
            payload = [
                _wire_demand(
                    f"tensor.{randomizer.randrange(7)}",
                    *(
                        (
                            randomizer.randrange(4),
                            randomizer.randrange(200),
                            randomizer.randrange(1, 40),
                        )
                        for _ in range(randomizer.randint(1, 12))
                    ),
                )
                for _ in range(randomizer.randint(0, 8))
            ]
            randomizer.shuffle(payload)
            payloads.append(payload)

        expected = merge_fqn_demands(
            FqnDemand.from_dict(raw_demand)
            for payload in payloads
            for raw_demand in payload
        )
        result = merge_fqn_demand_wire_payloads(iter(payloads))
        canonical_result = _merge_fqn_demand_wire_payloads_to_canonical_wire(
            iter(payloads)
        )
        canonical_materialized = _merge_canonical_fqn_demand_wire_payloads(
            (canonical_result.payload,)
        )

        assert result.demands == expected
        assert result.input_demand_count == sum(len(payload) for payload in payloads)
        assert result.input_range_count == sum(
            len(raw_demand["ranges"])
            for payload in payloads
            for raw_demand in payload
            if isinstance(raw_demand, dict)
            and isinstance(raw_demand.get("ranges"), list)
        )
        assert canonical_result.payload == [demand.to_dict() for demand in expected]
        assert canonical_result.input_demand_count == result.input_demand_count
        assert canonical_result.input_range_count == result.input_range_count
        assert canonical_result.output_demand_count == len(expected)
        assert canonical_result.output_range_count == sum(
            len(demand.ranges) for demand in expected
        )
        assert canonical_materialized.demands == expected


def test_wire_merge_materializes_only_final_union() -> None:
    payloads = [
        [_wire_demand("weight", *((0, offset, 2) for offset in range(0, 200, 2)))]
        for _ in range(20)
    ]

    result = merge_fqn_demand_wire_payloads(iter(payloads))

    assert result.input_demand_count == 20
    assert result.input_range_count == 2_000
    assert result.demands == (_demand("weight", (0, 0, 200)),)


def test_canonical_wire_merge_handles_large_synthetic_scale() -> None:
    demands = _large_synthetic_demands()
    expected = merge_fqn_demands(demands)
    payloads: list[list[object]] = [[] for _ in range(_LARGE_SCALE_NODE_COUNT)]
    for demand in demands:
        ranges_by_node: list[list[dict[str, object]]] = [
            [] for _ in range(_LARGE_SCALE_NODE_COUNT)
        ]
        for byte_range in demand.ranges:
            ranges_by_node[byte_range.source_rank % _LARGE_SCALE_NODE_COUNT].append(
                byte_range.to_dict()
            )
        for node_index, ranges in enumerate(ranges_by_node):
            if ranges:
                payloads[node_index].append(
                    {
                        "fqn": demand.fqn,
                        "ranges": ranges,
                    }
                )

    canonical = _merge_fqn_demand_wire_payloads_to_canonical_wire(payloads)
    materialized = merge_fqn_demand_wire_payloads(payloads)
    canonical_materialized = _merge_canonical_fqn_demand_wire_payloads(
        (canonical.payload,)
    )

    assert canonical.input_demand_count == (
        _LARGE_SCALE_NODE_COUNT * _LARGE_SCALE_DEMAND_COUNT
    )
    assert canonical.input_range_count == _LARGE_SCALE_RANGE_COUNT
    assert canonical.output_demand_count == _LARGE_SCALE_DEMAND_COUNT
    assert canonical.output_range_count == _LARGE_SCALE_RANGE_COUNT
    assert canonical.payload == [demand.to_dict() for demand in expected]
    assert materialized.demands == expected
    assert canonical_materialized.demands == expected


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [1],
        [{"fqn": "x"}],
        [
            {
                "fqn": "x",
                "ranges": [{"source_rank": 0, "offset": 0, "length": 1}],
                "extra": 1,
            }
        ],
        [{"fqn": "x", "ranges": []}],
        [{"fqn": "x", "ranges": "bad"}],
        [{"fqn": 1, "ranges": [{"source_rank": 0, "offset": 0, "length": 1}]}],
        [
            {
                "fqn": "x",
                "ranges": [{"source_rank": 0, "offset": 0, "length": 1, "extra": 1}],
            }
        ],
        [{"fqn": "x", "ranges": [{"source_rank": True, "offset": 0, "length": 1}]}],
        [{"fqn": "x", "ranges": [{"source_rank": -1, "offset": 0, "length": 1}]}],
        [{"fqn": "x", "ranges": [{"source_rank": 0, "offset": True, "length": 1}]}],
        [{"fqn": "x", "ranges": [{"source_rank": 0, "offset": -1, "length": 1}]}],
        [{"fqn": "x", "ranges": [{"source_rank": 0, "offset": 0, "length": True}]}],
        [{"fqn": "x", "ranges": [{"source_rank": 0, "offset": 0, "length": 0}]}],
        [
            {
                "fqn": "x",
                "ranges": [{"source_rank": 0, "offset": 1 << 64, "length": 1}],
            }
        ],
        [
            {
                "fqn": "x",
                "ranges": [{"source_rank": 0, "offset": 0, "length": 1 << 64}],
            }
        ],
        [
            {
                "fqn": "x",
                "ranges": [{"source_rank": 0, "offset": (1 << 64) - 1, "length": 2}],
            }
        ],
    ],
)
def test_wire_merge_rejects_malformed_payloads(payload: object) -> None:
    for merge in (
        merge_fqn_demand_wire_payloads,
        _merge_fqn_demand_wire_payloads_to_canonical_wire,
    ):
        with pytest.raises(ValueError):
            merge(iter((payload,)))


@pytest.mark.parametrize(
    "payload",
    [
        (_wire_demand("weight", (0, 0, 1)),),
        [
            _wire_demand("later", (1, 0, 1)),
            _wire_demand("earlier", (0, 0, 1)),
        ],
        [
            _wire_demand("weight", (0, 0, 1)),
            _wire_demand("weight", (1, 0, 1)),
        ],
        [_wire_demand("weight", (1, 0, 1), (0, 0, 1))],
        [_wire_demand("weight", (0, 0, 2), (0, 2, 1))],
        [_wire_demand("weight", (0, 0, 3), (0, 2, 2))],
        [
            {
                "fqn": "weight",
                "ranges": [{"source_rank": True, "offset": 0, "length": 1}],
            }
        ],
        [
            {
                "fqn": "weight",
                "ranges": [{"source_rank": 0, "offset": True, "length": 1}],
            }
        ],
        [
            {
                "fqn": "weight",
                "ranges": [{"source_rank": 0, "offset": 0, "length": True}],
            }
        ],
        [
            {
                "fqn": "weight",
                "ranges": [{"source_rank": 0, "offset": 1 << 64, "length": 1}],
            }
        ],
        [
            {
                "fqn": "weight",
                "ranges": [{"source_rank": 0, "offset": 0, "length": 1 << 64}],
            }
        ],
        [
            {
                "fqn": "weight",
                "ranges": [{"source_rank": 0, "offset": (1 << 64) - 1, "length": 2}],
            }
        ],
        [
            {
                "fqn": "weight",
                "ranges": ({"source_rank": 0, "offset": 0, "length": 1},),
            }
        ],
    ],
)
def test_canonical_wire_merge_rejects_noncanonical_payloads(
    payload: object,
) -> None:
    with pytest.raises(ValueError, match="canonical rank byte"):
        _merge_canonical_fqn_demand_wire_payloads((payload,))


def test_source_assignment_is_byte_balanced_and_permutation_invariant() -> None:
    demands = (
        _demand("source-2", (22, 0, 100)),
        _demand("source-0", (4, 0, 300)),
        _demand("source-1", (17, 0, 200)),
        _demand("source-3", (91, 0, 100)),
    )

    forward = assign_sources_to_nodes(demands, ("node-z", "node-a"))
    reverse = assign_sources_to_nodes(reversed(demands), ("node-a", "node-z"))

    assert forward == reverse
    assert forward.node_ids == ("node-a", "node-z")
    assert forward.node_source_ranks == ((4, 91), (17, 22))
    assert forward.node_bytes == (400, 300)
    assert forward.owner_for(4) == "node-a"
    assert planning_record_to_json(forward) == planning_record_to_json(reverse)


def test_source_assignment_prefers_local_consumers_without_changing_balance() -> None:
    demands = tuple(
        _demand(f"source-{source_rank}", (source_rank, 0, 100))
        for source_rank in range(4)
    )
    affinity = {
        0: {3: 100},
        8: {2: 100},
        16: {1: 100},
        24: {0: 100},
    }

    baseline = assign_sources_to_nodes(demands, (24, 8, 0, 16))
    result = assign_sources_to_nodes_with_stats(
        reversed(demands),
        (16, 0, 24, 8),
        source_consumer_bytes_by_node=affinity,
    )
    assignment = result.assignment

    assert baseline.node_bytes == assignment.node_bytes == (100, 100, 100, 100)
    assert assignment.node_source_ranks == ((3,), (2,), (1,), (0,))
    assert result.baseline_locality.local_consumer_bytes == 0
    assert result.chosen_locality.local_consumer_bytes == 400
    assert result.chosen_locality.remote_consumer_bytes == 0
    assert result.theoretical_max_local_consumer_bytes == 400


def test_source_assignment_affinity_preserves_skewed_lpt_balance() -> None:
    demands = (
        _demand("large", (0, 0, 10)),
        _demand("medium", (1, 0, 9)),
        _demand("small", (2, 0, 8)),
    )
    affinity = {
        0: {0: 10, 1: 9, 2: 8},
        8: {},
    }

    baseline = assign_sources_to_nodes(demands, (0, 8))
    result = assign_sources_to_nodes_with_stats(
        demands,
        (8, 0),
        source_consumer_bytes_by_node=affinity,
    )
    assignment = result.assignment

    assert assignment.node_bytes == baseline.node_bytes == (10, 17)
    assert max(assignment.node_bytes) == max(baseline.node_bytes)
    assert result.baseline_locality.local_consumer_bytes == 10
    assert result.chosen_locality.local_consumer_bytes == 10
    assert result.theoretical_max_local_consumer_bytes == 27


def test_source_assignment_does_not_accept_a_lower_total_affinity() -> None:
    demands = (
        _demand("large", (0, 0, 3)),
        _demand("medium", (1, 0, 2)),
        _demand("small", (2, 0, 1)),
    )
    affinity = {
        0: {},
        8: {0: 1, 1: 100, 2: 100},
    }

    baseline = assign_sources_to_nodes(demands, (0, 8))
    assignment = assign_sources_to_nodes(
        demands,
        (8, 0),
        source_consumer_bytes_by_node=affinity,
    )

    assert assignment == baseline
    assert assignment_locality_stats(assignment, affinity).local_consumer_bytes == 200


def test_source_assignment_affinity_counts_replicated_consumer_bytes() -> None:
    demands = (
        _demand("source-0", (0, 0, 100)),
        _demand("source-1", (1, 0, 100)),
    )
    affinity = {
        0: {0: 200, 1: 100},
        8: {0: 100, 1: 200},
    }

    assignment = assign_sources_to_nodes(
        demands,
        (8, 0),
        source_consumer_bytes_by_node=affinity,
    )
    stats = assignment_locality_stats(assignment, affinity)

    assert assignment.node_source_ranks == ((0,), (1,))
    assert stats.node_total_consumer_bytes == (300, 300)
    assert stats.node_local_consumer_bytes == (200, 200)
    assert stats.node_remote_consumer_bytes == (100, 100)
    assert stats.total_consumer_bytes == 600
    assert stats.local_consumer_bytes == 400
    assert stats.remote_consumer_bytes == 200


@pytest.mark.parametrize(
    "affinity,match",
    [
        ({0: {}, 8: {}, 16: {}}, "exactly the assignment nodes"),
        ({0: {0: 100}, 8: {7: 100}}, "unknown source rank 7"),
        ({0: {0: 100}, 8: {1: -1}}, "consumer_bytes"),
    ],
)
def test_source_assignment_rejects_invalid_affinity(
    affinity: object,
    match: str,
) -> None:
    demands = (
        _demand("source-0", (0, 0, 100)),
        _demand("source-1", (1, 0, 100)),
    )

    with pytest.raises(ValueError, match=match):
        assign_sources_to_nodes(
            demands,
            (0, 8),
            source_consumer_bytes_by_node=affinity,  # pyre-ignore[6]
        )


def test_batch_budget_is_per_node_and_oversized_components_are_isolated() -> None:
    demands = (
        _demand("node-a-first", (0, 0, 80)),
        _demand("node-a-second", (0, 100, 30)),
        _demand("node-b-first", (1, 0, 80)),
        _demand("node-b-second", (1, 100, 30)),
        _demand("large-a", (2, 0, 100)),
        _demand("large-b", (2, 90, 80)),
    )
    assignment = FileAssignment(
        node_ids=("node-a", "node-b"),
        node_source_ranks=((0, 2), (1,)),
        node_bytes=(280, 110),
    )

    batches = plan_pipeline_batches(
        reversed(demands),
        batch_budget_bytes=100,
        assignment=assignment,
    )
    works = build_batch_node_works(batches, assignment, consolidate_gap_bytes=0)

    assert batches == plan_pipeline_batches(demands, 100, assignment)
    oversized = next(
        batch
        for batch in batches
        if {demand.fqn for demand in batch.demands} == {"large-a", "large-b"}
    )
    assert oversized.total_bytes == 180
    assert len(oversized.demands) == 2
    oversized_work = next(
        work
        for work in works
        if work.batch_index == oversized.batch_index and work.node_id == "node-a"
    )
    assert oversized_work.exact_bytes == 170
    assert all(
        len(batch.demands) == 2
        for batch in batches
        if batch.batch_index != oversized.batch_index
    )
    assert {(work.batch_index, work.node_id) for work in works} == {
        (batch.batch_index, node_id)
        for batch in batches
        for node_id in assignment.node_ids
    }


def test_batch_budget_accounts_for_download_consolidation() -> None:
    demands = (
        _demand("first", (0, 0, 1)),
        _demand("second", (0, 100, 1)),
    )
    assignment = assign_sources_to_nodes(
        demands,
        ("node",),
        consolidate_gap_bytes=100,
    )

    batches = plan_pipeline_batches(
        demands,
        batch_budget_bytes=2,
        assignment=assignment,
        consolidate_gap_bytes=100,
    )
    works = build_batch_node_works(
        batches,
        assignment,
        consolidate_gap_bytes=100,
    )

    assert len(batches) == 2
    assert all(work.download_bytes <= 2 for work in works)
    assert batches == plan_pipeline_batches(
        reversed(demands),
        batch_budget_bytes=2,
        assignment=assignment,
        consolidate_gap_bytes=100,
    )


def test_consolidation_oversized_component_remains_isolated() -> None:
    demands = (_demand("oversized", (0, 0, 1), (0, 100, 1)),)
    assignment = assign_sources_to_nodes(
        demands,
        ("node",),
        consolidate_gap_bytes=100,
    )

    batches = plan_pipeline_batches(
        demands,
        batch_budget_bytes=2,
        assignment=assignment,
        consolidate_gap_bytes=100,
    )
    works = build_batch_node_works(
        batches,
        assignment,
        consolidate_gap_bytes=100,
    )

    assert len(batches) == 1
    assert tuple(demand.fqn for demand in batches[0].demands) == ("oversized",)
    assert works[0].download_bytes == 101


def test_download_consolidation_does_not_cross_another_batch() -> None:
    demands = (
        _demand("outer", (0, 0, 1), (0, 10, 1)),
        _demand("inner", (0, 5, 1)),
    )
    assignment = assign_sources_to_nodes(demands, ("node",))
    batches = plan_pipeline_batches(
        demands,
        batch_budget_bytes=2,
        assignment=assignment,
        consolidate_gap_bytes=10,
    )

    works = build_batch_node_works(batches, assignment, consolidate_gap_bytes=10)

    assert len(batches) == 2
    assert works[0].download_ranges == (ByteRange(0, 0, 1), ByteRange(0, 10, 1))
    assert works[1].download_ranges == (ByteRange(0, 5, 1),)


def test_fused_cooperative_planning_is_randomly_equivalent() -> None:
    randomizer = random.Random(20260823)
    for _ in range(100):
        node_ids = tuple(range(randomizer.randint(1, 5)))
        raw_demands = tuple(
            _demand(
                f"tensor.{randomizer.randrange(max(1, index // 3 + 1))}",
                *(
                    (
                        randomizer.randrange(10),
                        randomizer.randrange(700),
                        randomizer.randrange(1, 70),
                    )
                    for _ in range(randomizer.randint(1, 10))
                ),
            )
            for index in range(randomizer.randint(0, 45))
        )
        merged_demands = merge_fqn_demands(raw_demands)
        source_ranks = {
            byte_range.source_rank
            for demand in merged_demands
            for byte_range in demand.ranges
        }
        source_consumer_bytes_by_node = {
            node_id: {
                source_rank: randomizer.randrange(1_000)
                for source_rank in source_ranks
                if randomizer.random() < 0.65
            }
            for node_id in node_ids
        }
        gap_bytes = randomizer.randrange(16)
        batch_budget_bytes = randomizer.randrange(1_200)

        expected_assignment = assign_sources_to_nodes_with_stats(
            merged_demands,
            node_ids,
            gap_bytes,
            source_consumer_bytes_by_node=source_consumer_bytes_by_node,
        )
        expected_batches = plan_pipeline_batches(
            merged_demands,
            batch_budget_bytes,
            expected_assignment.assignment,
            gap_bytes,
        )
        expected_works = build_batch_node_works(
            expected_batches,
            expected_assignment.assignment,
            gap_bytes,
        )

        actual = _plan_cooperative_resharding_from_merged_demands(
            merged_demands,
            node_ids,
            batch_budget_bytes,
            gap_bytes,
            source_consumer_bytes_by_node=source_consumer_bytes_by_node,
        )

        assert actual.assignment_result == expected_assignment
        assert actual.batches == expected_batches
        assert actual.works == expected_works
        assert actual.assignment_ns >= 0
        assert actual.batching_ns >= 0
        assert actual.work_ns >= 0
        assert _rebuild_batch_node_works_from_canonical(actual.works, 0) == (
            build_batch_node_works(
                expected_batches,
                expected_assignment.assignment,
                0,
            )
        )


def test_fused_cooperative_planning_handles_large_synthetic_scale() -> None:
    demands = _large_synthetic_demands()
    source_bytes: Counter[int] = Counter()
    for demand in demands:
        for byte_range in demand.ranges:
            source_bytes[byte_range.source_rank] += byte_range.length
    source_consumer_bytes_by_node = {
        node_id: {
            source_rank: byte_count
            for source_rank, byte_count in source_bytes.items()
            if source_rank % _LARGE_SCALE_NODE_COUNT == node_id
        }
        for node_id in range(_LARGE_SCALE_NODE_COUNT)
    }

    result = _plan_cooperative_resharding_from_merged_demands(
        demands,
        tuple(range(_LARGE_SCALE_NODE_COUNT)),
        64 * 1024 * 1024 * 1024,
        8 * 1024 * 1024,
        source_consumer_bytes_by_node=source_consumer_bytes_by_node,
    )

    assert len(demands) == _LARGE_SCALE_DEMAND_COUNT
    assert sum(len(demand.ranges) for demand in demands) == _LARGE_SCALE_RANGE_COUNT
    assert (
        len(result.assignment_result.assignment.source_rank_to_node)
        == _LARGE_SCALE_SOURCE_COUNT
    )
    assert len(result.batches) == 2
    assert len(result.works) == len(result.batches) * _LARGE_SCALE_NODE_COUNT


def test_fused_cooperative_planning_scales_to_2048_sources_and_256_nodes() -> None:
    source_count = 2_048
    node_count = 256
    range_length = 8
    demands = tuple(
        _demand(
            f"tensor.{source_rank:04d}",
            (source_rank, source_rank * 16, range_length),
        )
        for source_rank in range(source_count)
    )
    node_ids = tuple(range(node_count))
    source_consumer_bytes_by_node = {
        node_id: {
            source_rank: range_length
            for source_rank in range(node_id, source_count, node_count)
        }
        for node_id in node_ids
    }
    expected_assignment = assign_sources_to_nodes_with_stats(
        demands,
        node_ids,
        source_consumer_bytes_by_node=source_consumer_bytes_by_node,
    )
    expected_batches = plan_pipeline_batches(
        demands,
        4 * range_length,
        expected_assignment.assignment,
    )
    expected_works = build_batch_node_works(
        expected_batches,
        expected_assignment.assignment,
        0,
    )

    result = _plan_cooperative_resharding_from_merged_demands(
        demands,
        node_ids,
        4 * range_length,
        0,
        source_consumer_bytes_by_node=source_consumer_bytes_by_node,
    )

    assert len(result.assignment_result.assignment.source_rank_to_node) == source_count
    assert len(result.batches) == 2
    assert len(result.works) == 2 * node_count
    assert result.assignment_result == expected_assignment
    assert result.batches == expected_batches
    assert result.works == expected_works


def test_empty_root_fqn_round_trips_through_planning_wire() -> None:
    demand = _demand("", (3, 10, 20))
    batch = PipelineBatch(0, (demand,))
    work = BatchNodeWork(0, "node", ("",), demand.ranges, demand.ranges)

    for record in (demand, batch, work):
        assert planning_record_from_json(planning_record_to_json(record)) == record
    assert merge_fqn_demand_wire_payloads(([demand.to_dict()],)).demands == (demand,)


def test_planning_wire_round_trips_and_rejects_unknown_fields() -> None:
    topology = _topology()
    demand = _demand("weight", (3, 10, 20))
    assignment = FileAssignment(("node-a", "node-b"), ((3,), ()), (20, 0))
    batch = PipelineBatch(0, (demand,))
    work = BatchNodeWork(0, "node-a", ("weight",), demand.ranges, demand.ranges)
    records = (
        topology.nodes[0],
        topology,
        topology.coordination_world,
        demand.ranges[0],
        demand,
        assignment,
        batch,
        work,
        ChunkPiece(0, 10, 6),
    )

    for record in records:
        encoded = planning_record_to_json(record)
        restored = planning_record_from_json(encoded)
        assert restored == record
        assert planning_record_to_json(restored) == encoded
        assert " " not in encoded

    envelope = json.loads(planning_record_to_json(ByteRange(0, 0, 10)))
    envelope["version"] = 2
    with pytest.raises(ValueError, match="version"):
        planning_record_from_json(json.dumps(envelope))
    envelope["version"] = 1
    envelope["payload"]["path"] = "/node-local/checkpoint.pt"
    with pytest.raises(ValueError, match="unexpected"):
        planning_record_from_json(json.dumps(envelope))
    envelope.pop("payload")
    with pytest.raises(ValueError, match="missing"):
        planning_record_from_json(json.dumps(envelope))


def _execution_plan_fixture() -> tuple[
    FileAssignment,
    tuple[PipelineBatch, ...],
    tuple[BatchNodeWork, ...],
    tuple[_LocalTarget, ...],
]:
    demands = (
        _demand("alpha", (0, 0, 10)),
        _demand("beta", (1, 100, 15)),
        _demand("gamma", (0, 40, 12)),
    )
    assignment = assign_sources_to_nodes(demands, (0, 8))
    batches = plan_pipeline_batches(
        demands,
        batch_budget_bytes=20,
        assignment=assignment,
    )
    works = build_batch_node_works(batches, assignment, consolidate_gap_bytes=4)
    targets = (
        _local_target("gamma", 0, (40, 12)),
        _local_target("beta", 1, (103, 4)),
        _local_target("alpha", 0, (2, 5)),
    )
    return assignment, batches, works, targets


def test_execution_plan_wire_projects_exact_rank_local_state() -> None:
    assignment, batches, works, targets = _execution_plan_fixture()

    wire = execution_plan_to_wire(assignment, batches, works)
    projection = project_execution_plan_wire(
        wire,
        expected_node_ids=assignment.node_ids,
        local_node_id=assignment.node_ids[0],
        local_targets=targets,
        node_capacities=dict.fromkeys(assignment.node_ids, 1_000),
    )

    assert set(wire) == {
        "batch_count",
        "fqn_batches",
        "node_ids",
        "sources",
        "version",
    }
    assert wire["node_ids"] == list(assignment.node_ids)
    assert wire["fqn_batches"] == [
        [fqn, batch_index]
        for fqn, batch_index in sorted(
            (demand.fqn, batch.batch_index)
            for batch in batches
            for demand in batch.demands
        )
    ]
    assert projection.batch_indices == tuple(range(len(batches)))
    expected_target_batches = {
        demand.fqn: batch.batch_index for batch in batches for demand in batch.demands
    }
    assert tuple(
        batch_index
        for batch_index, target_indices in enumerate(
            projection.local_target_indices_by_batch
        )
        for _ in target_indices
    ) == tuple(sorted(expected_target_batches[target.target_fqn] for target in targets))
    assert projection.source_owners == assignment.source_rank_to_node
    expected_active_nodes = tuple(
        tuple(
            node_id
            for node_id in assignment.node_ids
            if next(
                work
                for work in works
                if work.batch_index == batch.batch_index and work.node_id == node_id
            ).download_bytes
        )
        for batch in batches
    )
    assert projection.active_node_ids_by_batch == expected_active_nodes
    expected_local_downloads = tuple(
        next(
            work
            for work in works
            if work.batch_index == batch.batch_index
            and work.node_id == assignment.node_ids[0]
        ).download_ranges
        for batch in batches
    )
    assert (
        tuple(item.download_ranges for item in projection.local_downloads)
        == expected_local_downloads
    )
    assert all(
        isinstance(item, ProjectedBatchDownload) for item in projection.local_downloads
    )
    assert all(
        isinstance(item, ProjectedSourceSchedule)
        for item in projection.source_schedules.values()
    )
    with pytest.raises(TypeError):
        projection.source_owners[99] = 0  # pyre-ignore[16]


def test_execution_plan_projection_batches_by_target_fqn_after_rename() -> None:
    wire = _minimal_execution_wire()
    renamed = _local_target(
        "a",
        0,
        (0, 5),
        source_fqn="checkpoint.original_name",
    )

    projection = _project_minimal_execution_wire(wire, targets=(renamed,))

    assert projection.local_target_indices_by_batch == ((0,), ())


def test_execution_plan_projection_skips_zero_sized_targets() -> None:
    wire = _minimal_execution_wire()
    zero_before_nonzero = _local_target("not-in-plan", 99)
    nonzero = _local_target("b", 0, (20, 1))

    projection = _project_minimal_execution_wire(
        wire,
        targets=(zero_before_nonzero, nonzero),
    )

    assert projection.local_target_indices_by_batch == ((), (1,))

    empty_assignment = FileAssignment((0,), ((),), (0,))
    empty_projection = project_execution_plan_wire(
        execution_plan_to_wire(empty_assignment, (), ()),
        expected_node_ids=(0,),
        local_node_id=0,
        local_targets=(_local_target("only-zero", 12),),
        node_capacities={0: 0},
    )
    assert empty_projection.local_target_indices_by_batch == ()


def test_execution_plan_wire_is_deterministic_and_randomly_equivalent() -> None:
    randomizer = random.Random(20260822)
    for _ in range(50):
        demands = tuple(
            _demand(
                f"tensor.{index:03d}",
                (
                    randomizer.randrange(6),
                    index * 128,
                    randomizer.randrange(1, 65),
                ),
            )
            for index in range(randomizer.randrange(1, 60))
        )
        gap_bytes = randomizer.randrange(5)
        assignment = assign_sources_to_nodes(
            demands,
            (0, 8, 16),
            consolidate_gap_bytes=gap_bytes,
        )
        batches = plan_pipeline_batches(
            demands,
            batch_budget_bytes=randomizer.randrange(64, 513),
            assignment=assignment,
            consolidate_gap_bytes=gap_bytes,
        )
        works = build_batch_node_works(
            batches,
            assignment,
            consolidate_gap_bytes=gap_bytes,
        )
        selected_demands = demands[::3]
        targets = tuple(
            _local_target(
                demand.fqn,
                demand.ranges[0].source_rank,
                *((item.offset, item.length) for item in demand.ranges),
            )
            for demand in selected_demands
        )

        wire = execution_plan_to_wire(assignment, batches, works)
        assert wire == _execution_plan_to_wire_from_validated(
            assignment,
            batches,
            works,
        )
        assert wire == execution_plan_to_wire(
            assignment,
            tuple(reversed(batches)),
            tuple(reversed(works)),
        )
        local_node_id = randomizer.choice(assignment.node_ids)
        capacities = {
            node_id: max(
                (work.download_bytes for work in works if work.node_id == node_id),
                default=0,
            )
            for node_id in assignment.node_ids
        }
        projection = project_execution_plan_wire(
            wire,
            expected_node_ids=assignment.node_ids,
            local_node_id=local_node_id,
            local_targets=targets,
            node_capacities=capacities,
        )

        assert projection.source_owners == assignment.source_rank_to_node
        fqn_batches = {
            demand.fqn: batch.batch_index
            for batch in batches
            for demand in batch.demands
        }
        assert {
            target_index: batch_index
            for batch_index, target_indices in enumerate(
                projection.local_target_indices_by_batch
            )
            for target_index in target_indices
        } == {
            target_index: fqn_batches[target.target_fqn]
            for target_index, target in enumerate(targets)
        }
        assert tuple(
            item.download_ranges for item in projection.local_downloads
        ) == tuple(
            next(
                work
                for work in works
                if work.batch_index == batch.batch_index
                and work.node_id == local_node_id
            ).download_ranges
            for batch in batches
        )
        for work in works:
            ranges_by_source: dict[int, list[ByteRange]] = {}
            for byte_range in work.download_ranges:
                ranges_by_source.setdefault(byte_range.source_rank, []).append(
                    byte_range
                )
            for source_rank, source_ranges in ranges_by_source.items():
                assert projection.schedule_for(
                    work.batch_index,
                    source_rank,
                ).ranges == tuple(source_ranges)


def test_execution_plan_wire_supports_empty_rank_and_inactive_node() -> None:
    empty_assignment = FileAssignment((0, 8), ((), ()), (0, 0))
    empty_wire = execution_plan_to_wire(empty_assignment, (), ())

    empty_projection = project_execution_plan_wire(
        empty_wire,
        expected_node_ids=(0, 8),
        local_node_id=8,
        local_targets=(),
        node_capacities={0: 0, 8: 0},
    )

    assert empty_projection.batch_indices == ()
    assert empty_projection.local_target_indices_by_batch == ()
    assert empty_projection.source_owners == {}
    assert empty_projection.source_schedules == {}
    assert empty_projection.active_node_ids_by_batch == ()
    assert empty_projection.local_downloads == ()

    demand = _demand("only", (0, 0, 10))
    assignment = FileAssignment((0, 8), ((0,), ()), (10, 0))
    batches = (PipelineBatch(0, (demand,)),)
    works = build_batch_node_works(batches, assignment, consolidate_gap_bytes=0)
    inactive_node = 8
    projection = project_execution_plan_wire(
        execution_plan_to_wire(assignment, batches, works),
        expected_node_ids=assignment.node_ids,
        local_node_id=inactive_node,
        local_targets=(),
        node_capacities=dict.fromkeys(assignment.node_ids, 1_000),
    )
    assert all(not item.download_ranges for item in projection.local_downloads)


def test_execution_plan_encoder_rejects_non_equivalent_works() -> None:
    assignment, batches, works, _ = _execution_plan_fixture()

    with pytest.raises(ValueError, match="every batch/node"):
        execution_plan_to_wire(assignment, batches, works[:-1])

    first = works[0]
    wrong_fqns = BatchNodeWork(
        first.batch_index,
        first.node_id,
        ("not-the-batch",),
        first.exact_ranges,
        first.download_ranges,
    )
    with pytest.raises(ValueError, match="FQNs"):
        execution_plan_to_wire(assignment, batches, (wrong_fqns, *works[1:]))

    overlap_assignment = FileAssignment((0,), ((0,),), (2,))
    overlap_batches = (
        PipelineBatch(0, (_demand("a", (0, 0, 1)),)),
        PipelineBatch(1, (_demand("b", (0, 20, 1)),)),
    )
    overlap_works = (
        BatchNodeWork(0, 0, ("a",), (ByteRange(0, 0, 1),), (ByteRange(0, 0, 10),)),
        BatchNodeWork(
            1,
            0,
            ("b",),
            (ByteRange(0, 20, 1),),
            (ByteRange(0, 5, 20),),
        ),
    )
    with pytest.raises(ValueError, match="overlap across"):
        execution_plan_to_wire(
            overlap_assignment,
            overlap_batches,
            overlap_works,
        )


def _minimal_execution_wire() -> dict[str, object]:
    return {
        "batch_count": 2,
        "fqn_batches": [["a", 0], ["b", 1]],
        "node_ids": [0, 8],
        "sources": [
            [0, 0, [[0, [[0, 10]]], [1, [[20, 10]]]]],
        ],
        "version": 1,
    }


def _project_minimal_execution_wire(
    wire: object,
    *,
    targets: tuple[_LocalTarget, ...] = (),
    capacities: dict[int, int] | None = None,
) -> object:
    return project_execution_plan_wire(
        wire,
        expected_node_ids=(0, 8),
        local_node_id=0,
        local_targets=targets,
        node_capacities=capacities or {0: 100, 8: 100},
    )


def test_execution_plan_decoder_rejects_malformed_wire() -> None:
    malformed: list[object] = []
    unknown = _minimal_execution_wire()
    unknown["extra"] = 1
    malformed.append(unknown)
    wrong_version = _minimal_execution_wire()
    wrong_version["version"] = 2
    malformed.append(wrong_version)
    tuple_nodes = _minimal_execution_wire()
    tuple_nodes["node_ids"] = (0, 8)
    malformed.append(tuple_nodes)
    bool_batch = _minimal_execution_wire()
    bool_batch["batch_count"] = True
    malformed.append(bool_batch)
    unsorted_fqns = _minimal_execution_wire()
    unsorted_fqns["fqn_batches"] = [["b", 1], ["a", 0]]
    malformed.append(unsorted_fqns)
    duplicate_sources = _minimal_execution_wire()
    duplicate_sources["sources"] = [
        [0, 0, [[0, [[0, 10]]]]],
        [0, 0, [[1, [[20, 10]]]]],
    ]
    malformed.append(duplicate_sources)
    unsorted_schedules = _minimal_execution_wire()
    unsorted_schedules["sources"] = [
        [0, 0, [[1, [[20, 10]]], [0, [[0, 10]]]]],
    ]
    malformed.append(unsorted_schedules)
    adjacent_ranges = _minimal_execution_wire()
    adjacent_ranges["sources"] = [
        [0, 0, [[0, [[0, 5], [5, 5]]], [1, [[20, 10]]]]],
    ]
    malformed.append(adjacent_ranges)
    cross_batch_overlap = _minimal_execution_wire()
    cross_batch_overlap["sources"] = [
        [0, 0, [[0, [[0, 10]]], [1, [[5, 10]]]]],
    ]
    malformed.append(cross_batch_overlap)
    overflow = _minimal_execution_wire()
    overflow["sources"] = [
        [0, 0, [[0, [[(1 << 64) - 1, 2]]], [1, [[20, 10]]]]],
    ]
    malformed.append(overflow)

    for wire in malformed:
        with pytest.raises(ValueError):
            _project_minimal_execution_wire(wire)


def test_execution_plan_decoder_validates_topology_capacity_and_local_targets() -> None:
    wire = _minimal_execution_wire()

    with pytest.raises(ValueError, match="topology"):
        project_execution_plan_wire(
            wire,
            expected_node_ids=(8, 0),
            local_node_id=0,
            local_targets=(),
            node_capacities={0: 100, 8: 100},
        )
    with pytest.raises(ValueError, match="capacity"):
        _project_minimal_execution_wire(wire, capacities={0: 9, 8: 100})
    with pytest.raises(ValueError, match="absent"):
        _project_minimal_execution_wire(
            wire,
            targets=(_local_target("missing", 0, (0, 1)),),
        )
    with pytest.raises(ValueError, match="no execution-plan owner"):
        _project_minimal_execution_wire(
            wire,
            targets=(_local_target("a", 9, (0, 1)),),
        )
    with pytest.raises(ValueError, match="not covered"):
        _project_minimal_execution_wire(
            wire,
            targets=(_local_target("b", 0, (0, 1)),),
        )
    with pytest.raises(ValueError, match="not covered"):
        _project_minimal_execution_wire(
            wire,
            targets=(_local_target("a", 0, (9, 2)),),
        )


def test_execution_plan_wire_scales_with_fqns_not_repeated_demand_ranges() -> None:
    demand_count = 2_000
    demands = tuple(
        _demand(f"tensor.{index:05d}", (0, index, 1)) for index in range(demand_count)
    )
    assignment = FileAssignment((0, 8), ((0,), ()), (demand_count, 0))
    batches = (PipelineBatch(0, demands),)
    works = build_batch_node_works(batches, assignment, consolidate_gap_bytes=0)

    wire = execution_plan_to_wire(assignment, batches, works)
    encoded = json.dumps(wire, separators=(",", ":"))
    legacy = json.dumps(
        {
            "assignment": assignment.to_dict(),
            "batches": [batch.to_dict() for batch in batches],
        },
        separators=(",", ":"),
    )

    assert len(wire["fqn_batches"]) == demand_count
    assert wire["sources"] == [[0, 0, [[0, [[0, demand_count]]]]]]
    assert len(encoded) < len(legacy) // 2


def test_execution_plan_encoder_validates_owners_once_per_range() -> None:
    demand_count = 2_000
    node_ids = tuple(range(64))
    demands = tuple(
        _demand(
            f"tensor.{index:05d}",
            (index % len(node_ids), index * 4, 1),
        )
        for index in range(demand_count)
    )
    assignment = assign_sources_to_nodes(demands, node_ids)
    batches = (PipelineBatch(0, demands),)
    works = build_batch_node_works(batches, assignment, consolidate_gap_bytes=0)
    owner_call_count = 0
    original_owner_for = FileAssignment.owner_for

    def counted_owner_for(self: FileAssignment, source_rank: int) -> NodeId:
        nonlocal owner_call_count
        owner_call_count += 1
        return original_owner_for(self, source_rank)

    with mock.patch.object(FileAssignment, "owner_for", counted_owner_for):
        execution_plan_to_wire(assignment, batches, works)

    assert owner_call_count <= demand_count * 3 + len(node_ids)


def test_assignment_and_batch_validation_reject_incomplete_plans() -> None:
    demands = (_demand("a", (0, 0, 10)), _demand("b", (1, 0, 5)))
    incomplete = FileAssignment(("a", "b"), ((0,), ()), (10, 0))
    with pytest.raises(ValueError, match="exactly"):
        validate_assignment_coverage(demands, incomplete)

    first = _demand("first", (0, 0, 10))
    second = _demand("second", (0, 5, 10))
    with pytest.raises(ValueError, match="overlap-connected"):
        validate_batch_coverage(
            (first, second),
            (PipelineBatch(0, (first,)), PipelineBatch(1, (second,))),
        )


def test_chunk_splitting_preserves_exact_length() -> None:
    pieces = split_flat_range_into_chunks(50, 220, 100)

    assert tuple(piece.chunk_slot for piece in pieces) == (0, 1, 2)
    assert tuple(piece.offset_in_chunk for piece in pieces) == (50, 0, 0)
    assert tuple(piece.length for piece in pieces) == (50, 100, 70)
    assert sum(piece.length for piece in pieces) == 220
    assert split_flat_range_into_chunks(100, 0, 100) == ()
