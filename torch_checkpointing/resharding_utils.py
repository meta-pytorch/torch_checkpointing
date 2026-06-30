"""
Utility helpers shared by resharding implementations.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from .resharding import LoadPlan
from .types import NestedPath

logger: logging.Logger = logging.getLogger(__name__)

_V = TypeVar("_V")


def get_fqn_from_nested_path(path: NestedPath) -> str:
    """Convert NestedPath to fully qualified name string.

    Args:
        path: NestedPath to convert. Empty path maps to the root FQN.

    Returns:
        Dot-separated FQN string.
    """
    return ".".join(str(component) for component in path)


def convert_nested_path_dict_to_fqn(
    path_dict: dict[NestedPath, _V],
) -> tuple[dict[str, _V], dict[str, NestedPath]]:
    """Convert a dict with NestedPath keys to a dict with FQN string keys.

    Args:
        path_dict: Dictionary with NestedPath keys.

    Returns:
        A tuple of:
            - Dictionary with FQN string keys mapped to original values.
            - Dictionary mapping FQN strings back to their NestedPath keys.

    Raises:
        ValueError: If two different NestedPaths produce the same FQN,
            indicating a collision that would cause data loss.
    """
    result: dict[str, _V] = {}
    fqn_to_path: dict[str, NestedPath] = {}

    for path, value in path_dict.items():
        fqn = get_fqn_from_nested_path(path)

        if fqn in fqn_to_path:
            raise ValueError(
                f"FQN collision detected: two different NestedPaths produce the same FQN '{fqn}'. "
                f"Path 1: {fqn_to_path[fqn]}, Path 2: {path}"
            )

        fqn_to_path[fqn] = path
        result[fqn] = value

    return result, fqn_to_path


def deduplicate_source_chunks(
    input_plan: dict[str, list[LoadPlan]],
) -> tuple[dict[str, list[LoadPlan]], set[int]]:
    """Deduplicate source chunks by selecting the lowest source rank.

    When source data is replicated across multiple ranks, the lowest rank is the
    canonical source for each target chunk, so the selected set is globally
    deterministic via min-rank-per-slice selection.
    """
    # Step 1: Build a mapping of (target_fqn, offsets) -> possible source ranks
    possible_ranks_map: dict[tuple[str, tuple[int, ...]], set[int]] = {}
    load_plans_map: dict[tuple[str, tuple[int, ...], int], LoadPlan] = {}

    for target_fqn, load_plans in input_plan.items():
        for lp in load_plans:
            key = (target_fqn, lp.offsets)
            if key not in possible_ranks_map:
                possible_ranks_map[key] = set()
            possible_ranks_map[key].add(lp.src_rank)
            load_plans_map[(target_fqn, lp.offsets, lp.src_rank)] = lp

    # Step 2: Select the lowest source rank for each target chunk.
    assigned_keys: dict[tuple[str, tuple[int, ...]], int] = {}
    selected_ranks: set[int] = set()
    for key, ranks in possible_ranks_map.items():
        rank = min(ranks)
        assigned_keys[key] = rank
        selected_ranks.add(rank)

    # Step 3: Build the final result using the assigned ranks.
    optimized_result: dict[str, list[LoadPlan]] = {}

    for key, assigned_rank in assigned_keys.items():
        target_fqn, offsets = key

        # Add the load plan to the result
        if target_fqn not in optimized_result:
            optimized_result[target_fqn] = []

        optimized_result[target_fqn].append(
            load_plans_map[(target_fqn, offsets, assigned_rank)]
        )

    logger.info(f"Load plan generated with {len(selected_ranks)} ranks")
    return optimized_result, selected_ranks
