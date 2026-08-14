"""Deterministic balanced batches that colocate frozen hard-negative pairs."""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping, Sequence


def hard_negative_mapping(candidate_bank: Mapping, *, label: str) -> dict[str, str]:
    """Extract one complete anchor -> global-hardest-negative mapping."""

    bank = candidate_bank.get(label)
    if not isinstance(bank, Mapping):
        raise ValueError(f"candidate bank does not contain {label!r}")
    rows = bank.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate bank rows must be a non-empty list")
    mapping: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("candidate bank rows must be objects")
        anchor = str(row.get("anchor_task_id", ""))
        negative = str(row.get("global_hardest_task_id", ""))
        if not anchor or not negative:
            raise ValueError("candidate bank row is missing a task identity")
        if anchor == negative:
            raise ValueError("a global hardest negative cannot equal its anchor")
        if anchor in mapping:
            raise ValueError(f"duplicate candidate-bank anchor {anchor!r}")
        mapping[anchor] = negative
    if set(mapping.values()) - set(mapping):
        raise ValueError("candidate bank references tasks outside its anchor set")
    return mapping


def hard_negative_coverage(
    batches: Sequence[Sequence[int]], hardest_indices: Sequence[int]
) -> tuple[int, float]:
    """Count anchors whose frozen hardest negative shares their batch."""

    assignment: dict[int, int] = {}
    for batch_index, batch in enumerate(batches):
        for item in batch:
            if int(item) in assignment:
                raise ValueError("batch plan contains a duplicate dataset index")
            assignment[int(item)] = batch_index
    if set(assignment) != set(range(len(hardest_indices))):
        raise ValueError("batch plan must cover every dataset index exactly once")
    count = sum(
        assignment[index] == assignment[int(negative)]
        for index, negative in enumerate(hardest_indices)
    )
    return int(count), float(count / len(hardest_indices))


def _swap_delta(
    assignment: list[int],
    hardest: list[int],
    reverse: list[list[int]],
    left: int,
    right: int,
) -> int:
    left_group = assignment[left]
    right_group = assignment[right]
    if left_group == right_group:
        return 0
    affected = {left, right, *reverse[left], *reverse[right]}
    before = sum(assignment[index] == assignment[hardest[index]] for index in affected)
    assignment[left], assignment[right] = right_group, left_group
    after = sum(assignment[index] == assignment[hardest[index]] for index in affected)
    assignment[left], assignment[right] = left_group, right_group
    return int(after - before)


def _improve_partition(
    groups: list[list[int]], hardest: list[int], *, max_swaps: int
) -> list[list[int]]:
    count = len(hardest)
    assignment = [0] * count
    positions = [0] * count
    for group_index, group in enumerate(groups):
        for position, item in enumerate(group):
            assignment[item] = group_index
            positions[item] = position
    reverse = [[] for _ in range(count)]
    for anchor, negative in enumerate(hardest):
        reverse[negative].append(anchor)

    for _ in range(max_swaps):
        best: tuple[int, int, int] | None = None
        for anchor in range(count):
            negative = hardest[anchor]
            anchor_group = assignment[anchor]
            negative_group = assignment[negative]
            if anchor_group == negative_group:
                continue
            candidates = (
                (negative, replacement) for replacement in groups[anchor_group]
            )
            alternatives = (
                (anchor, replacement) for replacement in groups[negative_group]
            )
            for left, right in (*candidates, *alternatives):
                delta = _swap_delta(
                    assignment, hardest, reverse, int(left), int(right)
                )
                proposal = (delta, -int(left), -int(right))
                if best is None or proposal > best:
                    best = proposal
        if best is None or best[0] <= 0:
            break
        _, negative_left, negative_right = best
        left, right = -negative_left, -negative_right
        left_group, right_group = assignment[left], assignment[right]
        left_position, right_position = positions[left], positions[right]
        groups[left_group][left_position] = right
        groups[right_group][right_position] = left
        positions[left], positions[right] = right_position, left_position
        assignment[left], assignment[right] = right_group, left_group
    return groups


def build_balanced_hard_negative_batches(
    task_ids: Sequence[str],
    hardest_by_task: Mapping[str, str],
    *,
    batch_size: int,
    seed: int,
    restarts: int,
    max_swaps: int,
) -> tuple[list[list[int]], dict]:
    """Optimize a balanced partition while preserving one exposure per epoch."""

    dataset_ids = [str(task_id) for task_id in task_ids]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("training task identities must be unique")
    if set(dataset_ids) != set(hardest_by_task):
        raise ValueError("candidate-bank anchors differ from training task identities")
    if batch_size < 2 or len(dataset_ids) % batch_size:
        raise ValueError("hard-negative batching requires equal complete batches")
    if restarts <= 0 or max_swaps <= 0:
        raise ValueError("hard-negative partition search parameters must be positive")
    dataset_index_by_id = {
        task_id: index for index, task_id in enumerate(dataset_ids)
    }
    ids = sorted(dataset_ids)
    index_by_id = {task_id: index for index, task_id in enumerate(ids)}
    hardest = [index_by_id[str(hardest_by_task[task_id])] for task_id in ids]
    if any(index == negative for index, negative in enumerate(hardest)):
        raise ValueError("candidate bank contains a self-negative")

    best_batches: list[list[int]] | None = None
    best_key: tuple[int, tuple[tuple[int, ...], ...]] | None = None
    for restart in range(restarts):
        order = list(range(len(ids)))
        random.Random(int(seed) + restart * 104729).shuffle(order)
        groups = [
            order[start : start + batch_size]
            for start in range(0, len(order), batch_size)
        ]
        groups = _improve_partition(groups, hardest, max_swaps=max_swaps)
        canonical = tuple(sorted(tuple(sorted(group)) for group in groups))
        covered, _ = hard_negative_coverage(canonical, hardest)
        key = (covered, tuple(tuple(-item for item in group) for group in canonical))
        if best_key is None or key > best_key:
            best_key = key
            best_batches = [list(group) for group in canonical]
    assert best_batches is not None
    covered, coverage = hard_negative_coverage(best_batches, hardest)
    dataset_batches = [
        [dataset_index_by_id[ids[index]] for index in batch]
        for batch in best_batches
    ]
    return dataset_batches, {
        "task_count": len(ids),
        "batch_size": int(batch_size),
        "batches_per_epoch": len(best_batches),
        "global_hardest_covered_anchors": covered,
        "global_hardest_coverage": coverage,
        "random_partition_expected_coverage": (batch_size - 1) / (len(ids) - 1),
        "one_exposure_per_task_per_epoch": True,
        "partition_seed": int(seed),
        "search_restarts": int(restarts),
        "maximum_improving_swaps_per_restart": int(max_swaps),
        "batches": [
            {
                "indices": [dataset_index_by_id[ids[index]] for index in batch],
                "task_ids": [ids[index] for index in batch],
            }
            for batch in best_batches
        ],
    }


class EpochShuffledBatchSampler:
    """Reuse one frozen partition while shuffling batch and row order by epoch."""

    def __init__(self, batches: Sequence[Sequence[int]], *, seed: int) -> None:
        self.batches = [list(map(int, batch)) for batch in batches]
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1009)
        batches = [list(batch) for batch in self.batches]
        rng.shuffle(batches)
        for batch in batches:
            rng.shuffle(batch)
            yield batch
        self.epoch += 1

    def __len__(self) -> int:
        return len(self.batches)
