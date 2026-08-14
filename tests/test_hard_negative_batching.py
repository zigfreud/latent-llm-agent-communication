import pytest

from src.core.hard_negative_batching import (
    EpochShuffledBatchSampler,
    build_balanced_hard_negative_batches,
    hard_negative_coverage,
    hard_negative_mapping,
)


def _ring_bank(count: int):
    return {
        "H0_013": {
            "rows": [
                {
                    "anchor_task_id": f"task-{index}",
                    "global_hardest_task_id": f"task-{(index + 1) % count}",
                }
                for index in range(count)
            ]
        }
    }


def test_balanced_plan_preserves_exposure_and_improves_ring_coverage():
    task_ids = [f"task-{index}" for index in range(16)]
    mapping = hard_negative_mapping(_ring_bank(16), label="H0_013")
    batches, metadata = build_balanced_hard_negative_batches(
        task_ids,
        mapping,
        batch_size=4,
        seed=4007,
        restarts=4,
        max_swaps=64,
    )

    assert sorted(index for batch in batches for index in batch) == list(range(16))
    assert all(len(batch) == 4 for batch in batches)
    assert metadata["one_exposure_per_task_per_epoch"] is True
    assert metadata["global_hardest_coverage"] > 3 / 15


def test_coverage_rejects_duplicate_or_missing_indices():
    with pytest.raises(ValueError, match="duplicate"):
        hard_negative_coverage([[0, 1], [1, 2]], [1, 0, 1, 2])


def test_candidate_mapping_rejects_external_negative():
    bank = _ring_bank(4)
    bank["H0_013"]["rows"][0]["global_hardest_task_id"] = "outside"
    with pytest.raises(ValueError, match="outside"):
        hard_negative_mapping(bank, label="H0_013")


def test_epoch_sampler_changes_order_without_changing_membership():
    sampler = EpochShuffledBatchSampler([[0, 1], [2, 3], [4, 5]], seed=17)
    first = list(iter(sampler))
    second = list(iter(sampler))

    assert first != second
    assert sorted(index for batch in first for index in batch) == list(range(6))
    assert sorted(index for batch in second for index in batch) == list(range(6))
