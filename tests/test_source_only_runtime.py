import json

import pytest
import torch

from src.scripts.run_source_only_probe import (
    load_existing_results,
    random_norm_matched,
    stable_seed,
)


def test_random_control_matches_each_reference_norm():
    reference = torch.tensor([[3.0, 4.0, 0.0]])
    control = random_norm_matched(reference, seed=12)
    assert control.norm().item() == pytest.approx(reference.norm().item())
    assert torch.equal(control, random_norm_matched(reference, seed=12))


def test_stable_seed_is_deterministic_and_order_sensitive():
    assert stable_seed(1, 2, 3) == stable_seed(1, 2, 3)
    assert stable_seed(1, 2, 3) != stable_seed(3, 2, 1)


def test_resume_reader_rejects_duplicate_design_keys(tmp_path):
    row = {
        "task_id": "a",
        "condition": "source_latent",
        "training_seed": 41,
        "generation_seed": 101,
    }
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate result key"):
        load_existing_results(path)
