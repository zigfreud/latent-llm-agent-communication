from pathlib import Path

import pytest

from src.scripts.run_multiseed_training import run_multiseed


def test_multiseed_runner_refuses_existing_seed_directory(tmp_path):
    run_dir = tmp_path / "runs" / "seed-41"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty training directories"):
        run_multiseed(
            Path("config/LIP-PROTO-001_multiseed_training.yaml"),
            [41],
            tmp_path / "runs",
        )


def test_multiseed_runner_requires_unique_seeds(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        run_multiseed(
            Path("config/LIP-PROTO-001_multiseed_training.yaml"),
            [41, 41],
            tmp_path / "runs",
        )
