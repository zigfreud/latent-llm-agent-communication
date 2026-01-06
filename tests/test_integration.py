import os
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.pipelines.trainer import ShardDataset, load_sharded_dataset
from src.core.models import LIPAdapter
from src.core.loss import HybridContrastiveLoss


def _write_mock_shard(path, num_rows=2, src_dim=2048, tgt_dim=4096):
    data = []
    for _ in range(num_rows):
        data.append({
            "src_vector": torch.randn(1, src_dim),
            "tgt_vector": torch.randn(1, tgt_dim),
        })
    torch.save(data, path)


@pytest.fixture
def shard_dir(tmp_path):
    shard_path = tmp_path / "shards"
    shard_path.mkdir()
    _write_mock_shard(shard_path / "shard_0.pt")
    _write_mock_shard(shard_path / "shard_1.pt")
    return shard_path


def test_shard_dataset_loads(shard_dir):
    dataset = load_sharded_dataset(str(shard_dir))
    assert len(dataset) == 4
    src, tgt = dataset[0]
    assert src.shape == (2048,)
    assert tgt.shape == (4096,)


def test_single_train_step_forward_backward(shard_dir):
    dataset = load_sharded_dataset(str(shard_dir))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True)

    model = LIPAdapter()
    criterion = HybridContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    src_batch, tgt_batch = next(iter(loader))
    output = model(src_batch)
    loss, acc = criterion(output, tgt_batch)
    loss.backward()
    optimizer.step()

    assert output.shape == (2, 4096)
    assert torch.isfinite(loss).all()
    assert 0.0 <= acc.item() <= 1.0


def test_inference_output_shape_after_minimal_training(shard_dir):
    dataset = load_sharded_dataset(str(shard_dir))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True)

    model = LIPAdapter()
    criterion = HybridContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    src_batch, tgt_batch = next(iter(loader))
    loss, _ = criterion(model(src_batch), tgt_batch)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        inference_out = model(torch.randn(1, 2048))
    assert inference_out.shape == (1, 4096)
