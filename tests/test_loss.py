import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core.loss import HybridContrastiveLoss


def test_default_loss_returns_legacy_tuple_and_preserves_formula():
    pred = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 8)
    criterion = HybridContrastiveLoss(temperature=0.1, lambda_mse=0.35)

    result = criterion(pred, target)

    assert isinstance(result, tuple)
    assert len(result) == 4
    total_loss, nce_loss, mse_loss, acc = result
    assert torch.allclose(total_loss, nce_loss + (0.35 * mse_loss))
    assert torch.isfinite(total_loss)
    assert 0.0 <= acc.item() <= 1.0


def test_extended_loss_returns_finite_scalar_and_backward_works():
    pred = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 8)
    criterion = HybridContrastiveLoss(
        temperature=0.1,
        lambda_mse=0.35,
        lambda_reverse_nce=1.0,
        lambda_margin=0.1,
        lambda_norm=0.05,
    )

    result = criterion(pred, target)
    loss = result["total_loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_reverse_nce_can_be_enabled_without_shape_errors():
    pred = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    criterion = HybridContrastiveLoss(lambda_reverse_nce=1.0)

    result = criterion(pred, target)

    assert result["reverse_nce_loss"].shape == ()
    assert torch.isfinite(result["reverse_nce_loss"])


def test_margin_loss_positive_when_hard_negative_beats_positive():
    target = torch.eye(2)
    pred = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    criterion = HybridContrastiveLoss(lambda_margin=1.0, margin_target=0.05)

    result = criterion(pred, target)

    assert result["margin_loss"].item() > 0.0
    assert result["diagonal_margin_mean"].item() < 0.5


def test_margin_loss_near_zero_when_positive_exceeds_margin_target():
    target = torch.eye(2)
    pred = target.clone()
    criterion = HybridContrastiveLoss(lambda_margin=1.0, margin_target=0.05)

    result = criterion(pred, target)

    assert result["margin_loss"].item() == 0.0
    assert result["diagonal_margin_mean"].item() > 0.05


def test_norm_loss_is_lower_when_prediction_norm_matches_target_norm():
    target = torch.eye(2)
    matched_pred = target.clone()
    scaled_pred = target * 2.0
    criterion = HybridContrastiveLoss(lambda_norm=1.0)

    matched = criterion(matched_pred, target)
    scaled = criterion(scaled_pred, target)

    assert matched["norm_loss"].item() < scaled["norm_loss"].item()
    assert matched["norm_loss"].item() == 0.0
