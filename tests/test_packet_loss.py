import pytest
import torch

from src.core.packet_loss import (
    ComponentAwarePacketLoss,
    build_terminal_component_masks,
)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"temperature": 0.0}, "temperature"),
        ({"margin_target": -0.1}, "margin_target"),
        ({"lambda_huber": -1.0}, "weights"),
    ],
)
def test_packet_loss_rejects_invalid_hyperparameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ComponentAwarePacketLoss(**kwargs)


def test_terminal_masks_follow_per_task_function_name_length():
    masks = build_terminal_component_masks(
        torch.tensor([2, 3]),
        target_positions=24,
        boundary_positions=6,
    )

    assert masks["core"].sum(dim=1).tolist() == [16, 15]
    assert masks["name"].sum(dim=1).tolist() == [2, 3]
    assert masks["boundary"].sum(dim=1).tolist() == [6, 6]
    coverage = masks["core"].int() + masks["name"].int() + masks["boundary"].int()
    assert torch.equal(coverage, torch.ones_like(coverage))


def test_component_balancing_prevents_long_core_from_dominating_short_name():
    masks = build_terminal_component_masks(
        torch.tensor([2, 2]),
        target_positions=8,
        boundary_positions=2,
    )
    target = torch.zeros(2, 1, 8, 3)
    core_error = target.clone()
    name_error = target.clone()
    core_error[:, :, :4, :] = 1.0
    name_error[:, :, 4:6, :] = 1.0
    criterion = ComponentAwarePacketLoss(
        lambda_huber=1.0,
        lambda_cosine=0.0,
        lambda_symmetric_nce=0.0,
        lambda_margin=0.0,
        lambda_norm=0.0,
        component_weights={"core": 0.5, "name": 0.5, "boundary": 0.0},
    )

    core_loss = criterion(core_error, target, masks)["total_loss"]
    name_loss = criterion(name_error, target, masks)["total_loss"]

    assert core_loss.item() == pytest.approx(name_loss.item())


def test_contrastive_terms_prefer_matching_task_packets():
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(4, 2, 8, 5, generator=generator)
    masks = build_terminal_component_masks(
        torch.tensor([2, 2, 2, 2]),
        target_positions=8,
        boundary_positions=2,
    )
    criterion = ComponentAwarePacketLoss(
        lambda_huber=0.0,
        lambda_cosine=0.0,
        lambda_symmetric_nce=1.0,
        lambda_margin=0.1,
        lambda_norm=0.0,
    )

    matched = criterion(target.clone(), target, masks)
    shuffled = criterion(target.roll(1, dims=0), target, masks)

    assert matched["total_loss"] < shuffled["total_loss"]
    assert matched["joint_retrieval_top1"].item() == 1.0
    assert shuffled["joint_retrieval_top1"].item() == 0.0
    assert matched["joint_diagonal_margin"] > shuffled["joint_diagonal_margin"]


def test_packet_loss_is_finite_and_differentiable():
    prediction = torch.randn(3, 2, 8, 5, requires_grad=True)
    target = torch.randn(3, 2, 8, 5)
    masks = build_terminal_component_masks(
        torch.tensor([2, 3, 2]),
        target_positions=8,
        boundary_positions=2,
    )
    criterion = ComponentAwarePacketLoss()

    metrics = criterion(prediction, target, masks)
    metrics["total_loss"].backward()

    assert torch.isfinite(metrics["total_loss"])
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_disabled_norm_term_is_not_evaluated_for_zero_norm_target():
    prediction = torch.full((2, 1, 8, 5), 1e20)
    target = torch.zeros_like(prediction)
    masks = build_terminal_component_masks(
        torch.tensor([2, 2]),
        target_positions=8,
        boundary_positions=2,
    )
    criterion = ComponentAwarePacketLoss(
        lambda_huber=0.0,
        lambda_cosine=0.0,
        lambda_symmetric_nce=0.0,
        lambda_margin=0.0,
        lambda_norm=0.0,
    )

    metrics = criterion(prediction, target, masks)

    assert metrics["norm_loss"].item() == 0.0
    assert torch.isfinite(metrics["total_loss"])


def test_terminal_masks_reject_names_that_consume_the_entire_core():
    with pytest.raises(ValueError, match="leave at least one terminal core"):
        build_terminal_component_masks(
            torch.tensor([6]),
            target_positions=8,
            boundary_positions=2,
        )
