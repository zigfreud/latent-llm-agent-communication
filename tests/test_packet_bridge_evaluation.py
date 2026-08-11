import pytest
import torch

from src.core.packet_loss import build_terminal_component_masks
from src.evaluation.packet_bridge import (
    checkpoint_selection_key,
    normalized_transport_recovery,
    summarize_multi_replica_development_gate,
    summarize_packet_latent_metrics,
    summarize_replica_development_gate,
)


def _separated_packets(task_count=8):
    target = torch.zeros(task_count, 1, 8, task_count)
    for index in range(task_count):
        target[index, :, :, index] = 1.0
    masks = build_terminal_component_masks(
        torch.full((task_count,), 2),
        target_positions=8,
        boundary_positions=2,
    )
    return target, masks


def test_latent_metrics_retrieve_matching_packets_in_every_causal_region():
    target, masks = _separated_packets()

    report = summarize_packet_latent_metrics(target, target, masks)

    assert report["normalized_residual_rmse"] == 0.0
    for region in ("joint", "core", "name"):
        assert report["regions"][region]["retrieval_top1"] == 1.0
        assert report["regions"][region]["diagonal_margin_mean"] > 0.0


def test_checkpoint_selection_protects_weakest_component_before_mean_score():
    target, masks = _separated_packets()
    strong = summarize_packet_latent_metrics(target, target, masks)
    weak = summarize_packet_latent_metrics(target.roll(1, dims=0), target, masks)

    assert checkpoint_selection_key(strong, step=64) > checkpoint_selection_key(
        weak, step=32
    )


def test_development_gate_uses_one_holm_family_and_replica_threshold():
    target, masks = _separated_packets()
    metrics = summarize_packet_latent_metrics(target, target, masks)

    replica = summarize_replica_development_gate(metrics)
    combined = summarize_multi_replica_development_gate([replica, replica, {"passed": False}])

    assert replica["passed"] is True
    assert all(test["p_value_holm"] <= 0.05 for test in replica["family"])
    assert combined["passed"] is True
    assert combined["passing_replicas"] == 2


def test_normalized_transport_recovery_uses_causal_and_pragmatic_denominators():
    recovery = normalized_transport_recovery(
        learned_matched=0.60,
        learned_shuffled=0.10,
        oracle_matched=0.90,
        oracle_shuffled=0.00,
        text=0.80,
        neutral=0.20,
    )

    assert recovery["identity_recovery_ratio"] == pytest.approx(0.5 / 0.9)
    assert recovery["text_gain_recovery_ratio"] == pytest.approx(2 / 3)


def test_recovery_ratio_is_undefined_when_reference_effect_is_zero():
    recovery = normalized_transport_recovery(
        learned_matched=0.0,
        learned_shuffled=0.0,
        oracle_matched=0.0,
        oracle_shuffled=0.0,
        text=0.0,
        neutral=0.0,
    )

    assert recovery["identity_recovery_ratio"] is None
    assert recovery["text_gain_recovery_ratio"] is None
