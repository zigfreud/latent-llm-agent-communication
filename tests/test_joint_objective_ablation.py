from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from src.core.packet_loss import ComponentAwarePacketLoss, build_terminal_component_masks
from src.pipelines.packet_bridge import build_packet_loss
from src.pipelines.joint_objective_ablation import (
    POLICY, validate_joint_policy, joint_loss_config, joint_checkpoint_selection_key,
)
from src.pipelines.closed_loop_duration import evaluation_splits, evaluate_gate
from src.pipelines.receiver_aware_replay import _lf_sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_joint_identity_value_and_gradient_equal_joint_terms_without_dilution():
    torch.manual_seed(37)
    prediction = torch.randn(4, 2, 10, 5, requires_grad=True)
    target = torch.randn_like(prediction)
    masks = build_terminal_component_masks(torch.tensor([2, 3, 2, 3]), target_positions=10, boundary_positions=2)
    config = dict(lambda_huber=0., lambda_cosine=0., lambda_norm=0.,
                  lambda_symmetric_nce=1., lambda_margin=1.)
    joint = build_packet_loss(joint_loss_config(config, 'joint_only'))(prediction, target, masks)
    expected = joint['joint_symmetric_nce'] + joint['joint_margin_loss']
    assert torch.equal(joint['total_loss'], expected)
    actual_grad = torch.autograd.grad(joint['total_loss'], prediction, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, prediction, retain_graph=True)[0]
    assert torch.equal(actual_grad, expected_grad)
    assert actual_grad[:, :, -2:].abs().sum() > 0  # joint keeps boundary sites
    old = ComponentAwarePacketLoss(**config)(prediction, target, masks)
    regional = build_packet_loss(joint_loss_config(config, 'regional'))(prediction, target, masks)
    assert torch.equal(old['total_loss'], regional['total_loss'])
    assert torch.equal(torch.autograd.grad(old['total_loss'], prediction, retain_graph=True)[0],
                       torch.autograd.grad(regional['total_loss'], prediction)[0])
    for key in ('huber_loss', 'cosine_loss', 'norm_loss'):
        assert torch.equal(old[key], joint[key])


@pytest.mark.parametrize('regions', [[], ['joint', 'joint'], ['unknown']])
def test_invalid_identity_regions_rejected(regions):
    with pytest.raises(ValueError, match='identity_regions'):
        ComponentAwarePacketLoss(identity_regions=regions)


def test_joint_selection_does_not_depend_on_separate_regions():
    metrics = {'regions': {'joint': {'retrieval_top1': .9, 'diagonal_margin_mean': .02}},
               'normalized_residual_rmse': 1.2}
    key = joint_checkpoint_selection_key(metrics, step=128)
    assert key == (.9, .02, -1.2, -128)
    metrics['regions'].update(core={'retrieval_top1': 0}, name={'retrieval_top1': float('nan')})
    assert joint_checkpoint_selection_key(metrics, step=128) == key
    metrics['regions']['joint']['retrieval_top1'] = .95
    assert joint_checkpoint_selection_key(metrics, step=512) > key


def test_frozen_policy_and_no_gate_for_both_arms():
    policy = json.loads((ROOT/'config/H0-017_joint_objective_v1.json').read_text())
    base = _lf_sha256_file(ROOT/'config/LIP-H0-017_closed_loop_trajectory_corrector.yaml')
    for arm in POLICY['arms']:
        validate_joint_policy(policy, base_sha256=base, pilot=False, variant='closed_loop_live', arm=arm)
    for key, value in [('development_gate_allowed', True), ('primary_budget', 1024), ('seed', 4008)]:
        changed = deepcopy(policy)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_joint_policy(changed, base_sha256=base, pilot=False, variant='closed_loop_live', arm='joint_only')
    with pytest.raises(ValueError):
        validate_joint_policy(policy, base_sha256=base, pilot=False, variant='open_loop_zero_live', arm='joint_only')
    datasets = {split: object() for split in ('train', *evaluation_splits(policy))}
    def forbidden(**kwargs):
        pytest.fail('gate evaluator must not run')
    assert evaluate_gate(forbidden, duration=policy, datasets=datasets) is None
    original = {'component_weights': {'core': .45, 'name': .45, 'boundary': .1}, 'lambda_margin': 1.}
    changed = joint_loss_config(original, 'joint_only')
    assert changed.pop('identity_regions') == ['joint']
    assert changed == original and 'identity_regions' not in original
