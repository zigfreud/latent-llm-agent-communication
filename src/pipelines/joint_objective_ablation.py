"""Frozen development-only comparison of regional versus joint identity loss."""
from copy import deepcopy
import math

POLICY = {
    'diagnostic_id': 'H0-017-joint-objective-v1',
    'claim_status': 'exploratory_joint_objective_only',
    'base_experiment_config_sha256': 'c0df4a8db4c672a6bdd7437fe9fb0bc2c4a157a9d12914acb0d2b076872881ea',
    'arms': {'regional': ['joint', 'core', 'name'], 'joint_only': ['joint']},
    'variant': 'closed_loop_live',
    'joint_definition': 'full_packet_including_boundary',
    'checkpoint_selection': ['joint_retrieval_top1', 'joint_diagonal_margin_mean', '-normalized_residual_rmse', '-step'],
    'budgets': [128, 256, 512],
    'primary_budget': 512,
    'validation_interval': 8,
    'seed': 4007,
    'evaluation_splits': ['development_selection'],
    'development_gate_allowed': False,
    'confirmation_allowed': False,
    'resume_supported': False,
}


def validate_joint_policy(policy, *, base_sha256, pilot, variant, arm):
    if (policy != POLICY or base_sha256 != POLICY['base_experiment_config_sha256']
            or pilot or variant != POLICY['variant'] or arm not in POLICY['arms']):
        raise ValueError('joint objective ablation policy, frozen base, variant or arm differs')


def joint_loss_config(config, arm):
    result = deepcopy(config)
    result['identity_regions'] = list(POLICY['arms'][arm])
    return result


def joint_checkpoint_selection_key(metrics, *, step):
    joint = metrics['regions']['joint']
    values = (float(joint['retrieval_top1']), float(joint['diagonal_margin_mean']),
              -float(metrics['normalized_residual_rmse']))
    if not all(math.isfinite(value) for value in values):
        raise ValueError('joint checkpoint metrics must be finite')
    return (*values, -int(step))
