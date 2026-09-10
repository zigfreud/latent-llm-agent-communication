"""Explicit exploratory duration policy; never grants the frozen screen's gate."""
from pathlib import Path
import shutil
import torch

POLICY = {
    'diagnostic_id': 'H0-017-duration-v1',
    'claim_status': 'exploratory_duration_only',
    'base_experiment_config_sha256': 'c0df4a8db4c672a6bdd7437fe9fb0bc2c4a157a9d12914acb0d2b076872881ea',
    'budgets': [128, 256, 512],
    'primary_budget': 512,
    'validation_interval': 8,
    'seed': 4007,
    'evaluation_splits': ['development_selection'],
    'development_gate_allowed': False,
    'confirmation_allowed': False,
    'resume_supported': False,
}


def validate_duration_policy(policy, *, base_sha256, pilot):
    if pilot or policy != POLICY or base_sha256 != POLICY['base_experiment_config_sha256']:
        raise ValueError('duration diagnostic policy or frozen base differs')


def evaluation_splits(duration):
    return ('development_selection',) if duration else ('development_selection', 'development_gate')


def evaluate_gate(evaluator, *, duration, datasets, **kwargs):
    if duration:
        if 'development_gate' in datasets or 'confirmation' in datasets:
            raise ValueError('duration diagnostic must not construct gate datasets')
        return None
    return evaluator(dataset=datasets['development_gate'], **kwargs)


def save_duration_budget(*, output_dir, corrector, step, selection, best_path, best_step):
    """Snapshot without evaluating again, resetting the optimizer or consuming RNG."""
    endpoint = Path(output_dir) / f'endpoint_{step}.pt'
    selected = Path(output_dir) / f'selected_through_{step}.pt'
    if endpoint.exists() or selected.exists():
        raise FileExistsError('refusing to overwrite a duration snapshot')
    torch.save({'corrector_state': corrector.state_dict(), 'step': step,
                'selection_metrics': selection, 'resume_supported': False}, endpoint)
    shutil.copyfile(best_path, selected)
    best = torch.load(best_path, map_location='cpu', weights_only=True)
    if best['step'] != best_step or best_step > step:
        raise ValueError('selected checkpoint is outside budget')
    return {'budget': step, 'endpoint': {'checkpoint': endpoint.name, 'step': step, 'metrics': selection},
            'selected': {'checkpoint': selected.name, 'step': best_step, 'metrics': best['selection_metrics']}}
