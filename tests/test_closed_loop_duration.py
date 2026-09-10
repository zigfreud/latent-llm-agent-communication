from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from src.pipelines.closed_loop_duration import POLICY, validate_duration_policy, evaluation_splits, evaluate_gate, save_duration_budget
from src.pipelines.receiver_aware_replay import _lf_sha256_file

ROOT = Path(__file__).resolve().parents[1]

def test_duration_policy_matches_frozen_base_and_rejects_gate_or_budget_drift():
    path = ROOT/'config/H0-017_duration_diagnostic_v1.json'
    policy = json.loads(path.read_text())
    base = _lf_sha256_file(ROOT/'config/LIP-H0-017_closed_loop_trajectory_corrector.yaml')
    validate_duration_policy(policy, base_sha256=base, pilot=False)
    for key,value in [('primary_budget',1024),('development_gate_allowed',True),('seed',4008)]:
        changed = deepcopy(policy)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_duration_policy(changed, base_sha256=base, pilot=False)
    with pytest.raises(ValueError):
        validate_duration_policy(policy, base_sha256=base, pilot=True)

def test_duration_does_not_construct_or_evaluate_gate_dataset():
    calls = []
    def evaluate(**kwargs):
        calls.append(kwargs)
        return 'evaluated'
    datasets = {split: object() for split in ('train', *evaluation_splits(POLICY))}
    assert evaluate_gate(evaluate, duration=POLICY, datasets=datasets) is None
    assert calls == []
    with pytest.raises(ValueError):
        evaluate_gate(evaluate, duration=POLICY, datasets={**datasets,'development_gate':object()})
    gate = object()
    assert evaluate_gate(evaluate, duration=None, datasets={'development_gate':gate}) == 'evaluated'
    assert calls == [{'dataset':gate}]

def test_budget_saving_keeps_current_weights_rng_and_optimizer_trajectory(tmp_path):
    model = torch.nn.Linear(2,2)
    optimizer = torch.optim.AdamW(model.parameters())
    x = torch.ones(3,2)
    model(x).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()
    best = tmp_path/'best.pt'
    torch.save({'step':104,'corrector_state':model.state_dict(),'selection_metrics':{'score':1}},best)
    with torch.no_grad():
        model.weight.add_(1)
    before = deepcopy(model.state_dict())
    opt_before = deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    result = save_duration_budget(output_dir=tmp_path,corrector=model,step=128,
        selection={'score':2},best_path=best,best_step=104)
    assert result['selected']['step'] == 104
    assert result['endpoint']['metrics'] == {'score':2}
    endpoint = torch.load(tmp_path/'endpoint_128.pt',weights_only=True)
    selected = torch.load(tmp_path/'selected_through_128.pt',weights_only=True)
    assert torch.equal(endpoint['corrector_state']['weight'],before['weight'])
    assert not torch.equal(selected['corrector_state']['weight'],before['weight'])
    assert torch.equal(model.weight,before['weight']) and torch.equal(torch.get_rng_state(),rng)
    for key,state in opt_before['state'].items():
        for name,tensor in state.items():
            assert torch.equal(tensor,optimizer.state_dict()['state'][key][name])
    with pytest.raises(FileExistsError):
        save_duration_budget(output_dir=tmp_path,corrector=model,step=128,
            selection={},best_path=best,best_step=104)
