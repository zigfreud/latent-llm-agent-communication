from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.task_contract import public_prompt, interface_diagnostic, validate_registry, validate_grid, evaluate
from src.pipelines.oracle_experiment import task_sha256

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    policy = json.loads((ROOT / 'config/H0-017_task_contract_calibration_v1.json').read_text())
    registry = json.loads((ROOT / 'config/H0-017_task_contract_v2.json').read_text())
    return policy, registry


def test_all_frozen_signatures_accept_original_calls_without_execution():
    policy, registry = fixture()
    validate_registry(policy, registry)
    assert len(registry['tasks']) == 32
    assert len(policy['review_flagged_task_ids']) == 7


def test_public_prompt_does_not_read_hidden_tests_reference_or_old_outputs():
    _, registry = fixture()
    for task in registry['tasks']:
        changed = deepcopy(task)
        changed['original_task']['test_list'] = ['PRIVATE TEST SENTINEL']
        changed['reference_code_sha256'] = 'PRIVATE REFERENCE SENTINEL'
        changed['original_text_output_token_ids'] = [999999]
        for condition in ['text_original', 'text_signature', 'text_explicit']:
            assert public_prompt(registry, task, condition) == public_prompt(registry, changed, condition)


def test_original_and_signature_conditions_preserve_original_statement():
    _, registry = fixture()
    for task in registry['tasks']:
        original = task['original_task']['prompt']
        assert public_prompt(registry, task, 'text_original') == original
        assert public_prompt(registry, task, 'text_signature') == original + '\n\nRequired signature: ' + task['signature'] + '.'


@pytest.mark.parametrize('code,expected', [
    ('def f(a): return a', 'call_shape_mismatch'),
    ('def f(a,b): return a + "wrong type"', 'calls_bind'),
    ('def f(a,b=1): pass', 'calls_bind'),
    ('def f(a,*,b): pass', 'call_shape_mismatch'),
    ('def f(*args): pass', 'calls_bind'),
    ('def g(a,b): pass\nf=g', 'unknown_declaration'),
    ('def f(', 'syntax_error'),
])
def test_static_interface_does_not_execute_or_confuse_body_errors(code, expected):
    task = {'entry_point': 'f', 'test_list': ['assert f(1,2)==3']}
    assert interface_diagnostic(code, task)['status'] == expected


@pytest.mark.parametrize('field', ['signature', 'test_list', 'cohort', 'stratum'])
def test_registry_mutations_rejected(field):
    policy, registry = fixture()
    if field == 'signature': registry['tasks'][0]['signature'] = 'radix_sort(a,b)'
    if field == 'test_list': registry['tasks'][0]['original_task']['test_list'] = ['assert True']
    if field == 'cohort': policy['task_ids'].pop()
    if field == 'stratum': policy['review_flagged_task_ids'].pop()
    with pytest.raises(ValueError): validate_registry(policy, registry)


def make_grid():
    policy, registry = fixture()
    for t in registry['tasks']:
        t['reference_code_sha256'] = hashlib.sha256(b'pass').hexdigest()
    policy['registry_sha256'] = task_sha256(registry)
    rows = [{'task_id': t['task_id'], 'condition': c, 'task_spec': deepcopy(t['original_task']),
             'public_prompt': None if c == 'reference_source' else public_prompt(registry, t, c), 'output_text': 'pass'}
            for c in policy['conditions'] for t in registry['tasks']]
    metadata = {'policy': policy, 'registry': registry}
    tasks = {t['task_id']: t for t in registry['tasks']}
    for row in rows:
        row['metadata_sha256'] = task_sha256(metadata)
        row['input_ids_sha256'] = tasks[row['task_id']]['original_text_input_ids_sha256']
    return policy, registry, rows, metadata


@pytest.mark.parametrize('change', ['none', 'missing', 'duplicate', 'prompt', 'tests', 'reference', 'tokens', 'metadata'])
def test_grid_is_complete_and_bound_to_public_and_private_inputs(change):
    policy, registry, rows, metadata = make_grid()
    if change == 'missing': rows.pop()
    if change == 'duplicate': rows.append(rows[0])
    if change == 'prompt': rows[0]['public_prompt'] += ' answer leaked'
    if change == 'tests': rows[0]['task_spec']['test_list'] = ['assert True']
    if change == 'reference': rows[-1]['output_text'] = 'print(1)'
    if change == 'tokens': rows[0]['input_ids_sha256'] = 'bad'
    if change == 'metadata': rows[0]['metadata_sha256'] = 'bad'
    if change == 'none': validate_grid(policy, registry, rows, metadata)
    else:
        with pytest.raises(ValueError): validate_grid(policy, registry, rows, metadata)


def test_scoring_requires_validated_sandbox_before_reading_or_executing():
    with pytest.raises(ValueError, match='sandbox'):
        evaluate({}, 'nonexistent.jsonl', 'never-created', functional=True)
