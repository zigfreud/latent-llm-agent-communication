"""Versioned public task information and strict, sandbox-only calibration scoring."""
import ast
import inspect
import json
from pathlib import Path

from src.pipelines.oracle_experiment import task_sha256, prepare_output_dir, write_json, write_jsonl
from src.evaluation.semantics import evaluate_generation


def public_prompt(registry, task, condition):
    if condition == 'text_original':
        return task['original_task']['prompt']
    if condition == 'text_signature':
        return task['original_task']['prompt'] + '\n\nRequired signature: ' + task['signature'] + '.'
    if condition == 'text_explicit':
        return '\n\n'.join((registry['shared_public_rules'],
            'Required signature: ' + task['signature'] + '.',
            'Inputs: ' + task['inputs'], 'Return contract: ' + task['returns']))
    raise ValueError('unsupported public prompt condition')


def interface_diagnostic(code, task):
    """Inspect a declaration and literal test calls without executing candidate code.

    This is diagnostic only: aliases, decorators, and runtime rebinding can make
    static declarations inconclusive. Never reject a functional pass on this basis.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {'status': 'syntax_error'}
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == task['entry_point']]
    if len(functions) != 1 or functions[0].decorator_list:
        return {'status': 'unknown_declaration'}
    fn = functions[0]
    parameters = []
    positional = fn.args.posonlyargs + fn.args.args
    defaults_start = len(positional) - len(fn.args.defaults)
    for i, arg in enumerate(positional):
        kind = inspect.Parameter.POSITIONAL_ONLY if i < len(fn.args.posonlyargs) else inspect.Parameter.POSITIONAL_OR_KEYWORD
        parameters.append(inspect.Parameter(arg.arg, kind, default=None if i >= defaults_start else inspect.Parameter.empty))
    if fn.args.vararg:
        parameters.append(inspect.Parameter(fn.args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        parameters.append(inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY,
            default=None if default is not None else inspect.Parameter.empty))
    if fn.args.kwarg:
        parameters.append(inspect.Parameter(fn.args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    signature = inspect.Signature(parameters)
    calls = [n for test in task['test_list'] for n in ast.walk(ast.parse(test))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == task['entry_point']]
    if not calls or any(any(isinstance(a, ast.Starred) for a in c.args) or any(k.arg is None for k in c.keywords) for c in calls):
        return {'status': 'unknown_calls'}
    for call in calls:
        try:
            signature.bind(*([None] * len(call.args)), **{k.arg: None for k in call.keywords})
        except TypeError as exc:
            return {'status': 'call_shape_mismatch', 'detail': str(exc)}
    return {'status': 'calls_bind', 'async_declaration': isinstance(fn, ast.AsyncFunctionDef)}


def validate_registry(policy, registry):
    if policy['experiment_id'] != 'H0-017-task-contract-calibration-v1':
        raise ValueError('incorrect experiment')
    if task_sha256(registry) != policy['registry_sha256'] or registry['contract_id'] != policy['contract_id']:
        raise ValueError('contract registry changed')
    if policy['training_allowed'] or policy['confirmation_allowed']:
        raise ValueError('calibration cannot authorize training or confirmation')
    ids = [t['task_id'] for t in registry['tasks']]
    if ids != policy['task_ids'] or len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError('cohort changed')
    if policy['conditions'] != ['text_original', 'text_signature', 'text_explicit', 'reference_source']:
        raise ValueError('calibration conditions changed')
    flagged = [t['task_id'] for t in registry['tasks'] if t['review_flags']]
    if flagged != policy['review_flagged_task_ids']:
        raise ValueError('review stratum changed')
    for task in registry['tasks']:
        original = task['original_task']
        if original['task_id'] != task['task_id'] or task_sha256(original) != task['original_task_sha256']:
            raise ValueError('original task changed')
        tests = {k: original[k] for k in ('test_list', 'test_setup_code')}
        if task_sha256(tests) != task['tests_sha256']:
            raise ValueError('original tests changed')
        code = 'def ' + task['signature'] + ':\n    pass\n'
        if interface_diagnostic(code, original)['status'] != 'calls_bind':
            raise ValueError('declared signature cannot accept original tests')


def validate_grid(policy, registry, rows, metadata):
    validate_registry(policy, registry)
    if metadata['policy'] != policy or metadata['registry'] != registry:
        raise ValueError('generation provenance changed')
    expected = {(t, c) for t in policy['task_ids'] for c in policy['conditions']}
    keys = [(r['task_id'], r['condition']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('calibration grid incomplete or duplicated')
    tasks = {t['task_id']: t for t in registry['tasks']}
    for row in rows:
        if row['metadata_sha256'] != task_sha256(metadata):
            raise ValueError('row provenance changed')
        task = tasks[row['task_id']]
        if row['task_spec'] != task['original_task']:
            raise ValueError('scoring task or tests changed')
        if row['condition'] == 'reference_source':
            import hashlib
            if hashlib.sha256(row['output_text'].encode()).hexdigest() != task['reference_code_sha256']:
                raise ValueError('reference code changed')
        elif row['public_prompt'] != public_prompt(registry, task, row['condition']):
            raise ValueError('public input changed')
        elif row['condition'] == 'text_original' and row['input_ids_sha256'] != task['original_text_input_ids_sha256']:
            raise ValueError('original tokenized input changed')


def evaluate(config, generations_path, output_dir, *, functional=False,
             allow_incomplete=False, overwrite=False,
             candidate_process_policy=None, security_context=None):
    if not functional or not security_context or not security_context.get('validated') or candidate_process_policy is None:
        raise ValueError('calibration requires the validated Linux sandbox')
    rows = [json.loads(s) for s in Path(generations_path).read_text(encoding='utf-8').splitlines() if s.strip()]
    metadata = json.loads(Path(generations_path).with_suffix('.metadata.json').read_text(encoding='utf-8'))
    registry = metadata['registry']
    validate_grid(config, registry, rows, metadata)
    prepare_output_dir(Path(output_dir), overwrite=overwrite)
    scored = []
    for row in rows:
        result = evaluate_generation(row, row['task_spec'], run_functional=True, process_policy=candidate_process_policy)
        result['interface_diagnostic'] = interface_diagnostic(result['extracted_code'], row['task_spec'])
        scored.append(result)
    flagged = set(config['review_flagged_task_ids'])
    from collections import Counter
    def summarize(group):
        return {'tasks': len(group), 'functional_pass': sum(bool(r['functional_pass']) for r in group),
                'syntax_pass': sum(bool(r['syntax_pass']) for r in group),
                'call_shape_mismatch': sum(r['interface_diagnostic']['status'] == 'call_shape_mismatch' for r in group),
                'hit_token_limit': sum(bool(r['hit_token_limit']) for r in group),
                'failures': dict(Counter(r['functional_error_type'] for r in group if not r['functional_pass']))}
    totals = {c: {label: summarize([r for r in scored if r['condition'] == c and r['task_id'] in ids])
                  for label, ids in [('all_32', set(config['task_ids'])), ('review_flagged', flagged),
                                     ('remaining_25', set(config['task_ids']) - flagged)]}
              for c in config['conditions']}
    passes = {(r['task_id'], r['condition']): r['functional_pass'] for r in scored}
    paired = {}
    for baseline, treatment in [('text_original', 'text_signature'), ('text_signature', 'text_explicit'), ('text_original', 'text_explicit')]:
        paired[baseline + '_to_' + treatment] = {
            'gained_task_ids': [t for t in config['task_ids'] if passes[t, treatment] and not passes[t, baseline]],
            'lost_task_ids': [t for t in config['task_ids'] if passes[t, baseline] and not passes[t, treatment]]}
    summary = {'experiment_id': config['experiment_id'], 'claim_eligible': False,
        'scope': config['scope'], 'sandbox': security_context, 'conditions': totals, 'paired_comparisons': paired,
        'reference_passes_original_tests': totals['reference_source']['all_32']['functional_pass'] == 32,
        'no_candidate_repairs': True, 'tests_unchanged': True}
    write_jsonl(Path(output_dir) / 'scored.jsonl', scored)
    write_json(Path(output_dir) / 'summary.json', summary)
    return summary
