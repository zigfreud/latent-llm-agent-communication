"""Generate original/signature/explicit text answers without training or latent replay."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

from src.core.prompt_protocol import format_prompt, tokenizer_add_special_tokens
from src.evaluation.task_contract import public_prompt, validate_registry, validate_grid
from src.pipelines.oracle_experiment import task_sha256, write_jsonl
from src.pipelines.packet_trajectory import _atomic_json


def run(args):
    policy = json.loads(args.config.read_text(encoding='utf-8'))
    registry = json.loads(args.registry.read_text(encoding='utf-8'))
    validate_registry(policy, registry)
    references = json.loads(args.references.read_text(encoding='utf-8'))
    reference_map = {r['task_id']: r['code'] for r in references}
    if len(reference_map) != len(references) or set(reference_map) != set(policy['task_ids']):
        raise ValueError('reference cohort mismatch')
    for t in registry['tasks']:
        if hashlib.sha256(reference_map[t['task_id']].encode()).hexdigest() != t['reference_code_sha256']:
            raise ValueError('reference source mismatch')
    if args.validate_only:
        print(json.dumps({'validated': True, 'tasks': 32, 'conditions': policy['conditions'],
                          'candidate_code_executed': False, 'registry_sha256': task_sha256(registry)}))
        return
    import torch
    from src.pipelines.infer import load_target, model_input_device
    if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('paired calibration requires an L4 runtime')
    metadata = {'policy': policy, 'registry': registry,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'python': platform.python_version(), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
        'packages': {p: importlib.metadata.version(p) for p in ('transformers', 'bitsandbytes', 'accelerate')},
        'no_training': True, 'no_latent_packets': True}
    args.output.mkdir(parents=True, exist_ok=True)
    meta_path = args.output / 'generations.metadata.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError('resume provenance differs')
    _atomic_json(meta_path, metadata)
    _atomic_json(args.output / 'policy.json', policy)
    row_dir = args.output / 'records'
    row_dir.mkdir(exist_ok=True)
    receiver, tokenizer = load_target(policy['receiver']['model_id'], 'cuda:0', True, revision=policy['receiver']['revision'])
    receiver.requires_grad_(False)
    device = model_input_device(receiver)
    eos = receiver.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    kwargs = dict(policy['generation'], pad_token_id=tokenizer.eos_token_id, eos_token_id=list(eos_ids))
    runtime = {'generation_kwargs': kwargs, 'receiver': policy['receiver']}
    _atomic_json(args.output / 'generation_runtime.json', runtime)
    for condition in policy['conditions']:
        for task in registry['tasks']:
            tid = task['task_id']
            path = row_dir / f'{condition}_{tid}.json'
            if path.exists():
                if json.loads(path.read_text())['metadata_sha256'] != task_sha256(metadata):
                    raise ValueError('record provenance differs')
                continue
            row = {'task_id': tid, 'condition': condition, 'task_spec': task['original_task'],
                   'metadata_sha256': task_sha256(metadata), 'hit_token_limit': False}
            if condition == 'reference_source':
                row.update(output_text=reference_map[tid], public_prompt=None, output_tokens=None,
                           input_tokens=None, seconds=None, output_token_ids=None)
            else:
                prompt = public_prompt(registry, task, condition)
                formatted = format_prompt(prompt, tokenizer, policy['prompt_protocol'])
                inputs = tokenizer(formatted, add_special_tokens=tokenizer_add_special_tokens(policy['prompt_protocol']), return_tensors='pt').to(device)
                ids_hash = task_sha256(inputs['input_ids'].tolist())
                if condition == 'text_original' and ids_hash != task['original_text_input_ids_sha256']:
                    raise ValueError('original prompt tokenization changed')
                length = inputs['input_ids'].shape[1]
                torch.manual_seed(policy['seed'])
                torch.cuda.manual_seed_all(policy['seed'])
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = receiver.generate(**inputs, **kwargs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                continuation = generated[0, length:].tolist()
                row.update(public_prompt=prompt, input_ids_sha256=ids_hash, input_tokens=length,
                    output_token_ids=continuation, output_tokens=len(continuation), seconds=elapsed,
                    output_text=tokenizer.decode(continuation, skip_special_tokens=True).replace('</s>', '').strip(),
                    hit_token_limit=len(continuation) == kwargs['max_new_tokens'] and continuation[-1] not in eos_ids,
                    reproduces_old_text_tokens=continuation == task['original_text_output_token_ids'] if condition == 'text_original' else None)
                del generated
            _atomic_json(path, row)
            print(json.dumps({'event': 'generated', 'condition': condition, 'task_id': tid, 'tokens': row['output_tokens']}), flush=True)
        print(json.dumps({'event': 'condition_complete', 'condition': condition}), flush=True)
    rows = [json.loads((row_dir / f'{c}_{t}.json').read_text()) for c in policy['conditions'] for t in policy['task_ids']]
    validate_grid(policy, registry, rows, metadata)
    write_jsonl(args.output / 'generations.jsonl', rows)
    _atomic_json(args.output / 'generation_complete.json', {'records': len(rows), 'generations_sha256': hashlib.sha256((args.output / 'generations.jsonl').read_bytes()).hexdigest()})
    print('CONTRACT_CALIBRATION_GENERATED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/H0-017_task_contract_calibration_v1.json'))
    parser.add_argument('--registry', type=Path, default=Path('config/H0-017_task_contract_v2.json'))
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    run(parser.parse_args())
