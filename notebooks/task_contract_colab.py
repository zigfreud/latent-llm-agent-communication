"""Run in an authenticated Colab after pinning CA_REPO and CA_EXPECTED_COMMIT.

This supervisor never executes benchmark code; only the namespace evaluator does.
"""
from pathlib import Path
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time

import requests
from googleapiclient.http import MediaFileUpload

CA_REPO = Path(CA_REPO)
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CA_REPO, text=True).strip() != CA_EXPECTED_COMMIT:
    raise RuntimeError('code revision mismatch')
CA_OUT = Path('/content/lip-cache/results/LIP-H0-017/task-contract-calibration-v1')
CA_OUT.mkdir(parents=True, exist_ok=True)
CA_RECEIPTS = []
ca_context_path = CA_OUT / 'execution_context.json'
if ca_context_path.exists():
    ca_context = json.loads(ca_context_path.read_text())
    if ca_context['commit'] != CA_EXPECTED_COMMIT:
        raise RuntimeError('refusing to resume another code revision')
    CA_FOLDER = ca_context['folder_id']
else:
    CA_FOLDER = drive_api.files().create(body={
        'name': 'H0-017-task-contract-calibration-v1-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()),
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': ['1bNgw-DDCmwJjyHhvuKzZa1oVFIBC4xkI']}, fields='id').execute()['id']
    ca_context_path.write_text(json.dumps({'commit': CA_EXPECTED_COMMIT, 'folder_id': CA_FOLDER}, indent=2))


def ca_preserve(path, name=None):
    path = Path(path)
    name = name or path.name
    raw = path.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    existing = drive_api.files().list(q="'" + CA_FOLDER + "' in parents and trashed=false",
        fields='files(id,name,md5Checksum,size)', pageSize=100).execute()['files']
    matching = [f for f in existing if f['name'] == name]
    if matching:
        if len(matching) != 1 or matching[0].get('md5Checksum') != md5:
            raise RuntimeError('refusing changed overwrite: ' + name)
        saved = matching[0]
    else:
        saved = drive_api.files().create(body={'name': name, 'parents': [CA_FOLDER]},
            media_body=MediaFileUpload(str(path), resumable=True), fields='id,name,md5Checksum,size').execute()
    verified = drive_api.files().get(fileId=saved['id'], fields='id,name,md5Checksum,size').execute()
    if verified['md5Checksum'] != md5 or int(verified['size']) != len(raw):
        raise RuntimeError('persistent artifact differs: ' + name)
    receipt = {**verified, 'sha256': hashlib.sha256(raw).hexdigest(), 'folder_id': CA_FOLDER}
    CA_RECEIPTS.append(receipt)
    print('CONTRACT_BACKUP', json.dumps(receipt), flush=True)
    return receipt


def ca_snapshot(label):
    archive = Path(shutil.make_archive(str(CA_OUT.parent / ('contract-' + label + '-' + str(time.time_ns()))), 'zip', CA_OUT))
    return ca_preserve(archive)


ca_policy = CA_REPO / 'config/H0-017_task_contract_calibration_v1.json'
ca_registry = CA_REPO / 'config/H0-017_task_contract_v2.json'
for ca_file in [ca_context_path, ca_policy, ca_registry, CA_REPO / 'docs/H0-017_task_contract_calibration_v1.md']:
    ca_preserve(ca_file)

ca_url = 'https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/4bb6404fdc6cacfda99d4ac4205087b89d32030c/full/validation-00000-of-00001.parquet'
ca_response = requests.get(ca_url, timeout=90)
ca_response.raise_for_status()
if hashlib.sha256(ca_response.content).hexdigest() != '3f0ec060987432d99fe8fb409d31e6c67445b208a01741c5583517c80a10fe80':
    raise RuntimeError('benchmark source changed')
import pyarrow.parquet as pq
ca_ids = json.loads(ca_policy.read_text())['task_ids']
ca_refs = [{'task_id': str(r['task_id']), 'code': r['code']} for r in pq.read_table(io.BytesIO(ca_response.content)).to_pylist() if str(r['task_id']) in ca_ids]
ca_refs_path = CA_OUT / 'selected_references.json'
ca_refs_path.write_text(json.dumps(ca_refs, indent=2) + '\n')
ca_preserve(ca_refs_path)
ca_command = [sys.executable, '-u', '-m', 'src.scripts.run_task_contract_calibration',
    '--references', str(ca_refs_path), '--output', str(CA_OUT)]
subprocess.run(ca_command + ['--validate-only'], cwd=CA_REPO, check=True)
# Check namespace availability before loading a model. Full adversarial probes
# still run inside the hardened evaluator before any candidate execution.
subprocess.run(['unshare', '--mount', '--net', '--ipc', '--uts', 'true'], check=True)
with (CA_OUT / 'generation.log').open('a') as ca_log:
    ca_process = subprocess.Popen(ca_command, cwd=CA_REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for ca_line in ca_process.stdout:
            ca_log.write(ca_line)
            ca_log.flush()
            print(ca_line, end='', flush=True)
            if ca_line.startswith('{') and json.loads(ca_line).get('event') == 'condition_complete':
                ca_snapshot(json.loads(ca_line)['condition'])
        ca_rc = ca_process.wait()
    finally:
        if ca_process.poll() is None:
            ca_process.terminate()
            ca_process.wait(timeout=30)
    if ca_rc:
        ca_snapshot('generation-failed')
        raise RuntimeError('generation failed; evidence preserved')

ca_eval_dir = CA_OUT / 'functional-evaluation'
if not (ca_eval_dir / 'summary.json').exists():
    ca_score = subprocess.run([sys.executable, '-m', 'src.scripts.run_hardened_oracle_evaluation',
        '--config', str(ca_policy), '--generations', str(CA_OUT / 'generations.jsonl'),
        '--output-dir', str(ca_eval_dir)], cwd=CA_REPO, capture_output=True, text=True)
    (CA_OUT / 'evaluation.log').write_text(ca_score.stdout + ca_score.stderr)
    print(ca_score.stdout, ca_score.stderr)
    if ca_score.returncode:
        ca_snapshot('evaluation-failed')
        raise RuntimeError('isolated evaluation failed; evidence preserved')
ca_summary = json.loads((ca_eval_dir / 'summary.json').read_text())
if not ca_summary['sandbox']['validated']:
    raise RuntimeError('sandbox not validated')
ca_rows = [json.loads(line) for line in (ca_eval_dir / 'scored.jsonl').read_text().splitlines()]
ca_reproduced = sum(r.get('reproduces_old_text_tokens', False) for r in ca_rows if r['condition'] == 'text_original')
ca_report = ['# H0-017: calibração do contrato de correção', '',
    '96 respostas novas; 32 tarefas de desenvolvimento reutilizadas; sem treino ou replay latente.', '',
    '| Condição | Todos os testes | Incompatibilidade de chamada | Limite de tokens |', '|---|---:|---:|---:|']
for ca_condition, ca_groups in ca_summary['conditions'].items():
    ca_s = ca_groups['all_32']
    ca_report.append(f"| {ca_condition} | {ca_s['functional_pass']}/{ca_s['tasks']} | {ca_s['call_shape_mismatch']} | {ca_s['hit_token_limit']} |")
ca_report += ['', f'A condição original reproduziu {ca_reproduced}/32 respostas antigas token a token.',
    '', 'reference_source verifica os códigos de referência sob os testes originais; não é uma condição gerada pelo modelo.',
    '', 'Sete tarefas têm divergências de referência ou escolhas semânticas declaradas. Permanecem no denominador; os dois estratos estão no summary.json.',
    '', 'Nenhum candidato, assinatura ou teste foi reparado. Contrato completo altera informação disponível e, em alguns casos, explicita uma escolha semântica. Ganhos não demonstram melhora do canal latente nem generalização.',
    '', 'Pareamento: ' + json.dumps(ca_summary['paired_comparisons']),
    '', 'Código: ' + CA_EXPECTED_COMMIT,
    '', 'Drive: https://drive.google.com/drive/folders/' + CA_FOLDER]
(CA_OUT / 'RESULTADO.md').write_text('\n'.join(ca_report) + '\n')
for ca_file in [CA_OUT / 'generations.metadata.json', CA_OUT / 'generations.jsonl',
                ca_eval_dir / 'summary.json', ca_eval_dir / 'scored.jsonl', ca_eval_dir / 'sandbox_report.json', CA_OUT / 'RESULTADO.md']:
    ca_preserve(ca_file)
ca_final = ca_snapshot('complete')
(CA_OUT / 'backup_receipts.json').write_text(json.dumps(CA_RECEIPTS, indent=2) + '\n')
ca_preserve(CA_OUT / 'backup_receipts.json')
from IPython.display import display, Markdown
display(Markdown('\n'.join(ca_report)))
print('CONTRACT_CALIBRATION_COMPLETE_AND_PERSISTED', CA_FOLDER, json.dumps(ca_final), flush=True)
