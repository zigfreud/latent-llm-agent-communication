"""Run registered functional scoring inside a probed Linux namespace sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.evaluation.linux_sandbox import (
    DEFAULT_MASKED_PATHS,
    SANDBOX_VERSION,
    current_namespace,
    run_candidate_probe,
    validate_probe_report,
)
from src.evaluation.semantics import CandidateProcessPolicy
from src.pipelines.oracle_experiment import (
    load_yaml,
    prepare_output_dir,
    write_json,
)
from src.scripts.evaluate_oracle_packet_semantics import (
    evaluate as evaluate_oracle_packet_semantics,
)
from src.scripts.evaluate_packet_bridge_confirmation import (
    evaluate as evaluate_packet_bridge_confirmation,
)
from src.scripts.evaluate_functional_bridge_screen import (
    evaluate as evaluate_functional_bridge_screen,
)


NOBODY_UID = 65534
NOBODY_GID = 65534
REPO_ROOT = Path(__file__).resolve().parents[2]
INTERNAL_SOURCE = Path("/tmp/lip-work/source")
INTERNAL_OUTPUT = Path("/tmp/lip-work/output/evaluation")
INTERNAL_CONFIG = INTERNAL_SOURCE / "input" / "config.yaml"
INTERNAL_GENERATIONS = INTERNAL_SOURCE / "input" / "generations.jsonl"

NAMESPACE_SETUP = r"""
set -euo pipefail
python_bin="$1"
host_source="$2"
host_output="$3"
parent_mnt="$4"
parent_net="$5"
host_secret="$6"
allow_incomplete="$7"

mount --make-rprivate /
mount -t tmpfs -o mode=1777,nosuid,nodev,size=256m tmpfs /tmp
mkdir -p /tmp/lip-work/source /tmp/lip-work/output /tmp/lip-empty
mount --bind "$host_source" /tmp/lip-work/source
mount -o remount,bind,ro,nosuid,nodev /tmp/lip-work/source
mount --bind "$host_output" /tmp/lip-work/output
mount -o remount,bind,rw,nosuid,nodev /tmp/lip-work/output

for target in /root /home /content /mnt /var/tmp /workspace; do
  if [[ -e "$target" ]]; then
    mount --bind /tmp/lip-empty "$target"
    mount -o remount,bind,ro,nosuid,nodev,noexec "$target"
  fi
done

cd /tmp/lip-work/source
exec env -i LC_ALL=C.UTF-8 PATH=/usr/local/bin:/usr/bin:/bin \
  "$python_bin" -I -c \
  'import runpy, sys; sys.path.insert(0, "/tmp/lip-work/source"); runpy.run_module("src.scripts.run_hardened_oracle_evaluation", run_name="__main__")' \
  --internal-worker \
  --parent-mount-namespace "$parent_mnt" \
  --parent-network-namespace "$parent_net" \
  --host-secret "$host_secret" \
  --internal-allow-incomplete "$allow_incomplete"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/LIP-PROTO-007_oracle_packet_functional_capacity.yaml"),
    )
    parser.add_argument("--generations", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parent-mount-namespace", help=argparse.SUPPRESS)
    parser.add_argument("--parent-network-namespace", help=argparse.SUPPRESS)
    parser.add_argument("--host-secret", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--internal-allow-incomplete", help=argparse.SUPPRESS)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluator_for_config(config: dict[str, Any]):
    if config.get("experiment_id") == "LIP-PROTO-014":
        return evaluate_packet_bridge_confirmation
    if config.get("experiment_id") == "LIP-EVAL-033":
        return evaluate_functional_bridge_screen
    return evaluate_oracle_packet_semantics


def _make_tree_readonly(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise ValueError(f"sandbox source cannot contain symlinks: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _namespace_command(
    *,
    unshare: str,
    python: str,
    host_source: Path,
    host_output: Path,
    parent_mount_namespace: str,
    parent_network_namespace: str,
    host_secret: Path,
    allow_incomplete: bool,
) -> list[str]:
    return [
        unshare,
        "--mount",
        "--net",
        "--ipc",
        "--uts",
        "/bin/bash",
        "-c",
        NAMESPACE_SETUP,
        "lip-functional-sandbox",
        python,
        str(host_source),
        str(host_output),
        parent_mount_namespace,
        parent_network_namespace,
        str(host_secret),
        "1" if allow_incomplete else "0",
    ]


def _copy_inputs(config_path: Path, generations_path: Path, source: Path) -> dict[str, str]:
    metadata_path = generations_path.with_suffix(".metadata.json")
    for path in (config_path, generations_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    shutil.copytree(REPO_ROOT / "src", source / "src")
    input_dir = source / "input"
    input_dir.mkdir()
    copied = {
        "config": input_dir / "config.yaml",
        "generations": input_dir / "generations.jsonl",
        "metadata": input_dir / "generations.metadata.json",
    }
    shutil.copy2(config_path, copied["config"])
    shutil.copy2(generations_path, copied["generations"])
    shutil.copy2(metadata_path, copied["metadata"])
    hashes = {name: sha256_file(path) for name, path in copied.items()}
    _make_tree_readonly(source)
    return hashes


def _run_parent(args: argparse.Namespace) -> None:
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("the hardened evaluator requires Linux and effective UID 0")
    unshare = shutil.which("unshare")
    if not unshare:
        raise RuntimeError("unshare is required for the hardened evaluator")

    config_path = args.config.resolve()
    config = load_yaml(config_path)
    output_config = config.get("output", {})
    if args.generations is None and not output_config.get("generations_jsonl"):
        raise ValueError("--generations is required when config has no output path")
    if args.output_dir is None and not output_config.get("evaluation_dir"):
        raise ValueError("--output-dir is required when config has no output path")
    generations_path = (
        args.generations.resolve()
        if args.generations
        else Path(str(output_config["generations_jsonl"])).resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else Path(str(output_config["evaluation_dir"]))
        .with_name("functional-evaluation")
        .resolve()
    )

    with tempfile.TemporaryDirectory(prefix="lip-functional-", dir="/var/tmp") as raw_stage:
        stage = Path(raw_stage)
        stage.chmod(0o700)
        source = stage / "source"
        output = stage / "output"
        output.mkdir(mode=0o700)
        input_hashes = _copy_inputs(config_path, generations_path, source)
        host_secret = stage / "host-secret"
        host_secret.write_text("must-not-be-readable-by-candidates\n", encoding="utf-8")
        host_secret.chmod(stat.S_IRUSR | stat.S_IWUSR)

        command = _namespace_command(
            unshare=unshare,
            python=sys.executable,
            host_source=source,
            host_output=output,
            parent_mount_namespace=current_namespace("mnt"),
            parent_network_namespace=current_namespace("net"),
            host_secret=host_secret,
            allow_incomplete=args.allow_incomplete,
        )
        with generations_path.open("r", encoding="utf-8") as handle:
            generation_count = sum(1 for line in handle if line.strip())
        wall_timeout = max(60 * 60, generation_count * 6 + 10 * 60)
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=wall_timeout,
            check=False,
            env={"LC_ALL": "C.UTF-8", "PATH": "/usr/local/bin:/usr/bin:/bin"},
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(f"hardened evaluation failed:\n{detail}")

        sandbox_output = output / "evaluation"
        summary_path = sandbox_output / "summary.json"
        report_path = sandbox_output / "sandbox_report.json"
        if not summary_path.is_file() or not report_path.is_file():
            raise RuntimeError("hardened evaluator did not produce summary and probe report")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("subprocess_is_security_sandbox"):
            raise RuntimeError("result was not marked as a validated security sandbox")
        if input_hashes != summary.get("sandbox", {}).get("input_sha256"):
            raise RuntimeError("sandbox result does not bind the staged input hashes")

        prepare_output_dir(output_dir, overwrite=args.overwrite)
        shutil.copytree(sandbox_output, output_dir, dirs_exist_ok=True)

    print("Hardened functional evaluation completed")
    print(f"execution_mode: {summary['execution_mode']}")
    print(f"claim_eligible: {summary['claim_eligible']}")
    signal_key = (
        "semantic_transport_supported"
        if "semantic_transport_supported" in summary
        else "development_functional_signal_detected"
    )
    print(f"{signal_key}: {summary[signal_key]}")
    print(f"summary: {output_dir / 'summary.json'}")


def _run_worker(args: argparse.Namespace) -> None:
    if os.geteuid() != 0:
        raise RuntimeError("internal sandbox worker must retain namespace root")
    if not all(
        (args.parent_mount_namespace, args.parent_network_namespace, args.host_secret)
    ):
        raise ValueError("internal sandbox provenance arguments are required")
    policy = CandidateProcessPolicy(uid=NOBODY_UID, gid=NOBODY_GID)
    existing_masks = [path for path in DEFAULT_MASKED_PATHS if Path(path).exists()]
    report = run_candidate_probe(
        policy,
        masked_paths=existing_masks,
        source_dir=INTERNAL_SOURCE,
        output_dir=INTERNAL_OUTPUT,
        host_secret=args.host_secret,
    )
    checks = validate_probe_report(
        report,
        policy=policy,
        parent_mount_namespace=args.parent_mount_namespace,
        parent_network_namespace=args.parent_network_namespace,
        expected_masked_paths=existing_masks,
    )
    input_hashes = {
        "config": sha256_file(INTERNAL_CONFIG),
        "generations": sha256_file(INTERNAL_GENERATIONS),
        "metadata": sha256_file(INTERNAL_GENERATIONS.with_suffix(".metadata.json")),
    }
    security_context: dict[str, Any] = {
        "version": SANDBOX_VERSION,
        "validated": True,
        "candidate_uid": policy.uid,
        "candidate_gid": policy.gid,
        "checks": checks,
        "input_sha256": input_hashes,
        "probe_report": "sandbox_report.json",
    }
    config = load_yaml(INTERNAL_CONFIG)
    os.umask(0o077)
    evaluator = evaluator_for_config(config)
    summary = evaluator(
        config,
        INTERNAL_GENERATIONS,
        INTERNAL_OUTPUT,
        functional=True,
        allow_incomplete=args.internal_allow_incomplete == "1",
        overwrite=True,
        candidate_process_policy=policy,
        security_context=security_context,
    )
    write_json(
        INTERNAL_OUTPUT / "sandbox_report.json",
        {
            "version": SANDBOX_VERSION,
            "validated": True,
            "checks": checks,
            "observed": report,
        },
    )
    print(json.dumps({"summary": summary["execution_mode"], "sandbox": "passed"}))


def main() -> None:
    args = parse_args()
    if args.internal_worker:
        _run_worker(args)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
