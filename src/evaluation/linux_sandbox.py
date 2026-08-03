"""Linux namespace and privilege probes for untrusted functional evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.semantics import CandidateProcessPolicy, build_candidate_preexec


SANDBOX_VERSION = "lip-linux-functional-sandbox-v1"
DEFAULT_MASKED_PATHS = (
    "/root",
    "/home",
    "/content",
    "/mnt",
    "/var/tmp",
    "/workspace",
)

PROBE_RUNNER = r"""
import json
import os
import socket
import sys

request = json.loads(sys.stdin.read())

def status_value(name):
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(name + ":"):
                return line.split(":", 1)[1].strip()
    return None

def inaccessible(path):
    if os.path.isdir(path):
        try:
            os.listdir(path)
        except (FileNotFoundError, PermissionError):
            return True
        return False
    try:
        with open(path, "rb") as handle:
            handle.read(1)
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return True
    return False

def readonly(path):
    probe = os.path.join(path, ".lip-sandbox-write-probe")
    try:
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("unexpected")
    except (FileNotFoundError, PermissionError, OSError):
        return True
    else:
        try:
            os.unlink(probe)
        except OSError:
            pass
        return False

masked = {}
for path in request["masked_paths"]:
    if not os.path.exists(path):
        continue
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        entries = None
    masked[path] = {"entries": entries, "readonly": readonly(path)}

with open("/proc/net/route", encoding="utf-8") as handle:
    routes = [line.split() for line in handle.read().splitlines()[1:] if line]

report = {
    "uid": os.getuid(),
    "gid": os.getgid(),
    "effective_uid": os.geteuid(),
    "effective_gid": os.getegid(),
    "groups": os.getgroups(),
    "cap_eff": status_value("CapEff"),
    "no_new_privs": status_value("NoNewPrivs"),
    "environment": dict(sorted(os.environ.items())),
    "mount_namespace": os.readlink("/proc/self/ns/mnt"),
    "network_namespace": os.readlink("/proc/self/ns/net"),
    "interfaces": [name for _, name in socket.if_nameindex()],
    "default_route_present": any(len(row) > 1 and row[1] == "00000000" for row in routes),
    "masked_paths": masked,
    "source_readonly": readonly(request["source_dir"]),
    "output_inaccessible": inaccessible(request["output_dir"]),
    "host_secret_inaccessible": inaccessible(request["host_secret"]),
}
print(json.dumps(report, sort_keys=True))
"""


def current_namespace(kind: str) -> str:
    if kind not in {"mnt", "net"}:
        raise ValueError("namespace kind must be mnt or net")
    return os.readlink(f"/proc/self/ns/{kind}")


def run_candidate_probe(
    policy: CandidateProcessPolicy,
    *,
    masked_paths: Sequence[str],
    source_dir: Path,
    output_dir: Path,
    host_secret: Path,
) -> dict[str, Any]:
    request = {
        "masked_paths": list(masked_paths),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "host_secret": str(host_secret),
    }
    completed = subprocess.run(
        [sys.executable, "-I", "-c", PROBE_RUNNER],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
        env=dict(policy.environment),
        preexec_fn=build_candidate_preexec(policy),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "sandbox probe failed before candidate execution: "
            + (completed.stderr or completed.stdout)[-2000:]
        )
    try:
        report = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("sandbox probe returned invalid JSON") from exc
    if not isinstance(report, dict):
        raise RuntimeError("sandbox probe report must be an object")
    return report


def validate_probe_report(
    report: Mapping[str, Any],
    *,
    policy: CandidateProcessPolicy,
    parent_mount_namespace: str,
    parent_network_namespace: str,
    expected_masked_paths: Sequence[str],
) -> dict[str, bool]:
    environment = dict(report.get("environment", {}))
    allowed_environment = dict(policy.environment)
    masked = report.get("masked_paths", {})
    expected_existing = [path for path in expected_masked_paths if path in masked]
    checks = {
        "candidate_identity_dropped": (
            report.get("uid") == policy.uid
            and report.get("gid") == policy.gid
            and report.get("effective_uid") == policy.uid
            and report.get("effective_gid") == policy.gid
            and report.get("groups") == []
        ),
        "effective_capabilities_empty": report.get("cap_eff") == "0000000000000000",
        "no_new_privileges_enabled": (
            not policy.no_new_privs or report.get("no_new_privs") == "1"
        ),
        "environment_allowlisted": environment == allowed_environment,
        "mount_namespace_private": (
            report.get("mount_namespace") != parent_mount_namespace
        ),
        "network_namespace_private": (
            report.get("network_namespace") != parent_network_namespace
        ),
        "network_has_loopback_only": set(report.get("interfaces", [])) <= {"lo"},
        "network_has_no_default_route": not report.get("default_route_present", True),
        "sensitive_mounts_empty_and_readonly": bool(expected_existing)
        and all(
            masked[path].get("entries") == [] and masked[path].get("readonly")
            for path in expected_existing
        ),
        "source_mount_readonly": bool(report.get("source_readonly")),
        "result_directory_inaccessible": bool(report.get("output_inaccessible")),
        "host_stage_secret_inaccessible": bool(
            report.get("host_secret_inaccessible")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("sandbox probe failed checks: " + ", ".join(failed))
    return checks
