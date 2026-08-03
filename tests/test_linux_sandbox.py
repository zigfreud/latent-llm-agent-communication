import copy

import pytest

from src.evaluation.linux_sandbox import validate_probe_report
from src.evaluation.semantics import CandidateProcessPolicy
from src.scripts.run_hardened_oracle_evaluation import _namespace_command


def valid_report(policy):
    return {
        "uid": policy.uid,
        "gid": policy.gid,
        "effective_uid": policy.uid,
        "effective_gid": policy.gid,
        "groups": [],
        "cap_eff": "0000000000000000",
        "no_new_privs": "1",
        "environment": dict(policy.environment),
        "mount_namespace": "mnt:[2]",
        "network_namespace": "net:[2]",
        "interfaces": ["lo"],
        "default_route_present": False,
        "masked_paths": {
            "/root": {"entries": [], "readonly": True},
            "/content": {"entries": [], "readonly": True},
        },
        "source_readonly": True,
        "output_inaccessible": True,
        "host_secret_inaccessible": True,
    }


def test_probe_validation_requires_every_registered_security_property():
    policy = CandidateProcessPolicy(uid=65534, gid=65534)
    report = valid_report(policy)
    checks = validate_probe_report(
        report,
        policy=policy,
        parent_mount_namespace="mnt:[1]",
        parent_network_namespace="net:[1]",
        expected_masked_paths=["/root", "/content"],
    )
    assert checks and all(checks.values())

    for field in ("no_new_privs", "source_readonly", "output_inaccessible"):
        broken = copy.deepcopy(report)
        broken[field] = "0" if field == "no_new_privs" else False
        with pytest.raises(RuntimeError, match="sandbox probe failed"):
            validate_probe_report(
                broken,
                policy=policy,
                parent_mount_namespace="mnt:[1]",
                parent_network_namespace="net:[1]",
                expected_masked_paths=["/root", "/content"],
            )


def test_namespace_command_freezes_all_required_linux_namespaces():
    command = _namespace_command(
        unshare="/usr/bin/unshare",
        python="/usr/bin/python3",
        host_source="/var/tmp/stage/source",
        host_output="/var/tmp/stage/output",
        parent_mount_namespace="mnt:[1]",
        parent_network_namespace="net:[1]",
        host_secret="/var/tmp/stage/secret",
        allow_incomplete=False,
    )
    assert command[:8] == [
        "/usr/bin/unshare",
        "--mount",
        "--net",
        "--pid",
        "--fork",
        "--mount-proc",
        "--ipc",
        "--uts",
    ]
    assert command[-1] == "0"
