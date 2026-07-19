"""Syntax and opt-in subprocess evaluation for generated Python code."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from typing import Any, Mapping


FENCED_CODE = re.compile(
    r"```(?:python|py)?\s*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


SUBPROCESS_RUNNER = r"""
import json
import sys

try:
    import resource
except ImportError:
    resource = None

payload = json.loads(sys.stdin.read())
if resource is not None:
    memory = int(payload["memory_mb"]) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (int(payload["cpu_seconds"]), int(payload["cpu_seconds"])))

namespace = {"__name__": "__candidate__"}
try:
    setup = payload.get("test_setup_code") or ""
    if setup:
        exec(compile(setup, "<test_setup>", "exec"), namespace, namespace)
    exec(compile(payload["code"], "<candidate>", "exec"), namespace, namespace)
    for test in payload.get("tests", []):
        exec(compile(test, "<test>", "exec"), namespace, namespace)
except BaseException as exc:
    print(json.dumps({"passed": False, "error_type": type(exc).__name__, "error": str(exc)[:1000]}))
else:
    print(json.dumps({"passed": True, "error_type": None, "error": None}))
"""


def extract_code(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("generated output must be text")
    match = FENCED_CODE.search(text)
    return (match.group(1) if match else text).strip()


def check_syntax(code: str) -> dict[str, Any]:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {
            "syntax_pass": False,
            "syntax_error": f"{exc.msg} (line {exc.lineno})",
        }
    return {"syntax_pass": True, "syntax_error": None}


def normalize_tests(task: Mapping[str, Any]) -> list[str]:
    tests = task.get("test_list", task.get("tests", []))
    if isinstance(tests, str):
        tests = [tests]
    if not isinstance(tests, list) or any(not isinstance(test, str) for test in tests):
        raise ValueError("task tests must be a string or list of strings")
    if not tests:
        raise ValueError("functional evaluation requires at least one test")
    return tests


def run_functional_tests(
    code: str,
    task: Mapping[str, Any],
    *,
    timeout_seconds: float = 5.0,
    memory_mb: int = 512,
) -> dict[str, Any]:
    """Run untrusted code in a resource-limited subprocess, not a security sandbox."""

    payload = {
        "code": code,
        "test_setup_code": task.get("test_setup_code", ""),
        "tests": normalize_tests(task),
        "memory_mb": int(memory_mb),
        "cpu_seconds": max(1, int(timeout_seconds)),
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", SUBPROCESS_RUNNER],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "functional_pass": False,
            "functional_error_type": "TimeoutExpired",
            "functional_error": f"exceeded {timeout_seconds} seconds",
        }

    stdout_lines = completed.stdout.strip().splitlines()
    if not stdout_lines:
        return {
            "functional_pass": False,
            "functional_error_type": "RunnerFailure",
            "functional_error": (completed.stderr or "runner produced no result")[:1000],
        }
    try:
        result = json.loads(stdout_lines[-1])
    except json.JSONDecodeError:
        return {
            "functional_pass": False,
            "functional_error_type": "RunnerProtocolError",
            "functional_error": completed.stdout[-1000:],
        }
    return {
        "functional_pass": bool(result.get("passed")),
        "functional_error_type": result.get("error_type"),
        "functional_error": result.get("error"),
    }


def evaluate_generation(
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    run_functional: bool = False,
    timeout_seconds: float = 5.0,
    memory_mb: int = 512,
) -> dict[str, Any]:
    code = extract_code(str(record.get("output_text", "")))
    result = dict(record)
    result["extracted_code"] = code
    result.update(check_syntax(code))
    if run_functional and result["syntax_pass"]:
        result.update(
            run_functional_tests(
                code,
                task,
                timeout_seconds=timeout_seconds,
                memory_mb=memory_mb,
            )
        )
    elif run_functional:
        result.update(
            {
                "functional_pass": False,
                "functional_error_type": "SyntaxError",
                "functional_error": result["syntax_error"],
            }
        )
    else:
        result["functional_pass"] = None
        result["functional_error_type"] = None
        result["functional_error"] = None
    return result
