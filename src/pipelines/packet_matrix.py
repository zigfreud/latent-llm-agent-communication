"""Registered multi-objective, multi-replica matrix for LIP-PROTO-014."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from src.core.packet_bundle import sha256_file
from src.evaluation.packet_bridge import summarize_multi_replica_development_gate
from src.pipelines.oracle_experiment import load_yaml
from src.pipelines.packet_bridge import train_packet_bridge


def _variant_names(config: Mapping) -> list[str]:
    variants = config.get("objectives", {}).get("variants")
    if not isinstance(variants, Mapping) or not variants:
        raise ValueError("objectives.variants must define at least one system")
    return [str(name) for name in variants]


def build_replica_config(
    contract: Mapping,
    *,
    bundle_dir: Path | str,
    output_dir: Path | str,
    variant_name: str,
    seed: int,
    device: str = "auto",
    require_real: bool = True,
    max_updates: int | None = None,
) -> dict:
    """Resolve one frozen matrix cell into the bridge trainer schema."""

    variants = contract["objectives"]["variants"]
    if variant_name not in variants:
        raise ValueError(f"unknown objective variant: {variant_name}")
    variant = variants[variant_name]
    model_kind = str(variant["model_kind"])
    model = {"kind": model_kind}
    if model_kind == "query_conditioned":
        model.update(dict(contract["bridge"]))
    elif model_kind != "structured_linear":
        raise ValueError(f"unsupported model kind: {model_kind}")

    training = {
        key: value
        for key, value in contract["training"].items()
        if key not in {"seeds", "default_output_dir"}
    }
    if max_updates is not None:
        if max_updates <= 0:
            raise ValueError("max_updates override must be positive")
        training["max_updates"] = int(max_updates)
        training["validation_interval"] = min(
            int(training["validation_interval"]), int(max_updates)
        )
    return {
        "experiment_id": str(contract["experiment_id"]),
        "protocol_version": str(contract["protocol_version"]),
        "objective_variant": variant_name,
        "objective_role": str(variant["role"]),
        "device": device,
        "output_dir": str(output_dir),
        "seed": int(seed),
        "data": {
            "bundle_dir": str(bundle_dir),
            "require_real": bool(require_real),
            "boundary_positions": int(
                contract["packets"]["target"]["boundary_positions"]
            ),
        },
        "model": model,
        "training": training,
        "loss": dict(variant["loss"]),
        "development_gate": {
            "alpha": float(contract["development_gate"]["alpha"]),
            "statistics_seed": int(
                contract["development_gate"]["statistics_seed"]
            ),
        },
    }


def _read_completed_run(run_dir: Path, resolved: Mapping) -> dict | None:
    summary_path = run_dir / "run_summary.json"
    config_path = run_dir / "resolved_config.yaml"
    if not summary_path.is_file() or not config_path.is_file():
        return None
    existing = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if existing != dict(resolved):
        raise ValueError(f"completed run config differs from requested cell: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"completed run summary is invalid: {summary_path}")
    return summary


def _aggregate_variant(
    reports: Sequence[Mapping],
    *,
    expected_replica_count: int,
    minimum_passing_replicas: int,
) -> dict:
    gate_reports = [report["development_gate"] for report in reports]
    complete = len(gate_reports) == expected_replica_count
    if not complete:
        return {
            "complete": False,
            "replica_count": len(gate_reports),
            "expected_replica_count": expected_replica_count,
            "minimum_passing_replicas": minimum_passing_replicas,
            "passing_replicas": sum(
                bool(report.get("passed")) for report in gate_reports
            ),
            "passed": False,
            "replicas": gate_reports,
        }
    aggregate = summarize_multi_replica_development_gate(
        gate_reports,
        minimum_passing_replicas=minimum_passing_replicas,
    )
    return {"complete": True, **aggregate}


def run_packet_bridge_matrix(
    config_path: Path | str,
    *,
    bundle_dir: Path | str,
    output_dir: Path | str | None = None,
    variants: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    device: str = "auto",
    allow_nonclaim_bundle: bool = False,
    resume: bool = False,
    overwrite: bool = False,
    max_updates: int | None = None,
) -> dict:
    contract_path = Path(config_path)
    contract = load_yaml(contract_path)
    configured_variants = _variant_names(contract)
    selected_variants = list(variants or configured_variants)
    if not selected_variants or any(
        name not in configured_variants for name in selected_variants
    ):
        raise ValueError("matrix variants must be a non-empty registered subset")
    configured_seeds = [int(seed) for seed in contract["training"]["seeds"]]
    selected_seeds = [int(seed) for seed in (seeds or configured_seeds)]
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("matrix seeds must be a non-empty unique sequence")
    if any(seed not in configured_seeds for seed in selected_seeds):
        raise ValueError("matrix seed is outside the registered seed set")

    output_root = Path(output_dir or contract["training"]["default_output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    reports: dict[str, list[dict]] = {name: [] for name in selected_variants}
    for variant_name in selected_variants:
        for seed in selected_seeds:
            run_dir = output_root / variant_name / f"seed-{seed}"
            resolved = build_replica_config(
                contract,
                bundle_dir=bundle_dir,
                output_dir=run_dir,
                variant_name=variant_name,
                seed=seed,
                device=device,
                require_real=not allow_nonclaim_bundle,
                max_updates=max_updates,
            )
            completed = _read_completed_run(run_dir, resolved) if resume else None
            if completed is not None:
                reports[variant_name].append(completed)
                continue
            if run_dir.exists():
                if not overwrite:
                    raise FileExistsError(f"matrix cell already exists: {run_dir}")
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)
            resolved_path = run_dir / "input_config.yaml"
            resolved_path.write_text(
                yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
            )
            reports[variant_name].append(train_packet_bridge(resolved_path))

    minimum = int(contract["development_gate"]["minimum_passing_replicas"])
    aggregate = {
        name: _aggregate_variant(
            variant_reports,
            expected_replica_count=len(configured_seeds),
            minimum_passing_replicas=minimum,
        )
        for name, variant_reports in reports.items()
    }
    primary = str(contract["objectives"]["primary"])
    primary_gate = aggregate.get(primary)
    full_registered_matrix = (
        len(selected_variants) == len(configured_variants)
        and set(selected_variants) == set(configured_variants)
        and len(selected_seeds) == len(configured_seeds)
        and set(selected_seeds) == set(configured_seeds)
        and max_updates is None
        and not allow_nonclaim_bundle
    )
    summary = {
        "experiment_id": str(contract["experiment_id"]),
        "protocol_version": str(contract["protocol_version"]),
        "contract_config": str(contract_path),
        "contract_config_sha256": sha256_file(contract_path),
        "bundle_dir": str(bundle_dir),
        "bundle_manifest_sha256": sha256_file(Path(bundle_dir) / "manifest.json"),
        "selected_variants": selected_variants,
        "selected_seeds": selected_seeds,
        "registered_variants": configured_variants,
        "registered_seeds": configured_seeds,
        "full_registered_matrix": full_registered_matrix,
        "development_gates": aggregate,
        "primary_variant": primary,
        "ready_for_confirmation": bool(
            full_registered_matrix and primary_gate and primary_gate["passed"]
        ),
        "runs": {
            name: [
                {
                    "seed": report["seed"],
                    "summary": str(
                        output_root / name / f"seed-{report['seed']}" / "run_summary.json"
                    ),
                    "development_gate_passed": bool(
                        report["development_gate"]["passed"]
                    ),
                }
                for report in variant_reports
            ]
            for name, variant_reports in reports.items()
        },
    }
    (output_root / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
