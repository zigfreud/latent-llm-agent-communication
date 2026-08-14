"""Run or aggregate frozen H0-016 hard-negative replication cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipelines.hard_negative_replication import (
    aggregate_hard_negative_replication,
    run_hard_negative_replication_training,
    validate_hard_negative_replication_contract,
)
from src.pipelines.oracle_experiment import load_json_object, load_yaml
from src.pipelines.packet_trajectory import _atomic_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("config/LIP-H0-016_hard_negative_replication.yaml"),
    )
    parser.add_argument(
        "--parent-config",
        type=Path,
        default=Path("config/LIP-PROTO-014_source_conditioned_residual_packet.yaml"),
    )
    parser.add_argument(
        "--predecessor-registry",
        type=Path,
        default=Path("experiments/registry/LIP-H0-015_hard_negative_batches.json"),
    )
    parser.add_argument("--candidate-bank", type=Path)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-device", default="auto")
    parser.add_argument("--dry-run-contract", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--seed-4001-summary", type=Path)
    parser.add_argument("--seed-4003-summary", type=Path)
    parser.add_argument("--aggregate-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = load_yaml(args.experiment_config)
    parent = load_yaml(args.parent_config)
    validate_hard_negative_replication_contract(
        experiment,
        parent,
        experiment_path=args.experiment_config,
        parent_path=args.parent_config,
        predecessor_registry_path=args.predecessor_registry,
    )
    if args.dry_run_contract:
        print("LIP-H0-016 contract validated")
        return
    if args.aggregate:
        if (
            args.seed_4001_summary is None
            or args.seed_4003_summary is None
            or args.aggregate_output is None
        ):
            raise ValueError("aggregation requires both summaries and an output")
        payload = aggregate_hard_negative_replication(
            experiment,
            load_json_object(args.predecessor_registry),
            {
                4001: load_json_object(args.seed_4001_summary),
                4003: load_json_object(args.seed_4003_summary),
            },
        )
        _atomic_json(args.aggregate_output, payload)
        print(json.dumps(payload["aggregate_gate"]))
        return
    if (
        args.candidate_bank is None
        or args.bundle_dir is None
        or args.output_dir is None
        or args.seed is None
    ):
        raise ValueError("cell execution requires candidate bank, bundle, output, and seed")
    if int(args.seed) not in (4001, 4003):
        raise ValueError("H0-016 seed must be 4001 or 4003")
    result = run_hard_negative_replication_training(
        experiment_path=args.experiment_config,
        parent_path=args.parent_config,
        predecessor_registry_path=args.predecessor_registry,
        candidate_bank_path=args.candidate_bank,
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        variant_name="hard_negative_batches_unrolled",
        seed=int(args.seed),
        pilot=False,
        target_device=str(args.target_device),
        colab_compute_units_before=None,
    )
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "seed": result["seed"],
                "summary": str(args.output_dir / "run_summary.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
