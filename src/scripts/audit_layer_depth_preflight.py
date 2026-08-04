"""Apply the registered pre-confirmation authorization amendment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.oracle_layer_depth import summarize_preflight_authorization
from src.pipelines.oracle_experiment import load_json_object, load_yaml, write_json


DEFAULT_CONFIG = Path("config/LIP-PROTO-009_oracle_layer_depth.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--scored-generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} must be a JSON object")
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("experiment_id") != "LIP-PROTO-009":
        raise ValueError("preflight amendment is registered only for LIP-PROTO-009")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    metadata = load_json_object(args.metadata)
    records = read_jsonl(args.scored_generations)
    summary = summarize_preflight_authorization(
        records,
        metadata,
        maximum_self_logit_delta=float(config["memory"]["maximum_self_logit_delta"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, summary)
    print(f"pre-confirmation authorization: {summary['passed']}")
    print(f"summary: {args.output}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
