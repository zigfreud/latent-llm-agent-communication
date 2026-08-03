"""Train independent adapter replicas from one base configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.pipelines.trainer import train


def parse_args():
    parser = argparse.ArgumentParser(description="Run independent LIP adapter seeds.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def write_summary(output_root: Path, payload: dict):
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "multiseed_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    lines = [
        f"# {payload['base_experiment_id']} multi-seed training",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in payload['seeds'])}",
        f"- Successful runs: {len(payload['runs'])}",
        "",
    ]
    for run in payload["runs"]:
        lines.append(
            f"- Seed {run['seed']}: best loss {run['best_loss']:.6f}; "
            f"checkpoint `{run['checkpoint']}`"
        )
    with open(output_root / "multiseed_summary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_multiseed(
    config_path: Path,
    seeds: list[int],
    output_root: Path,
    *,
    device: str | None = None,
    max_steps: int | None = None,
) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of unique integers")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must contain a YAML mapping")
    base_id = str(config.get("experiment_id") or config.get("experiment_name", "experiment"))
    run_dirs = {seed: output_root / f"seed-{seed}" for seed in seeds}
    occupied = [path for path in run_dirs.values() if path.is_dir() and any(path.iterdir())]
    if occupied:
        raise FileExistsError(
            "refusing to reuse non-empty training directories: "
            + ", ".join(str(path) for path in occupied)
        )

    runs = []
    for seed in seeds:
        run_id = f"{base_id}-seed-{seed}"
        run_dir = run_dirs[seed]
        metrics = train(
            str(config_path),
            experiment_id=run_id,
            output_dir=str(run_dir),
            max_steps=max_steps,
            device=device,
            seed=seed,
        )
        checkpoint_path = run_dir / "best_model.pth"
        if not checkpoint_path.is_file():
            raise RuntimeError(f"training did not produce {checkpoint_path}")
        runs.append(
            {
                "seed": seed,
                "experiment_id": run_id,
                "output_dir": str(run_dir),
                "checkpoint": str(checkpoint_path),
                "best_loss": metrics["best_loss"],
                "final_loss": metrics["final_loss"],
                "steps_completed": metrics["steps_completed"],
                "dataset_manifest": metrics.get("dataset_manifest"),
            }
        )

    payload = {
        "base_experiment_id": base_id,
        "config": str(config_path),
        "seeds": seeds,
        "runs": runs,
    }
    write_summary(output_root, payload)
    return payload


def main():
    args = parse_args()
    summary = run_multiseed(
        args.config,
        args.seeds,
        args.output_root,
        device=args.device,
        max_steps=args.max_steps,
    )
    print("Multi-seed training completed")
    print(f"runs: {len(summary['runs'])}")
    print(f"summary: {args.output_root / 'multiseed_summary.json'}")


if __name__ == "__main__":
    main()
