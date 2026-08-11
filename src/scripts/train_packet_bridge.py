"""CLI entrypoint for cached LIP packet bridge training."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipelines.packet_bridge import train_packet_bridge


def parse_args():
    parser = argparse.ArgumentParser(description="Train one LIP packet bridge replica")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main():
    result = train_packet_bridge(parse_args().config)
    print("LIP packet bridge training completed")
    print(f"experiment_id: {result['experiment_id']}")
    print(f"model_kind: {result['model_kind']}")
    print(f"best_step: {result['best_step']}")
    print(f"development_gate_passed: {result['development_gate']['passed']}")


if __name__ == "__main__":
    main()
