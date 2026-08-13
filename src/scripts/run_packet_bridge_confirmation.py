"""Generate the resumable LIP-PROTO-014 functional confirmation grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipelines.packet_confirmation import run_packet_bridge_confirmation


DEFAULT_CONFIG = Path(
    "config/LIP-PROTO-014_source_conditioned_residual_packet.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--training-bundle", type=Path, required=True)
    parser.add_argument("--confirmation-bundle", type=Path, required=True)
    parser.add_argument("--matrix-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prediction-batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-new-records",
        type=int,
        default=None,
        help="Operational chunk limit; the same frozen grid remains nonclaim until complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_packet_bridge_confirmation(
        args.config,
        training_bundle_dir=args.training_bundle,
        confirmation_bundle_dir=args.confirmation_bundle,
        matrix_summary_path=args.matrix_summary,
        output_path=args.output,
        device=args.device,
        prediction_batch_size=args.prediction_batch_size,
        resume=args.resume,
        overwrite=args.overwrite,
        max_new_records=args.max_new_records,
    )
    print("LIP packet functional confirmation generation finished")
    print(f"records: {metadata['records']}/{metadata['expected_records']}")
    print(f"complete: {metadata['complete']}")
    print(f"claim_eligible: {metadata['claim_eligible']}")
    print(f"metadata: {args.output.with_suffix('.metadata.json')}")


if __name__ == "__main__":
    main()
