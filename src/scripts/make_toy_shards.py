import argparse
from pathlib import Path

import torch


DEFAULT_OUTPUT_DIR = Path("datasets/LIP-H0-001/smoke_shards")


def build_samples(num_samples, input_dim, output_dim, seed):
    generator = torch.Generator().manual_seed(seed)
    src_vectors = torch.randn(num_samples, input_dim, generator=generator)
    projection = torch.randn(input_dim, output_dim, generator=generator) / (input_dim ** 0.5)
    tgt_vectors = (src_vectors @ projection) + (
        0.01 * torch.randn(num_samples, output_dim, generator=generator)
    )

    return [
        {
            "src_vector": src_vectors[index],
            "tgt_vector": tgt_vectors[index],
        }
        for index in range(num_samples)
    ]


def write_shard(output_dir, num_samples, input_dim, output_dim, seed):
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shard_path = output_path / "shard_0.pt"
    torch.save(build_samples(num_samples, input_dim, output_dim, seed), shard_path)
    return shard_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate toy shards for the LIP-H0-001 CPU smoke run.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    shard_path = write_shard(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seed=args.seed,
    )
    print(f"Wrote {shard_path}")


if __name__ == "__main__":
    main()
