from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

# repo imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core.models import LIPAdapter


@dataclass
class ResultRow:
    n: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mean_cos_pos: float
    mean_cos_neg: float
    mean_margin_pos_neg: float


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iter_shard_items(shard_paths: List[str]):
    """
    Yields dict items with keys: src_vector, tgt_vector.
    Shards can be list[dict] or dict with 'data' etc (we handle list only strictly).
    """
    for p in shard_paths:
        data = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(data, list):
            for item in data:
                if "src_vector" in item and "tgt_vector" in item:
                    yield item
        else:
            # soft fail
            continue


def collect_pairs_from_shards(
    dataset_dir: str,
    n_samples: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shard_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No .pt shards found in: {dataset_dir}")

    rng = random.Random(seed)
    # shuffle shards for sampling diversity
    shard_paths = shard_paths[:]
    rng.shuffle(shard_paths)

    src_list = []
    tgt_list = []

    for item in iter_shard_items(shard_paths):
        src = item["src_vector"]
        tgt = item["tgt_vector"]

        # expected shapes: (1, d) or (d,)
        if isinstance(src, torch.Tensor):
            src = src.squeeze().float().cpu()
        if isinstance(tgt, torch.Tensor):
            tgt = tgt.squeeze().float().cpu()

        if src.ndim != 1 or tgt.ndim != 1:
            continue

        src_list.append(src)
        tgt_list.append(tgt)

        if len(src_list) >= n_samples:
            break

    if len(src_list) < n_samples:
        raise RuntimeError(f"Collected only {len(src_list)} samples < requested {n_samples}")

    src_mat = torch.stack(src_list, dim=0)  # (N, ds)
    tgt_mat = torch.stack(tgt_list, dim=0)  # (N, dt)
    return src_mat, tgt_mat


def load_adapter(ckpt_path: str, device: str) -> LIPAdapter:
    adapter = LIPAdapter().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # allow both pure state_dict and checkpoint dict
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    adapter.load_state_dict(state)
    adapter.eval()
    return adapter


def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: (N, d)
    b: (N, d)
    returns S: (N, N) with cosine similarities
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()


def recall_at_k(sim: torch.Tensor, k: int) -> float:
    """
    sim: (N, N), correct match is i==j
    """
    N = sim.size(0)
    topk = torch.topk(sim, k=k, dim=1).indices  # (N, k)
    correct = torch.arange(N, device=sim.device).unsqueeze(1)  # (N,1)
    hits = (topk == correct).any(dim=1).float().mean().item()
    return float(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True, help="Dir with .pt shards (items with src_vector/tgt_vector).")
    ap.add_argument("--adapter_ckpt", type=str, required=True, help="Path to adapter state_dict (e.g., checkpoints/Gen6.2/adapter.pth).")
    ap.add_argument("--n_samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_csv", type=str, default="results_retrieval_eval.csv")
    ap.add_argument("--out_json", type=str, default="results_retrieval_eval.json")
    args = ap.parse_args()

    set_seed(args.seed)

    print(f"Loading pairs from shards: {args.dataset_dir}")
    src_mat, tgt_mat = collect_pairs_from_shards(args.dataset_dir, args.n_samples, args.seed)
    print(f"Collected N={src_mat.shape[0]} | src_dim={src_mat.shape[1]} | tgt_dim={tgt_mat.shape[1]}")

    device = args.device
    adapter = load_adapter(args.adapter_ckpt, device=device)

    with torch.no_grad():
        src = src_mat.to(device)
        tgt = tgt_mat.to(device)

        mapped = adapter(src)  # (N, tgt_dim) expected
        # Evaluate pure translator fidelity; no energy scaling needed here.
        sim = cosine_matrix(mapped, tgt)  # (N, N)

        r1 = recall_at_k(sim, 1)
        r5 = recall_at_k(sim, 5)
        r10 = recall_at_k(sim, 10)

        # pos/neg cos
        pos = sim.diag()  # (N,)
        # sample one negative per row: choose a random j != i
        N = sim.size(0)
        neg_idx = torch.randint(low=0, high=N - 1, size=(N,), device=device)
        # shift to avoid i==j
        neg_idx = (neg_idx + torch.arange(N, device=device)) % N
        neg = sim[torch.arange(N, device=device), neg_idx]

        mean_pos = pos.mean().item()
        mean_neg = neg.mean().item()
        mean_margin = (pos - neg).mean().item()

    row = ResultRow(
        n=args.n_samples,
        recall_at_1=r1,
        recall_at_5=r5,
        recall_at_10=r10,
        mean_cos_pos=float(mean_pos),
        mean_cos_neg=float(mean_neg),
        mean_margin_pos_neg=float(mean_margin),
    )

    print("\n" + "=" * 60)
    print("Retrieval Alignment Results")
    print("=" * 60)
    print(f"N={row.n}")
    print(f"Recall@1 : {row.recall_at_1*100:.2f}%")
    print(f"Recall@5 : {row.recall_at_5*100:.2f}%")
    print(f"Recall@10: {row.recall_at_10*100:.2f}%")
    print(f"Mean Cos (pos)  : {row.mean_cos_pos:.4f}")
    print(f"Mean Cos (neg)  : {row.mean_cos_neg:.4f}")
    print(f"Mean Margin pos-neg: {row.mean_margin_pos_neg:.4f}")

    # Save CSV (single row)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.__dict__.keys()))
        w.writeheader()
        w.writerow(row.__dict__)

    # Save JSON (+ metadata)
    meta = {
        "dataset_dir": args.dataset_dir,
        "adapter_ckpt": args.adapter_ckpt,
        "seed": args.seed,
        "device": device,
        "torch_version": torch.__version__,
    }
    out = {"metrics": row.__dict__, "meta": meta}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved: {args.out_csv} | {args.out_json}")


if __name__ == "__main__":
    main()
