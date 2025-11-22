import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

DATASET_PATH = os.path.join("datasets", "dataset_vectors_1k.pt")
OUTPUT_DIR = os.path.join("experiments", "logs")
DEVICE = "cpu"
EPOCHS = 15
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"🔥 Starting Scientific Benchmark on {DEVICE}...")

class VectorDataset(Dataset):
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Dataset not found at {path}. Please run 'experiments/dataset_generator.py' first.")
        self.data = torch.load(path)
        print(f"✅ Loaded {len(self.data)} samples from {path}")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]['src_vector'].squeeze(0).float(), self.data[idx]['tgt_vector'].squeeze(0).float()


dataset = VectorDataset(DATASET_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2048, 512), nn.LayerNorm(512), nn.GELU())
        self.dec = nn.Sequential(nn.Linear(512, 4096), nn.LayerNorm(4096))
    def forward(self, x): return self.dec(self.enc(x))


experiments = [
    {"name": "Gen1_MSE_Only", "loss_type": "mse"},
    {"name": "Gen2_Cosine_Only", "loss_type": "cosine"},
    {"name": "Gen4_LIP_Hybrid", "loss_type": "hybrid"}
]

mse_fn = nn.MSELoss()
cos_fn = nn.CosineEmbeddingLoss()

for exp in experiments:
    print(f"\n🧪 Running Experiment: {exp['name']}...")

    model = Adapter().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=0.001)
    metrics = []

    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_cos_sim = 0

        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            target_ones = torch.ones(src.shape[0]).to(DEVICE)

            opt.zero_grad()
            pred = model(src)

            # Loss Switching Logic
            if exp['loss_type'] == 'mse':
                loss = mse_fn(pred, tgt)
            elif exp['loss_type'] == 'cosine':
                loss = cos_fn(pred, tgt, target_ones)
            elif exp['loss_type'] == 'hybrid':
                loss = mse_fn(pred, tgt) + (2.0 * cos_fn(pred, tgt, target_ones))

            loss.backward()
            opt.step()

            epoch_loss += loss.item()

            # Validation Metric (Cosine Similarity)
            with torch.no_grad():
                sim = torch.nn.functional.cosine_similarity(pred, tgt).mean().item()
                epoch_cos_sim += sim

        avg_loss = epoch_loss / len(loader)
        avg_sim = epoch_cos_sim / len(loader)

        metrics.append({"epoch": epoch+1, "loss": avg_loss, "similarity": avg_sim})
        if (epoch+1) % 5 == 0:
            print(f"   Ep {epoch+1}: Loss {avg_loss:.4f} | Sim {avg_sim:.4f}")

    # Save Raw Data
    outfile = os.path.join(OUTPUT_DIR, f"{exp['name']}.json")
    with open(outfile, "w") as f:
        json.dump(metrics, f, indent=4)

print(f"\n✅ Benchmark data saved to '{OUTPUT_DIR}/'. Ready for plotting.")