import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

DATASET_PATH = os.path.join("datasets", "dataset_vectors_1k.pt")
DEVICE = "cpu"
EPOCHS = 20
BATCH_SIZE = 32

print(f"⚖️  Starting Cross-Validation (Train/Test Split)...")


class VectorDataset(Dataset):
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Dataset not found at {path}")
        self.data = torch.load(path)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]['src_vector'].squeeze(0).float(), self.data[idx]['tgt_vector'].squeeze(0).float()


full_dataset = VectorDataset(DATASET_PATH)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"✅ Data: {len(full_dataset)} Total | {train_size} Train | {test_size} Test")


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2048, 512), nn.LayerNorm(512), nn.GELU())
        self.dec = nn.Sequential(nn.Linear(512, 4096), nn.LayerNorm(4096))
    def forward(self, x): return self.dec(self.enc(x))


def evaluate(model, loader):
    model.eval()
    total_cos = 0
    total_mse = 0
    mse_fn = nn.MSELoss()
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            pred = model(src)
            total_cos += torch.nn.functional.cosine_similarity(pred, tgt).mean().item()
            total_mse += mse_fn(pred, tgt).item()
    return total_cos / len(loader), total_mse / len(loader)


models_config = [
    {"name": "Baseline (Naive MSE)", "loss": "mse"},
    {"name": "Geometric (Cosine)", "loss": "cos"},
    {"name": "LIP Protocol (Hybrid)", "loss": "hybrid"}
]

results_table = []
mse_fn = nn.MSELoss()
cos_fn = nn.CosineEmbeddingLoss()

for config in models_config:
    print(f"\n🏃 Training: {config['name']}...")
    model = Adapter().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(EPOCHS):
        for src, tgt in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            ones = torch.ones(src.shape[0]).to(DEVICE)
            opt.zero_grad()
            pred = model(src)

            if config['loss'] == 'mse': loss = mse_fn(pred, tgt)
            elif config['loss'] == 'cos': loss = cos_fn(pred, tgt, ones)
            else: loss = mse_fn(pred, tgt) + (2.0 * cos_fn(pred, tgt, ones))

            loss.backward()
            opt.step()

    test_sim, test_mse = evaluate(model, test_loader)
    print(f"   🎯 Final Result (Test Set): Sim={test_sim:.4f} | MSE={test_mse:.4f}")

    results_table.append({
        "Architecture": "Dual-Encoder (Bottleneck)",
        "Loss Strategy": config['name'],
        "Cosine Similarity (Higher is better)": f"{test_sim:.4f}",
        "MSE Loss (Lower is better)": f"{test_mse:.4f}"
    })


df = pd.DataFrame(results_table)
csv_path = os.path.join("experiments", "logs", "final_results_table.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
df.to_csv(csv_path, index=False)

print("\n" + "="*50)
print(f"📊 Table saved to: {csv_path}")
print("="*50)
print(df.to_markdown(index=False))