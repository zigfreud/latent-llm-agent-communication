import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lab_journal import LabJournal
from torch.utils.data import DataLoader, Dataset

CONFIG = {
    "experiment_name": "Gen4_Code_Specialist",
    "input_file": "datasets/dataset_code_only.pt",
    "device": "cpu",
    "batch_size": 32,
    "epochs": 50,
    "lr": 0.001,
    "architecture": "2048->512->4096"
}

print(f"🔥 Iniciando Experimento Controlado: {CONFIG['experiment_name']}")
journal = LabJournal(CONFIG['experiment_name'], CONFIG)

class VectorDataset(Dataset):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Arquivo {file_path} sumiu! Rode a Data Factory.")
        self.data = torch.load(file_path)
        print(f"✅ {len(self.data)} amostras carregadas.")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]['src_vector'].squeeze(0).float(), self.data[idx]['tgt_vector'].squeeze(0).float()

dataset = VectorDataset(CONFIG['input_file'])
train_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

class HeteroAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2048, 512), nn.LayerNorm(512), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(512, 4096), nn.LayerNorm(4096))
    def forward(self, x): return self.decoder(self.encoder(x))


adapter = HeteroAdapter().to(CONFIG['device'])
optimizer = optim.AdamW(adapter.parameters(), lr=CONFIG['lr'])
mse_loss = nn.MSELoss()
cos_loss = nn.CosineEmbeddingLoss()

print(f"🚀 Treinando Gen-3...")
for epoch in range(CONFIG['epochs']):
    total_loss, total_cos, total_mse = 0, 0, 0

    for src_vec, tgt_vec in train_loader:
        src_vec, tgt_vec = src_vec.to(CONFIG['device']), tgt_vec.to(CONFIG['device'])
        target_ones = torch.ones(src_vec.shape[0]).to(CONFIG['device'])

        optimizer.zero_grad()
        pred_vec = adapter(src_vec)

        l_mse = mse_loss(pred_vec, tgt_vec)
        l_cos = cos_loss(pred_vec, tgt_vec, target_ones)
        loss = l_mse + (2.0 * l_cos)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += l_mse.item()
        total_cos += l_cos.item()

    avg_loss = total_loss / len(train_loader)
    avg_mse = total_mse / len(train_loader)
    avg_cos = total_cos / len(train_loader)
    journal.log_metric(epoch+1, avg_loss, avg_mse, avg_cos)

    if (epoch+1) % 10 == 0:
        print(f"Época {epoch+1}/{CONFIG['epochs']} | Loss: {avg_loss:.4f} (Cos: {avg_cos:.4f})")
        journal.save_model(adapter, f"checkpoint_ep{epoch+1}.pth")


journal.save_model(adapter, "final_adapter_gen3.pth")
print("\n✅ Experimento Concluído. Dados salvos em 'experiments_log/'.")
print("👉 Copie o arquivo 'final_adapter_gen3.pth' para 'lip_hetero_adapter.pth' para testar na demo.")