import os
import yaml
import glob
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.core.models import LIPAdapter
from src.core.loss import HybridContrastiveLoss
from src.core.utils import get_device, set_seed, setup_logger


class ShardDataset(Dataset):
    def __init__(self, filepath):
        try:
            # Forçamos map_location='cpu' para evitar erros de device entre torch versions
            data = torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"❌ ARQUIVO CORROMPIDO: {filepath} | Erro: {e}")
            self.data = []
            return

        if isinstance(data, list):
            self.data = data
        else:
            # AQUI ESTÁ O SEGREDO: Vamos ver o que tem dentro
            print(f"⚠️ TIPO ERRADO em {os.path.basename(filepath)}: Recebi {type(data)} em vez de 'list'.")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Garante que estamos retornando tensores float32, não importa como foi salvo
        return item['src_vector'].squeeze().float(), item['tgt_vector'].squeeze().float()


def load_sharded_dataset(directory):
    files = glob.glob(os.path.join(directory, "*.pt"))

    if not files:
        raise FileNotFoundError(f"❌ Nenhum shard (.pt) encontrado em {directory}")

    print(f"📂 Indexando {len(files)} shards...")
    valid_files = [f for f in files if os.path.getsize(f) > 0]
    return ConcatDataset([ShardDataset(f) for f in valid_files])


def train(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['output_dir'], exist_ok=True)
    logger = setup_logger(cfg['output_dir'])
    device = get_device(cfg.get('device', 'auto'))
    set_seed(cfg.get('seed', 42))

    logger.info(f"🔧 Iniciando Treino: {cfg['experiment_name']}")
    logger.info(f"🖥️  Hardware: {device}")

    dataset = load_sharded_dataset(cfg['data']['dataset_path'])
    loader = DataLoader(
        dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=(str(device) == 'cuda'),
        drop_last=True
    )
    logger.info(f"📊 Total samples: {len(dataset)}")

    model = LIPAdapter(
        input_dim=cfg['model']['input_dim'],
        output_dim=cfg['model']['output_dim']
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg['training']['learning_rate']),
        weight_decay=cfg['training'].get('weight_decay', 0.01)
    )

    criterion = HybridContrastiveLoss(
        temperature=cfg['loss']['temperature']
    )

    start_epoch = 0
    best_loss = float('inf')
    ckpt_path = os.path.join(cfg['output_dir'], "last_checkpoint.pth")

    if os.path.exists(ckpt_path):
        logger.info(f"🔄 Resuming by... {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))

    model.train()

    for epoch in range(start_epoch, cfg['training']['epochs']):
        epoch_loss = 0
        epoch_acc = 0
        steps = 0

        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src)
            loss, acc = criterion(output, tgt)
            mse = 0.0
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            steps += 1

        avg_loss = epoch_loss / steps
        avg_acc = epoch_acc / steps
        logger.info(f"Ep {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.2f}%")

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_loss': best_loss,
            'config': cfg
        }, ckpt_path)


        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(cfg['output_dir'], "best_model.pth"))
            logger.info("🌟 New best model saved.")

    logger.info("🏁 Process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    train(args.config)