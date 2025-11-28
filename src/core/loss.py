import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, lambda_mse=1.0):
        super().__init__()
        self.temperature = temperature
        self.lambda_mse = lambda_mse
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()


    def forward(self, features_raw, targets_raw):
        """
        features_raw: original adapters output
        targets_raw:  target models raw vectors
        """

        feats_norm = F.normalize(features_raw, p=2, dim=1)
        targets_norm = F.normalize(targets_raw, p=2, dim=1)
        logits = torch.matmul(feats_norm, targets_norm.T) / self.temperature
        labels = torch.arange(logits.shape[0]).to(logits.device)

        loss_nce = self.cross_entropy(logits, labels)
        loss_mse = self.mse(features_raw, targets_raw)
        total_loss = loss_nce + (self.lambda_mse * loss_mse)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()

        return total_loss, loss_nce, loss_mse, acc