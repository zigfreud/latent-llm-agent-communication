import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()


    def forward(self, source, target):
        """
        source: original adapters output
        target:  target models vectors
        """

        source_norm = F.normalize(source, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        
        logits = torch.matmul(source_norm, target_norm.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        loss = self.cross_entropy(logits, labels)
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            
        return loss, acc