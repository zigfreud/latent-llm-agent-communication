import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature=0.07,
        lambda_mse=1.0,
        lambda_reverse_nce=0.0,
        reverse_nce_weight=None,
        lambda_margin=0.0,
        margin_target=0.05,
        lambda_norm=0.0,
        eps=1e-8,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.lambda_mse = float(lambda_mse)
        self.lambda_reverse_nce = (
            reverse_nce_weight if reverse_nce_weight is not None else lambda_reverse_nce
        )
        self.lambda_reverse_nce = float(self.lambda_reverse_nce)
        self.lambda_margin = float(lambda_margin)
        self.margin_target = float(margin_target)
        self.lambda_norm = float(lambda_norm)
        self.eps = float(eps)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    @property
    def has_extended_terms(self):
        return any(
            weight != 0.0
            for weight in (
                self.lambda_reverse_nce,
                self.lambda_margin,
                self.lambda_norm,
            )
        )

    def _margin_metrics(self, cosine):
        batch_size = cosine.shape[0]
        zero = cosine.new_zeros(())
        if batch_size < 2:
            return {
                "margin_loss": zero,
                "offdiag_cosine_mean": None,
                "diagonal_margin_mean": None,
                "hard_negative_cosine_mean": None,
            }

        labels = torch.arange(batch_size, device=cosine.device)
        positive = cosine[labels, labels]
        mask = torch.eye(batch_size, dtype=torch.bool, device=cosine.device)
        negatives = cosine.masked_fill(mask, float("-inf"))

        hardest_row = torch.max(negatives, dim=1).values
        row_margin = positive - hardest_row
        row_loss = F.relu(self.margin_target - row_margin).mean()

        hardest_col = torch.max(negatives, dim=0).values
        col_margin = positive - hardest_col
        col_loss = F.relu(self.margin_target - col_margin).mean()

        return {
            "margin_loss": 0.5 * (row_loss + col_loss),
            "offdiag_cosine_mean": cosine.masked_select(~mask).mean(),
            "diagonal_margin_mean": row_margin.mean(),
            "hard_negative_cosine_mean": hardest_row.mean(),
        }

    def forward(self, features_raw, targets_raw):
        """
        features_raw: original adapters output
        targets_raw:  target models raw vectors
        """

        feats_norm = F.normalize(features_raw, p=2, dim=1)
        targets_norm = F.normalize(targets_raw, p=2, dim=1)
        cosine = torch.matmul(feats_norm, targets_norm.T)
        logits = cosine / self.temperature
        labels = torch.arange(logits.shape[0]).to(logits.device)

        loss_nce_forward = self.cross_entropy(logits, labels)
        loss_nce_reverse = self.cross_entropy(logits.T, labels)
        loss_mse = self.mse(features_raw, targets_raw)
        margin_metrics = self._margin_metrics(cosine)

        pred_norm_scalar = torch.linalg.vector_norm(features_raw, dim=1)
        target_norm_scalar = torch.linalg.vector_norm(targets_raw, dim=1)
        norm_ratio = pred_norm_scalar / (target_norm_scalar + self.eps)
        loss_norm = torch.mean((norm_ratio - 1.0) ** 2)

        total_loss = (
            loss_nce_forward
            + (self.lambda_reverse_nce * loss_nce_reverse)
            + (self.lambda_mse * loss_mse)
            + (self.lambda_margin * margin_metrics["margin_loss"])
            + (self.lambda_norm * loss_norm)
        )

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            cosine_diag_mean = torch.diagonal(cosine).mean()
            norm_ratio_mean = norm_ratio.mean()

        if not self.has_extended_terms:
            return total_loss, loss_nce_forward, loss_mse, acc

        return {
            "total_loss": total_loss,
            "nce_loss": loss_nce_forward,
            "forward_nce_loss": loss_nce_forward,
            "reverse_nce_loss": loss_nce_reverse,
            "mse_loss": loss_mse,
            "margin_loss": margin_metrics["margin_loss"],
            "norm_loss": loss_norm,
            "cosine_diag_mean": cosine_diag_mean,
            "offdiag_cosine_mean": margin_metrics["offdiag_cosine_mean"],
            "diagonal_margin_mean": margin_metrics["diagonal_margin_mean"],
            "hard_negative_cosine_mean": margin_metrics["hard_negative_cosine_mean"],
            "norm_ratio_mean": norm_ratio_mean,
            "accuracy": acc,
        }
