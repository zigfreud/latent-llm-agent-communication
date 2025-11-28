import torch.nn as nn

class LIPAdapter(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=4096, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),

            nn.Linear(512, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, output_dim)
        )


    def forward(self, x):
        return self.net(x)