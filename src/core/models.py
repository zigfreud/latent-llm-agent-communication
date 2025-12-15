import numpy
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        return x + self.net(x)


class LIPAdapter(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=4096):
        super().__init__()
        intermediate_dim = int((input_dim - (hidden_dim/2)))
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.processor = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        
    def forward(self, x):
        x = x.to(torch.float32)
        z = self.encoder(x)
        z = self.processor(z)
        out = self.decoder(z)
        return out
    
