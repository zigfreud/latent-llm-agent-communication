"""Trainable state-conditioned corrections for frozen receiver trajectories."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.core.packet_bridge import SourcePacketEncoder


class ReceiverStateCorrector(nn.Module):
    """Predict a normalized receiver delta from source code and live state."""

    def __init__(
        self,
        *,
        target_width: int,
        target_layers: int,
        target_positions: int,
        bridge_width: int,
        attention_heads: int,
        feedforward_width: int,
        decoder_blocks: int,
        dropout: float,
        condition_on_live_state: bool,
    ) -> None:
        super().__init__()
        if target_width <= 0 or target_layers <= 0 or target_positions <= 0:
            raise ValueError("target dimensions must be positive")
        if bridge_width <= 0 or bridge_width % attention_heads:
            raise ValueError("bridge_width must be positive and divisible by heads")
        if decoder_blocks <= 0:
            raise ValueError("decoder_blocks must be positive")
        self.target_width = int(target_width)
        self.target_layers = int(target_layers)
        self.target_positions = int(target_positions)
        self.bridge_width = int(bridge_width)
        self.condition_on_live_state = bool(condition_on_live_state)

        self.live_projection = nn.Linear(target_width, bridge_width, bias=False)
        self.live_normalization = nn.LayerNorm(bridge_width)
        self.layer_embeddings = nn.Embedding(target_layers, bridge_width)
        self.position_embeddings = nn.Embedding(target_positions, bridge_width)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=bridge_width,
                    nhead=attention_heads,
                    dim_feedforward=feedforward_width,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(decoder_blocks)
            ]
        )
        self.output_normalization = nn.LayerNorm(bridge_width)
        self.delta_head = nn.Linear(bridge_width, target_width)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        protocol_code: torch.Tensor,
        live_normalized: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        if not isinstance(protocol_code, torch.Tensor) or protocol_code.ndim != 3:
            raise ValueError("protocol_code must have [batch, slots, bridge_width]")
        if protocol_code.shape[-1] != self.bridge_width:
            raise ValueError("protocol_code width differs from corrector width")
        if not isinstance(live_normalized, torch.Tensor) or live_normalized.ndim != 3:
            raise ValueError("live_normalized must have [batch, positions, width]")
        if tuple(live_normalized.shape[:2]) != (
            protocol_code.shape[0],
            self.target_positions,
        ) or live_normalized.shape[-1] != self.target_width:
            raise ValueError("live receiver packet shape differs from corrector contract")
        layer = int(layer_index)
        if layer < 0 or layer >= self.target_layers:
            raise ValueError("layer_index is outside the corrector depth")

        if self.condition_on_live_state:
            live_code = self.live_normalization(
                self.live_projection(live_normalized)
            )
        else:
            live_code = torch.zeros(
                live_normalized.shape[0],
                self.target_positions,
                self.bridge_width,
                device=live_normalized.device,
                dtype=protocol_code.dtype,
            )
        positions = torch.arange(
            self.target_positions, device=live_normalized.device
        )
        query = (
            live_code
            + self.layer_embeddings.weight[layer][None, None, :]
            + self.position_embeddings(positions)[None, :, :]
        )
        for block in self.blocks:
            query = block(query, protocol_code)
        return self.delta_head(self.output_normalization(query))


class ClosedLoopTrajectoryBridge(nn.Module):
    """Frozen source encoder plus a trainable sequential receiver corrector."""

    def __init__(
        self,
        encoder: SourcePacketEncoder,
        corrector: ReceiverStateCorrector,
    ) -> None:
        super().__init__()
        if encoder.bridge_width != corrector.bridge_width:
            raise ValueError("encoder and corrector bridge widths must match")
        self.encoder = encoder
        self.corrector = corrector

    def freeze_encoder(self) -> None:
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def encode(self, source_packet: torch.Tensor) -> torch.Tensor:
        return self.encoder(source_packet)

    def correction(
        self,
        protocol_code: torch.Tensor,
        live_normalized: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        return self.corrector(
            protocol_code,
            live_normalized,
            layer_index=layer_index,
        )
