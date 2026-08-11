"""Learned source-to-receiver packet bridges for the LIP protocol."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _validate_packet_shape(
    packet: torch.Tensor,
    *,
    layers: int,
    positions: int,
    width: int,
    label: str,
) -> None:
    if not isinstance(packet, torch.Tensor) or packet.ndim != 4:
        raise ValueError(f"{label} must have shape [batch, layers, positions, width]")
    expected = (layers, positions, width)
    if tuple(packet.shape[1:]) != expected:
        raise ValueError(
            f"{label} trailing shape {tuple(packet.shape[1:])} does not match {expected}"
        )


def _decoder_block(
    *,
    width: int,
    attention_heads: int,
    feedforward_width: int,
    dropout: float,
) -> nn.TransformerDecoderLayer:
    return nn.TransformerDecoderLayer(
        d_model=width,
        nhead=attention_heads,
        dim_feedforward=feedforward_width,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class SourcePacketEncoder(nn.Module):
    """Compress multi-layer source states into a fixed LIP communication code."""

    def __init__(
        self,
        *,
        source_width: int = 2048,
        source_layers: int = 24,
        source_positions: int = 32,
        protocol_slots: int = 32,
        bridge_width: int = 512,
        attention_heads: int = 8,
        feedforward_width: int = 2048,
        decoder_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if source_layers <= 0 or source_positions <= 0 or protocol_slots <= 0:
            raise ValueError("source layers, positions, and protocol slots must be positive")
        if bridge_width % attention_heads:
            raise ValueError("bridge_width must be divisible by attention_heads")
        if decoder_blocks <= 0:
            raise ValueError("decoder_blocks must be positive")

        self.source_width = int(source_width)
        self.source_layers = int(source_layers)
        self.source_positions = int(source_positions)
        self.protocol_slots = int(protocol_slots)
        self.bridge_width = int(bridge_width)

        self.source_projection = nn.Linear(source_width, bridge_width)
        self.source_normalization = nn.LayerNorm(bridge_width)
        self.source_layer_embeddings = nn.Embedding(source_layers, bridge_width)
        self.source_position_embeddings = nn.Embedding(source_positions, bridge_width)
        self.protocol_queries = nn.Parameter(
            torch.empty(protocol_slots, bridge_width)
        )
        self.blocks = nn.ModuleList(
            [
                _decoder_block(
                    width=bridge_width,
                    attention_heads=attention_heads,
                    feedforward_width=feedforward_width,
                    dropout=dropout,
                )
                for _ in range(decoder_blocks)
            ]
        )
        self.output_normalization = nn.LayerNorm(bridge_width)
        nn.init.normal_(self.protocol_queries, mean=0.0, std=bridge_width**-0.5)

    def forward(self, source_packet: torch.Tensor) -> torch.Tensor:
        _validate_packet_shape(
            source_packet,
            layers=self.source_layers,
            positions=self.source_positions,
            width=self.source_width,
            label="source_packet",
        )
        projected = self.source_normalization(self.source_projection(source_packet))
        layer_ids = torch.arange(self.source_layers, device=source_packet.device)
        position_ids = torch.arange(self.source_positions, device=source_packet.device)
        projected = (
            projected
            + self.source_layer_embeddings(layer_ids)[None, :, None, :]
            + self.source_position_embeddings(position_ids)[None, None, :, :]
        )
        memory = projected.flatten(start_dim=1, end_dim=2)
        code = self.protocol_queries[None, :, :].expand(source_packet.shape[0], -1, -1)
        for block in self.blocks:
            code = block(code, memory)
        return self.output_normalization(code)


class ReceiverPacketDecoder(nn.Module):
    """Decode a fixed LIP code into normalized receiver residual sites."""

    def __init__(
        self,
        *,
        target_width: int = 4096,
        target_layers: int = 8,
        target_positions: int = 24,
        bridge_width: int = 512,
        attention_heads: int = 8,
        feedforward_width: int = 2048,
        decoder_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if target_layers <= 0 or target_positions <= 0:
            raise ValueError("target layers and positions must be positive")
        if bridge_width % attention_heads:
            raise ValueError("bridge_width must be divisible by attention_heads")
        if decoder_blocks <= 0:
            raise ValueError("decoder_blocks must be positive")

        self.target_width = int(target_width)
        self.target_layers = int(target_layers)
        self.target_positions = int(target_positions)
        self.bridge_width = int(bridge_width)

        self.target_layer_embeddings = nn.Embedding(target_layers, bridge_width)
        self.target_position_embeddings = nn.Embedding(target_positions, bridge_width)
        self.blocks = nn.ModuleList(
            [
                _decoder_block(
                    width=bridge_width,
                    attention_heads=attention_heads,
                    feedforward_width=feedforward_width,
                    dropout=dropout,
                )
                for _ in range(decoder_blocks)
            ]
        )
        self.output_normalization = nn.LayerNorm(bridge_width)
        self.residual_head = nn.Linear(bridge_width, target_width)

    def _queries(self, *, device: torch.device, batch_size: int) -> torch.Tensor:
        layer_ids = torch.arange(self.target_layers, device=device)
        position_ids = torch.arange(self.target_positions, device=device)
        queries = (
            self.target_layer_embeddings(layer_ids)[:, None, :]
            + self.target_position_embeddings(position_ids)[None, :, :]
        )
        return queries.flatten(start_dim=0, end_dim=1)[None, :, :].expand(
            batch_size, -1, -1
        )

    def forward(self, protocol_code: torch.Tensor) -> torch.Tensor:
        if not isinstance(protocol_code, torch.Tensor) or protocol_code.ndim != 3:
            raise ValueError("protocol_code must have shape [batch, slots, bridge_width]")
        if protocol_code.shape[-1] != self.bridge_width:
            raise ValueError(
                f"protocol_code width {protocol_code.shape[-1]} does not match "
                f"{self.bridge_width}"
            )
        decoded = self._queries(
            device=protocol_code.device,
            batch_size=protocol_code.shape[0],
        )
        for block in self.blocks:
            decoded = block(decoded, protocol_code)
        residual = self.residual_head(self.output_normalization(decoded))
        return residual.reshape(
            protocol_code.shape[0],
            self.target_layers,
            self.target_positions,
            self.target_width,
        )


class LIPPacketBridge(nn.Module):
    """Composable sender encoder and receiver decoder for one learned endpoint pair."""

    def __init__(self, encoder: SourcePacketEncoder, decoder: ReceiverPacketDecoder) -> None:
        super().__init__()
        if encoder.bridge_width != decoder.bridge_width:
            raise ValueError("encoder and decoder bridge widths must match")
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, source_packet: torch.Tensor) -> torch.Tensor:
        return self.encoder(source_packet)

    def decode(self, protocol_code: torch.Tensor) -> torch.Tensor:
        return self.decoder(protocol_code)

    def forward(self, source_packet: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(source_packet))


class StructuredLinearPacketBridge(nn.Module):
    """Input-linear packet baseline with content-independent source-site mixing."""

    def __init__(
        self,
        *,
        source_width: int = 2048,
        source_layers: int = 24,
        source_positions: int = 32,
        target_width: int = 4096,
        target_layers: int = 8,
        target_positions: int = 24,
    ) -> None:
        super().__init__()
        self.source_width = int(source_width)
        self.source_layers = int(source_layers)
        self.source_positions = int(source_positions)
        self.target_width = int(target_width)
        self.target_layers = int(target_layers)
        self.target_positions = int(target_positions)
        self.source_sites = self.source_layers * self.source_positions
        self.target_sites = self.target_layers * self.target_positions

        self.source_site_mixing = nn.Parameter(
            torch.empty(self.target_sites, self.source_sites)
        )
        self.shared_projection = nn.Linear(source_width, target_width, bias=False)
        self.target_scale = nn.Parameter(torch.ones(self.target_sites, target_width))
        self.target_bias = nn.Parameter(torch.zeros(self.target_sites, target_width))
        nn.init.normal_(
            self.source_site_mixing,
            mean=0.0,
            std=1.0 / math.sqrt(self.source_sites),
        )

    def forward(self, source_packet: torch.Tensor) -> torch.Tensor:
        _validate_packet_shape(
            source_packet,
            layers=self.source_layers,
            positions=self.source_positions,
            width=self.source_width,
            label="source_packet",
        )
        flattened = source_packet.flatten(start_dim=1, end_dim=2)
        mixed = torch.einsum("sm,bmd->bsd", self.source_site_mixing, flattened)
        output = self.shared_projection(mixed)
        output = output * self.target_scale[None, :, :] + self.target_bias[None, :, :]
        return output.reshape(
            source_packet.shape[0],
            self.target_layers,
            self.target_positions,
            self.target_width,
        )


def reconstruct_target_packet(
    normalized_residual: torch.Tensor,
    scaffold: torch.Tensor,
    site_scale: torch.Tensor,
) -> torch.Tensor:
    """Undo site normalization and add the training-only receiver scaffold."""

    if normalized_residual.ndim != 4:
        raise ValueError("normalized_residual must have rank four")
    if tuple(scaffold.shape) != tuple(normalized_residual.shape[1:]):
        raise ValueError("scaffold shape must match one target packet")
    if tuple(site_scale.shape) != tuple(normalized_residual.shape[1:3]):
        raise ValueError("site_scale shape must match target layer and position axes")
    return (
        scaffold[None, :, :, :]
        + normalized_residual * site_scale[None, :, :, None]
    )
