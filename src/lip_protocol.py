import json
import time
import uuid
import torch
import base64
import numpy as np
from typing import List
from dataclasses import dataclass, asdict


@dataclass
class LIPHeader:
    protocol_version: str = "1.0"
    packet_id: str = ""
    timestamp: float = 0.0
    source_model: str = "Unknown"
    target_model_family: str = "Llama-3"
    vector_dim: int = 512
    energy_signature: float = 1.0
    intent_class: str = "general"
    encrypted: bool = False

@dataclass
class LIPPayload:
    data_b64: str = ""
    dtype: str = "float32"
    shape: List[int] = None

class LIPPacket:
    """
    The Universal Envelope for semantic intent transport.
    """
    def __init__(self, vector: torch.Tensor, source_model: str, intent: str = "general"):
        self.header = LIPHeader()
        self.payload = LIPPayload()

        # Header Metadata
        self.header.packet_id = str(uuid.uuid4())
        self.header.timestamp = time.time()
        self.header.source_model = source_model

        # Physics: Calculate energy before compression
        self.header.energy_signature = vector.norm(p=2).item()
        self.header.intent_class = intent

        self._pack_vector(vector)

    def _pack_vector(self, vector: torch.Tensor):
        """Converts PyTorch Tensor -> Bytes -> Base64"""
        vec_np = vector.detach().cpu().float().numpy()

        self.header.vector_dim = vec_np.shape[-1]
        self.payload.dtype = str(vec_np.dtype)
        self.payload.shape = list(vec_np.shape)

        buffer = vec_np.tobytes()
        self.payload.data_b64 = base64.b64encode(buffer).decode('utf-8')

    def to_json(self) -> str:
        """Generates the final packet for network transmission."""
        packet_dict = {
            "header": asdict(self.header),
            "payload": asdict(self.payload)
        }
        return json.dumps(packet_dict)

    @staticmethod
    def from_json(json_str: str) -> 'LIPPacket':
        """Reconstructs the packet on the receiver side."""
        data = json.loads(json_str)
        packet = LIPPacket.__new__(LIPPacket)
        packet.header = LIPHeader(**data['header'])
        packet.payload = LIPPayload(**data['payload'])
        return packet

    def unpack_vector(self, device='cpu') -> torch.Tensor:
        """Reconstructs the PyTorch Tensor from Base64."""
        buffer = base64.b64decode(self.payload.data_b64)
        dtype = np.dtype(self.payload.dtype)

        vec_np = np.frombuffer(buffer, dtype=dtype)
        vec_np = vec_np.reshape(self.payload.shape)

        return torch.from_numpy(vec_np).to(device)

    def calibrate_energy(self, target_reference_energy: float, vector: torch.Tensor) -> torch.Tensor:
        """
        LIP Magic: Adjusts received vector energy to match target model physics.
        """
        current_energy = vector.norm(p=2)
        # Cross-Correction Factor
        scale_factor = target_reference_energy / (current_energy + 1e-6)
        return vector * scale_factor