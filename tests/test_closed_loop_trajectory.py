from pathlib import Path
from copy import deepcopy

import torch
import pytest

from src.core.closed_loop_trajectory import ReceiverStateCorrector
from src.core.receiver_closed_loop import evolve_receiver_with_closed_loop_corrector
from src.pipelines.oracle_experiment import load_yaml
from src.pipelines.closed_loop_trajectory import validate_closed_loop_contract
from src.pipelines.closed_loop_trajectory import (
    _build_source_encoder,
    build_closed_loop_bridge,
    corrector_state_sha256,
    prepare_paired_initialization,
)
from src.core.packet_bundle import sha256_file


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "config" / "LIP-H0-017_closed_loop_trajectory_corrector.yaml"
PARENT = ROOT / "config" / "LIP-PROTO-014_source_conditioned_residual_packet.yaml"
LEARNED = ROOT / "experiments" / "registry" / "LIP-H0-016_hard_negative_replication.json"
FUNCTIONAL = ROOT / "experiments" / "registry" / "LIP-EVAL-037_oracle_native_packet_blend_screen.json"
SOURCE = ROOT / "experiments" / "registry" / "LIP-H0-015_hard_negative_batches.json"


def _validate(experiment):
    validate_closed_loop_contract(
        experiment,
        load_yaml(PARENT),
        experiment_path=EXPERIMENT,
        parent_path=PARENT,
        learned_registry_path=LEARNED,
        functional_registry_path=FUNCTIONAL,
        source_registry_path=SOURCE,
    )


class AddLayer(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 1.0


class ToyReceiver:
    def __init__(self):
        self.model = type(
            "Backbone",
            (),
            {"layers": torch.nn.ModuleList([AddLayer() for _ in range(4)])},
        )()

    def __call__(self, input_ids, attention_mask, **kwargs):
        hidden = input_ids.float().unsqueeze(-1)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return type("Output", (), {"logits": hidden})()


class ScalarCorrector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, protocol_code, live_normalized, *, layer_index):
        return torch.ones_like(live_normalized) * self.delta


def test_corrector_zero_head_makes_untrained_operator_a_noop():
    corrector = ReceiverStateCorrector(
        target_width=4,
        target_layers=2,
        target_positions=3,
        bridge_width=4,
        attention_heads=2,
        feedforward_width=8,
        decoder_blocks=1,
        dropout=0.0,
        condition_on_live_state=True,
    )
    delta = corrector(
        torch.randn(2, 5, 4),
        torch.randn(2, 3, 4),
        layer_index=1,
    )
    assert torch.count_nonzero(delta).item() == 0


def test_state_blind_corrector_ignores_live_packet():
    corrector = ReceiverStateCorrector(
        target_width=4,
        target_layers=2,
        target_positions=3,
        bridge_width=4,
        attention_heads=2,
        feedforward_width=8,
        decoder_blocks=1,
        dropout=0.0,
        condition_on_live_state=False,
    )
    torch.nn.init.normal_(corrector.delta_head.weight)
    code = torch.randn(2, 5, 4)
    first = corrector(code, torch.randn(2, 3, 4), layer_index=0)
    second = corrector(code, torch.randn(2, 3, 4), layer_index=0)
    assert torch.allclose(first, second)


def test_closed_loop_updates_accumulate_through_receiver_evolution():
    corrector = ScalarCorrector()
    states = evolve_receiver_with_closed_loop_corrector(
        ToyReceiver(),
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        },
        positions=torch.tensor([2]),
        protocol_code=torch.zeros(1, 1, 1),
        corrector=corrector,
        scaffold=torch.zeros(3, 1, 1),
        site_scale=torch.ones(3, 1),
        layer_indices=[0, 1, 2],
    )
    incoming = states["incoming_before_correction"][:, :, 0, 0]
    corrected = states["residual_input"][:, :, 0, 0]
    assert incoming.tolist() == [[3.0, 4.5, 6.0]]
    assert corrected.tolist() == [[3.5, 5.0, 6.5]]
    corrected.sum().backward()
    assert corrector.delta.grad is not None
    assert float(corrector.delta.grad) > 3.0


def test_h0_017_contract_keeps_confirmation_closed():
    experiment = load_yaml(EXPERIMENT)
    _validate(experiment)
    assert experiment["experiment_id"] == "LIP-H0-017"
    assert experiment["variants"]["primary"] == "closed_loop_live"
    assert experiment["variants"]["control"] == "open_loop_zero_live"
    assert experiment["source_encoder"]["freeze_parameters"] is True
    assert experiment["receiver"]["freeze_all_parameters"] is True
    assert experiment["confirmation"]["status"] == "prohibited_in_H0-017"
    assert experiment["confirmation"]["eval_038_execution_authorized"] is False


def test_h0_017_contract_rejects_a_state_aware_control():
    experiment = deepcopy(load_yaml(EXPERIMENT))
    experiment["variants"]["systems"]["open_loop_zero_live"][
        "condition_on_live_state"
    ] = True
    try:
        _validate(experiment)
    except ValueError as exc:
        assert "control system" in str(exc)
    else:
        raise AssertionError("state-aware control drift should be rejected")


def test_real_bridge_builder_pairs_initial_weights_before_training(tmp_path):
    experiment = deepcopy(load_yaml(EXPERIMENT))
    experiment["source_encoder"].update(
        protocol_slots=3, bridge_width=4, attention_heads=2,
        feedforward_width=8, encoder_blocks=1,
    )
    experiment["corrector"].update(
        target_layers=2, target_positions=3, target_width=4,
        bridge_width=4, attention_heads=2, feedforward_width=8,
    )
    shape = (2, 3, 4)
    encoder = _build_source_encoder(experiment, shape)
    source_path = tmp_path / "source.pt"
    torch.save({"source_shape": shape, "model_state": {
        "encoder." + name: tensor for name, tensor in encoder.state_dict().items()
    }}, source_path)
    experiment["predecessors"]["source_encoder_checkpoint"]["sha256"] = sha256_file(source_path)

    def build(variant, unrelated_seed):
        torch.manual_seed(unrelated_seed)
        torch.randn(137)
        return build_closed_loop_bridge(
            experiment, source_shape=shape,
            source_checkpoint_path=source_path, variant_name=variant,
        )

    control = build("open_loop_zero_live", 19)
    treatment = build("closed_loop_live", 97)
    assert control.corrector.condition_on_live_state is False
    assert treatment.corrector.condition_on_live_state is True
    for name, tensor in control.corrector.state_dict().items():
        assert torch.equal(tensor, treatment.corrector.state_dict()[name]), name
    shared_path = tmp_path / "paired_initialization.pt"
    kwargs = dict(seed=4007, experiment_config_sha256="a" * 64,
                  source_encoder_checkpoint_sha256=sha256_file(source_path))
    first = prepare_paired_initialization(control.corrector, shared_path, **kwargs)
    original_bytes = shared_path.read_bytes()
    second = prepare_paired_initialization(treatment.corrector, shared_path, **kwargs)
    assert first == second
    assert shared_path.read_bytes() == original_bytes
    experiment["training"]["seed"] = 4008
    different = build("closed_loop_live", 97)
    assert corrector_state_sha256(different.corrector.state_dict()) != first["corrector_state_sha256"]
    with pytest.raises(ValueError, match="seeded weights differ"):
        prepare_paired_initialization(different.corrector, shared_path, **kwargs)
    assert shared_path.read_bytes() == original_bytes

    corrupt = torch.load(shared_path, weights_only=True)
    corrupt["corrector_state"]["delta_head.bias"][0] = 1.0
    torch.save(corrupt, shared_path)
    with pytest.raises(ValueError, match="tensor checksum differs"):
        prepare_paired_initialization(treatment.corrector, shared_path, **kwargs)
