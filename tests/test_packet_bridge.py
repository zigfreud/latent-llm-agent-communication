import torch

from src.core.packet_bridge import (
    LIPPacketBridge,
    ReceiverPacketDecoder,
    SourcePacketEncoder,
    StructuredLinearPacketBridge,
    reconstruct_target_packet,
)


def _small_bridge():
    encoder = SourcePacketEncoder(
        source_width=12,
        source_layers=3,
        source_positions=5,
        protocol_slots=4,
        bridge_width=8,
        attention_heads=2,
        feedforward_width=16,
        decoder_blocks=1,
        dropout=0.0,
    )
    decoder = ReceiverPacketDecoder(
        target_width=10,
        target_layers=2,
        target_positions=6,
        bridge_width=8,
        attention_heads=2,
        feedforward_width=16,
        decoder_blocks=1,
        dropout=0.0,
    )
    return LIPPacketBridge(encoder, decoder)


def test_packet_bridge_exposes_fixed_protocol_code_and_target_packet():
    bridge = _small_bridge()
    source = torch.randn(2, 3, 5, 12)

    code = bridge.encode(source)
    packet = bridge.decode(code)

    assert code.shape == (2, 4, 8)
    assert packet.shape == (2, 2, 6, 10)
    assert torch.allclose(packet, bridge(source))


def test_packet_bridge_backpropagates_across_encoder_and_decoder():
    bridge = _small_bridge()
    source = torch.randn(2, 3, 5, 12)
    loss = bridge(source).square().mean()

    loss.backward()

    assert bridge.encoder.source_projection.weight.grad is not None
    assert bridge.encoder.protocol_queries.grad is not None
    assert bridge.decoder.residual_head.weight.grad is not None
    assert all(torch.isfinite(parameter.grad).all() for parameter in bridge.parameters())


def test_structured_linear_baseline_is_linear_in_source_packet_at_initialization():
    bridge = StructuredLinearPacketBridge(
        source_width=7,
        source_layers=2,
        source_positions=3,
        target_width=5,
        target_layers=2,
        target_positions=4,
    )
    first = torch.randn(2, 2, 3, 7)
    second = torch.randn(2, 2, 3, 7)

    combined = bridge(0.25 * first - 0.75 * second)
    separated = 0.25 * bridge(first) - 0.75 * bridge(second)

    assert combined.shape == (2, 2, 4, 5)
    assert torch.allclose(combined, separated, atol=1e-5, rtol=1e-5)


def test_reconstruct_target_packet_undoes_site_normalization():
    residual = torch.ones(2, 2, 3, 4)
    scaffold = torch.full((2, 3, 4), 5.0)
    site_scale = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    reconstructed = reconstruct_target_packet(residual, scaffold, site_scale)

    assert reconstructed.shape == residual.shape
    assert torch.equal(reconstructed[0, 0, :, 0], torch.tensor([6.0, 7.0, 8.0]))
    assert torch.equal(reconstructed[1, 1, :, 3], torch.tensor([9.0, 10.0, 11.0]))


def test_packet_bridge_rejects_wrong_source_layout():
    bridge = _small_bridge()

    try:
        bridge(torch.randn(2, 3, 4, 12))
    except ValueError as exc:
        assert "source_packet trailing shape" in str(exc)
    else:
        raise AssertionError("wrong source layout should fail")
