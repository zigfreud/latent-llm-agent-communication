from src.pipelines.packet_matrix import build_replica_config


def _contract():
    common_loss = {
        "lambda_huber": 1.0,
        "lambda_cosine": 0.25,
        "lambda_symmetric_nce": 0.0,
        "lambda_margin": 0.0,
        "lambda_norm": 0.0,
    }
    return {
        "experiment_id": "LIP-PROTO-014",
        "protocol_version": "packet-v1",
        "packets": {"target": {"boundary_positions": 6}},
        "bridge": {
            "protocol_slots": 32,
            "bridge_width": 64,
            "attention_heads": 8,
            "feedforward_width": 128,
            "encoder_blocks": 2,
            "decoder_blocks": 2,
            "dropout": 0.1,
        },
        "training": {
            "seeds": [11, 13, 17],
            "default_output_dir": "runs/test",
            "batch_size": 4,
            "max_updates": 100,
            "validation_interval": 20,
            "learning_rate": 0.001,
        },
        "objectives": {
            "variants": {
                "nonlinear": {
                    "role": "primary",
                    "model_kind": "query_conditioned",
                    "loss": common_loss,
                },
                "linear": {
                    "role": "architecture_baseline",
                    "model_kind": "structured_linear",
                    "loss": common_loss,
                },
            }
        },
        "development_gate": {"alpha": 0.05, "statistics_seed": 31},
    }


def test_replica_config_resolves_modular_bridge_and_smoke_budget():
    resolved = build_replica_config(
        _contract(),
        bundle_dir="bundle",
        output_dir="run",
        variant_name="nonlinear",
        seed=11,
        device="cpu",
        require_real=False,
        max_updates=7,
    )

    assert resolved["model"]["kind"] == "query_conditioned"
    assert resolved["model"]["protocol_slots"] == 32
    assert resolved["training"]["max_updates"] == 7
    assert resolved["training"]["validation_interval"] == 7
    assert resolved["data"]["require_real"] is False


def test_replica_config_keeps_linear_baseline_structurally_linear():
    resolved = build_replica_config(
        _contract(),
        bundle_dir="bundle",
        output_dir="run",
        variant_name="linear",
        seed=13,
    )

    assert resolved["model"] == {"kind": "structured_linear"}
    assert resolved["objective_role"] == "architecture_baseline"
