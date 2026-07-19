from pathlib import Path

import yaml


def test_source_only_config_has_independent_seeds_and_required_controls():
    path = Path("config/LIP-PROTO-001_source_only_eval.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    checkpoint_seeds = [item["training_seed"] for item in config["adapter"]["checkpoints"]]
    assert len(checkpoint_seeds) >= 3
    assert len(set(checkpoint_seeds)) == len(checkpoint_seeds)
    assert len(set(config["generation"]["seeds"])) >= 3
    assert {
        "neutral_no_lip",
        "text_only_no_lip",
        "source_latent",
        "shuffled_source_latent",
        "random_norm_matched",
        "oracle_target_latent",
    } == set(config["conditions"])
    training = yaml.safe_load(
        Path("config/LIP-PROTO-001_multiseed_training.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert training["data"]["require_bundle_manifest"] is True
    assert training["data"]["expected_extraction_mode"] == "real"
    assert training["data"]["expected_trace_id"] == config["adapter"][
        "training_bundle_trace_id"
    ]


def test_target_extraction_layer_matches_injection_layer():
    path = Path("config/LIP-PROTO-001_source_only_eval.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["extraction"]["target_layer"] == config["lip"]["layer_idx"]
    assert config["extraction"]["token_position"] == "last_non_padding"
    assert config["lip"]["injection_mode"] == "replace"
    assert config["lip"]["vector_scaling"] == "none"
    assert config["lip"]["gain"] == 1.0
    assert config["runtime"]["source_load_4bit"] is True
    assert config["runtime"]["load_4bit"] is True
    assert config["runtime"]["quantization_compute_dtype"] == "float16"

    sampling = yaml.safe_load(
        Path("config/LIP-PROTO-001_mbpp_sampling.yaml").read_text(encoding="utf-8")
    )
    assert sampling["target_layer"] == config["lip"]["layer_idx"]
    assert sampling["dtype"] == config["runtime"]["quantization_compute_dtype"]
    assert sampling["max_length"] == config["extraction"]["max_length"]
    assert sampling["train_trace_id"].startswith("LIP-PROTO-001")
