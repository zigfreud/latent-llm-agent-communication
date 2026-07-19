import json
from pathlib import Path

import yaml

from src.scripts.materialize_mbpp_prompt_configs import materialize_configs


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
    assert config["data"]["heldout_bundle_trace_id"] == "LIP-PROTO-001-DATA-EVAL"
    assert config["data"]["heldout_bundle_manifest"].endswith(
        "mbpp_eval_bundle_32/manifest.json"
    )


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
    assert sampling["tasks_jsonl"] == config["data"]["tasks_jsonl"]


def test_materializer_writes_exact_heldout_task_file(tmp_path):
    source = yaml.safe_load(
        Path("config/LIP-PROTO-001_mbpp_sampling.yaml").read_text(encoding="utf-8")
    )
    source.update(
        {
            "output_dir": str(tmp_path / "generated"),
            "tasks_jsonl": str(tmp_path / "tasks.jsonl"),
            "train_bundle_dir": str(tmp_path / "train_bundle"),
            "eval_bundle_dir": str(tmp_path / "eval_bundle"),
            "train_output_zip": str(tmp_path / "train.zip"),
            "eval_output_zip": str(tmp_path / "eval.zip"),
        }
    )
    config_path = tmp_path / "sampling.yaml"
    config_path.write_text(yaml.safe_dump(source), encoding="utf-8")

    result = materialize_configs(config_path, mock_data=True)
    rows = [
        json.loads(line)
        for line in Path(result["tasks_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    eval_config = yaml.safe_load(Path(result["eval_config"]).read_text(encoding="utf-8"))
    assert len(rows) == source["eval_count"]
    assert [row["task_id"] for row in rows] == eval_config["data"]["sampled_ids"]
    assert [row["prompt"] for row in rows] == eval_config["data"]["prompts"]
    assert all(row["test_list"] == [] for row in rows)


def test_protocol_requirements_are_portable_between_cpu_and_colab():
    general = Path("requirements.txt").read_text(encoding="utf-8")
    protocol = Path("requirements-protocol.txt").read_text(encoding="utf-8")
    cpu_protocol = Path("requirements-protocol-cpu.txt").read_text(encoding="utf-8")
    assert 'torch-directml; platform_system == "Windows"' in general
    assert "torch-directml" not in protocol
    assert "torch>=2.0.0" in protocol
    assert "bitsandbytes" in protocol
    assert "kernels" not in protocol
    assert "-r requirements-protocol.txt" in cpu_protocol
    assert "kernels>=0.11.1" in cpu_protocol
