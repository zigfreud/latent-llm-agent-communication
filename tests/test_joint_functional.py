from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from src.core.receiver_closed_loop import evolve_receiver_with_closed_loop_corrector
from src.core.receiver_closed_loop_generation import closed_loop_prefill
from src.evaluation.joint_functional import donor_map, validate_grid
from src.pipelines.oracle_experiment import task_sha256


def tiny_receiver():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(91)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
        bos_token_id=1, eos_token_id=31, pad_token_id=0)).eval()
    model.requires_grad_(False)
    return model


def correction(code, live, *, layer_index):
    return 0.2 * live + code.mean() * (layer_index + 1)


def setup():
    inputs = {"input_ids": torch.tensor([[1, 4, 5, 8]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    args = dict(positions=torch.tensor([2, 3]), protocol_code=torch.ones(1, 2, 8) * 0.04,
                corrector=correction, scaffold=torch.zeros(2, 2, 8),
                site_scale=torch.ones(2, 2), layer_indices=[0, 1])
    return inputs, args


def test_cached_generation_preserves_closed_loop_prefill_and_only_corrects_once():
    model = tiny_receiver()
    inputs, args = setup()
    with torch.inference_mode():
        reference = evolve_receiver_with_closed_loop_corrector(model, inputs, **args)
        with closed_loop_prefill(model, prompt_length=4, **args) as audit:
            result = model.generate(**inputs, max_new_tokens=4, min_new_tokens=4, do_sample=False, use_cache=True)
        assert result.shape[1] == 8
        for i in range(2):
            torch.testing.assert_close(audit["incoming"][i], reference["incoming_before_correction"][:, i])
            torch.testing.assert_close(audit["corrected"][i], reference["residual_input"][:, i])
            assert audit["decode_calls"][i] == 3
    assert all(not layer._forward_pre_hooks for layer in model.model.layers)


def test_zero_delta_is_native_generation_and_hooks_clean_up_on_failure():
    model = tiny_receiver()
    inputs, args = setup()
    args["corrector"] = lambda code, live, **kw: torch.zeros_like(live)
    with torch.inference_mode():
        expected = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        with closed_loop_prefill(model, prompt_length=4, **args):
            observed = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        assert torch.equal(expected, observed)
        with pytest.raises(ValueError, match="cached"):
            with closed_loop_prefill(model, prompt_length=4, **args):
                model(**inputs)
                model(**inputs)
    assert all(not layer._forward_pre_hooks for layer in model.model.layers)


def fixture_grid():
    policy = json.loads((Path(__file__).resolve().parents[1] / "config/H0-017_joint_functional_v1.json").read_text())
    tasks = {tid: {"task_id": tid, "entry_point": "f", "test_list": ["assert f()==1"]} for tid in policy["task_ids"]}
    donors = donor_map(policy["task_ids"], policy["seed"])
    rows = [{"task_id": tid, "condition": c, "task_spec": tasks[tid],
             "donor_task_id": None if c in ("text", "no_message") else donors[tid] if c.endswith("shuffled") else tid}
            for c in policy["conditions"] for tid in policy["task_ids"]]
    meta = {"policy": policy, "task_hashes": {tid: task_sha256(t) for tid, t in tasks.items()}}
    return policy, rows, meta


def test_derangement_preserves_every_message_without_fixed_points():
    policy, rows, meta = fixture_grid()
    donors = donor_map(policy["task_ids"], policy["seed"])
    assert set(donors) == set(donors.values())
    assert all(t != d for t, d in donors.items())
    validate_grid(policy, rows, meta)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "tests", "donor", "policy"])
def test_evaluator_rejects_invalid_evidence(corruption):
    policy, rows, meta = deepcopy(fixture_grid())
    if corruption == "missing": rows.pop()
    if corruption == "duplicate": rows.append(rows[0])
    if corruption == "tests": rows[0]["task_spec"]["test_list"] = ["assert True"]
    if corruption == "donor": rows[0]["donor_task_id"] = "wrong"
    if corruption == "policy": meta["policy"] = {**policy, "seed": 7}
    with pytest.raises(ValueError):
        validate_grid(policy, rows, meta)
