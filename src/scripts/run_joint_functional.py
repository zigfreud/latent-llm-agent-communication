"""Frozen-checkpoint functional diagnostic on the already-open selection set."""
import argparse
from contextlib import contextmanager, nullcontext
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import torch

from src.core.packet_bundle import load_packet_records, sha256_file, sha256_json, validate_packet_bundle
from src.core.prompt_protocol import format_prompt, protocol_pair_metadata, tokenizer_add_special_tokens
from src.core.receiver_closed_loop import evolve_receiver_with_closed_loop_corrector
from src.core.receiver_closed_loop_generation import closed_loop_prefill
from src.evaluation.joint_functional import donor_map, validate_grid
from src.pipelines.closed_loop_trajectory import build_closed_loop_bridge
from src.pipelines.infer import load_target, model_input_device
from src.pipelines.oracle_experiment import load_yaml, task_sha256, write_jsonl
from src.pipelines.oracle_memory import _register_replay_hooks
from src.pipelines.packet_confirmation import _neutral_inputs, _suffix_positions, _target_inputs_from_record
from src.pipelines.packet_trajectory import _atomic_json
from src.scripts.materialize_packet_bridge_tasks import _load_rows, _normalize_rows


@contextmanager
def oracle_prefill(model, positions, packet):
    handles = _register_replay_hooks(model, positions, {i: packet[i] for i in range(8)}, replay_mode="replace")
    try:
        yield None
    finally:
        for handle in handles:
            handle.remove()


def run(args):
    policy = load_yaml(args.config)
    assert policy["experiment_id"] == "H0-017-joint-functional-v1"
    assert policy["split"] == "development_selection"
    assert not policy["training_allowed"] and not policy["confirmation_allowed"]
    assert policy["generation"]["use_cache"] and policy["generation"]["num_beams"] == 1
    assert not policy["generation"]["do_sample"]
    base = load_yaml(Path("config/LIP-H0-017_closed_loop_trajectory_corrector.yaml"))
    parent = load_yaml(Path(base["parent"]["config"]))
    if sha256_file(args.bundle / "manifest.json") != policy["bundle_manifest_sha256"]:
        raise ValueError("unexpected packet bundle")
    validate_packet_bundle(args.bundle, require_real=True)
    records = load_packet_records(args.bundle, split=policy["split"])
    assert [str(r["task_id"]) for r in records] == policy["task_ids"]
    record_map = {str(r["task_id"]): r for r in records}
    data = parent["data"]
    raw = _load_rows(data["dataset"], data["dataset"]["development_split"], mock_data=False, minimum_count=32)
    candidates = _normalize_rows(raw, prompt_field=data["dataset"]["prompt_field"],
                                 max_prompt_chars=data["selection"]["max_prompt_chars"],
                                 split=data["dataset"]["development_split"], salt=data["selection"]["salt"])
    task_map = {t["task_id"]: t for t in candidates if t["task_id"] in record_map}
    del raw, candidates
    for task_id, record in record_map.items():
        if task_sha256(task_map[task_id]) != record["task_sha256"]:
            raise ValueError(f"task specification mismatch: {task_id}")
    checkpoints = {arm: args.checkpoints / arm / "best_checkpoint.pt" for arm in policy["checkpoints"]}
    if {arm: sha256_file(path) for arm, path in checkpoints.items()} != policy["checkpoint_sha256"]:
        raise ValueError("selected checkpoint hashes changed")
    stats_paths = {arm: args.checkpoints / arm / "target_statistics.pt" for arm in checkpoints}
    stats = torch.load(stats_paths["regional"], map_location="cpu", weights_only=True)
    other = torch.load(stats_paths["joint_only"], map_location="cpu", weights_only=True)
    assert all(torch.equal(stats[k], other[k]) for k in ("scaffold", "site_scale"))
    del other
    metadata = {
        "policy": policy,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256": {arm: sha256_file(path) for arm, path in checkpoints.items()},
        "statistics_sha256": {arm: sha256_file(path) for arm, path in stats_paths.items()},
        "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
        "task_hashes": {tid: task_sha256(task_map[tid]) for tid in policy["task_ids"]},
        "donors": donor_map(policy["task_ids"], policy["seed"]),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "receiver": base["receiver"],
        "source_extraction_cost_measured": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "generations.metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("resume provenance differs from frozen metadata")
    _atomic_json(metadata_path, metadata)
    _atomic_json(args.output / "policy.json", policy)
    write_jsonl(args.output / "tasks.jsonl", [task_map[tid] for tid in policy["task_ids"]])
    print(json.dumps({"event": "preflight_verified", "tasks": len(records), "metadata": metadata}), flush=True)
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("this diagnostic requires the existing L4 runtime")
    receiver, tokenizer = load_target(base["receiver"]["model_id"], "cuda:0", True, revision=base["receiver"]["revision"])
    receiver.requires_grad_(False)
    device = model_input_device(receiver)
    _, neutral_inputs = _neutral_inputs(parent, tokenizer, device)
    positions = _suffix_positions(neutral_inputs, base["receiver"]["packet_offsets"])
    scaffold, site_scale = (stats[k].to(device) for k in ("scaffold", "site_scale"))
    bridge = build_closed_loop_bridge(base, source_shape=tuple(records[0]["source_packet"].shape),
                                     source_checkpoint_path=args.source_checkpoint, variant_name="closed_loop_live").to(device).eval()
    bridge.requires_grad_(False)
    target_protocol = protocol_pair_metadata(parent["prompt_protocols"])["target"]
    for tid, record in record_map.items():
        formatted = format_prompt(task_map[tid]["prompt"], tokenizer, target_protocol)
        ids = tokenizer(formatted, add_special_tokens=tokenizer_add_special_tokens(target_protocol))["input_ids"]
        if ids != record["target_input_ids"]:
            raise ValueError("text baseline tokenization differs from packet source")
    kwargs = dict(policy["generation"], pad_token_id=tokenizer.eos_token_id)
    # Preserve all model-native stop IDs, including the chat end-of-turn token.
    eos = receiver.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    kwargs["eos_token_id"] = list(eos_ids)
    _atomic_json(args.output / "generation_runtime.json", {"generation_kwargs": kwargs,
                 "accelerator": torch.cuda.get_device_name(0),
                 "neutral_input_ids": neutral_inputs["input_ids"].tolist(),
                 "protocol_code_shape": [32, 512], "protocol_code_dtype": "float32",
                 "protocol_code_bytes": 32 * 512 * 4,
                 "source_packets_cached": True, "full_receiver_layers": len(receiver.model.layers)})
    row_dir = args.output / "records"
    row_dir.mkdir(exist_ok=True)
    checks = {}
    active_arm = None
    for condition in policy["conditions"]:
        arm = next((a for a in checkpoints if condition.startswith(a + "_")), None)
        if arm and arm != active_arm:
            ck = torch.load(checkpoints[arm], map_location="cpu", weights_only=True)
            assert ck["step"] == policy["checkpoints"][arm] and ck["variant"] == "closed_loop_live"
            bridge.corrector.load_state_dict(ck["corrector_state"], strict=True)
            del ck
            active_arm = arm
            # Validate the generation prefill against the training/evaluation path
            # before looking at any functional scores.
            with torch.inference_mode():
                code = bridge.encode(records[0]["source_packet"].float()[None].to(device))
                hook_args = dict(positions=positions, protocol_code=code, corrector=bridge.correction,
                                 scaffold=scaffold, site_scale=site_scale, layer_indices=list(range(8)))
                reference = evolve_receiver_with_closed_loop_corrector(receiver, neutral_inputs, **hook_args)
                with closed_loop_prefill(receiver, prompt_length=neutral_inputs["input_ids"].shape[1], **hook_args) as audit:
                    receiver.generate(**neutral_inputs, **{**kwargs, "max_new_tokens": 2})
                observed = torch.stack([audit["incoming"][i] for i in range(8)], dim=1)
                expected = reference["incoming_before_correction"]
                error = float((observed.float() - expected.float()).abs().max())
                agrees = torch.allclose(observed.float(), expected.float(), atol=0.005, rtol=0.005)
                checks[arm] = {"max_abs_difference": error, "allclose": agrees, "decode_calls": audit["decode_calls"]}
                if not agrees:
                    raise RuntimeError(f"generation prefill differs from training path: {checks[arm]}")
                del reference, observed, expected, audit, code
            _atomic_json(args.output / "prefill_parity.json", checks)
            print(json.dumps({"event": "prefill_parity", "arm": arm, **checks[arm]}), flush=True)
        for task_id in policy["task_ids"]:
            path = row_dir / f"{condition}_{task_id}.json"
            if path.exists():
                continue
            torch.manual_seed(policy["seed"])
            torch.cuda.manual_seed_all(policy["seed"])
            donor = metadata["donors"][task_id] if condition.endswith("shuffled") else task_id
            inputs = _target_inputs_from_record(record_map[task_id], device) if condition == "text" else neutral_inputs
            prompt_length = inputs["input_ids"].shape[1]
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                if arm:
                    code = bridge.encode(record_map[donor]["source_packet"].float()[None].to(device))
                    context = closed_loop_prefill(receiver, prompt_length=prompt_length, positions=positions,
                        protocol_code=code, corrector=bridge.correction, scaffold=scaffold,
                        site_scale=site_scale, layer_indices=list(range(8)))
                elif condition.startswith("oracle"):
                    context = oracle_prefill(receiver, positions, record_map[donor]["target_packet"])
                else:
                    donor = None
                    context = nullcontext()
                with context as audit:
                    generated = receiver.generate(**inputs, **kwargs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                continuation = generated[0, prompt_length:].tolist()
                output_text = tokenizer.decode(continuation, skip_special_tokens=True).replace("</s>", "").strip()
                hook_audit = None if not arm else {"corrected_layers": list(audit["incoming"]), "decode_calls": audit["decode_calls"]}
            row = {"task_id": task_id, "condition": condition, "donor_task_id": donor,
                   "task_spec": task_map[task_id], "output_text": output_text,
                   "input_ids_sha256": sha256_json(inputs["input_ids"].tolist()),
                   "input_tokens": prompt_length, "output_token_ids": continuation,
                   "output_tokens": len(continuation), "seconds_cached_source_to_response": elapsed,
                   "hit_token_limit": len(continuation) == kwargs["max_new_tokens"] and continuation[-1] not in eos_ids,
                   "hook_audit": hook_audit, "metadata_sha256": sha256_file(metadata_path)}
            _atomic_json(path, row)
            print(json.dumps({"event": "generated", "condition": condition, "task_id": task_id,
                              "tokens": len(continuation), "seconds": round(elapsed, 3)}), flush=True)
            del generated, audit
        print(json.dumps({"event": "condition_complete", "condition": condition}), flush=True)
    rows = [json.loads((row_dir / f"{c}_{t}.json").read_text()) for c in policy["conditions"] for t in policy["task_ids"]]
    assert all(r["metadata_sha256"] == sha256_file(metadata_path) for r in rows)
    validate_grid(policy, rows, metadata)
    write_jsonl(args.output / "generations.jsonl", rows)
    _atomic_json(args.output / "generation_complete.json", {"records": len(rows), "generations_sha256": sha256_file(args.output / "generations.jsonl")})
    print("JOINT_FUNCTIONAL_GENERATION_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/H0-017_joint_functional_v1.json"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
