"""Benchmark suite for LLM-to-LLM communication methods."""

from __future__ import annotations

import os
import sys
import gc
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
sys.path.insert(0, _PROJECT_ROOT)

import abc
import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.core.models import LIPAdapter
from src.integrations.hooks import make_lip_hook
from src.lip_protocol import LIPPacket

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@dataclass(frozen=True)
class BenchmarkResult:
    scenario: str
    prompt_id: int
    prompt: str
    source_processing_time: int
    transport_overhead: int
    payload_size_bytes: int
    target_ingestion_time: int
    total_latency: int

    @classmethod
    def from_metrics(cls, *, scenario: str, prompt_id: int, prompt: str, metrics: Dict[str, Any]) -> "BenchmarkResult":
        return cls(
            scenario=scenario,
            prompt_id=prompt_id,
            prompt=prompt,
            source_processing_time=int(metrics["source_processing_time"]),
            transport_overhead=int(metrics["transport_overhead"]),
            payload_size_bytes=int(metrics["payload_size_bytes"]),
            target_ingestion_time=int(metrics["target_ingestion_time"]),
            total_latency=int(metrics["total_latency"]),
        )

class BenchmarkScenario(abc.ABC):
    @abc.abstractmethod
    def prepare(self, source_model: Any, target_model: Any) -> None: raise NotImplementedError
    @abc.abstractmethod
    def execute(self, prompt: str) -> Dict[str, Any]: raise NotImplementedError

class JsonBaselineScenario(BenchmarkScenario):
    def __init__(self, source_tokenizer, target_tokenizer, *, device=None, source_max_new_tokens=64, target_max_new_tokens=64):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.device = device
        self.source_max_new_tokens = source_max_new_tokens
        self.target_max_new_tokens = target_max_new_tokens
        self.source_model = None
        self.target_model = None

    def prepare(self, source_model, target_model):
        self.source_model = source_model
        self.target_model = target_model
        if hasattr(self.source_model, "eval"): self.source_model.eval()
        if hasattr(self.target_model, "eval"): self.target_model.eval()

    def execute(self, prompt: str) -> Dict[str, Any]:
        cleanup()
        total_start = time.perf_counter_ns()

        # SOURCE: Read the full prompt.
        source_start = time.perf_counter_ns()
        source_inputs = self.source_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16000)
        if self.device: source_inputs = {k: v.to(self.device) for k, v in source_inputs.items()}
        
        with torch.no_grad():
            source_output_ids = self.source_model.generate(
                **source_inputs, max_new_tokens=self.source_max_new_tokens,
                pad_token_id=self.source_tokenizer.eos_token_id
            )
        source_end = time.perf_counter_ns()

        # TRANSPORT
        input_len = int(source_inputs["input_ids"].shape[-1])
        generated_ids = source_output_ids[0][input_len:]
        generated_text = self.source_tokenizer.decode(generated_ids, skip_special_tokens=True)
        source_processing_time = source_end - source_start

        transport_start = time.perf_counter_ns()
        payload = {"role": "user", "content": generated_text}
        payload_json = json.dumps(payload, ensure_ascii=False)
        transport_end = time.perf_counter_ns()
        payload_size_bytes = len(payload_json.encode("utf-8"))
        transport_overhead = transport_end - transport_start

        # TARGET: Read only the compact JSON payload.
        ingestion_start = time.perf_counter_ns()
        decoded = json.loads(payload_json)
        target_text = decoded["content"]
        target_inputs = self.target_tokenizer(target_text, return_tensors="pt", truncation=True, max_length=4096)
        if self.device: target_inputs = {k: v.to(self.device) for k, v in target_inputs.items()}
        ingestion_end = time.perf_counter_ns()
        target_ingestion_time = ingestion_end - ingestion_start

        with torch.no_grad():
            _ = self.target_model.generate(
                **target_inputs, max_new_tokens=self.target_max_new_tokens,
                pad_token_id=self.target_tokenizer.eos_token_id
            )
        total_end = time.perf_counter_ns()

        return {
            "source_processing_time": source_processing_time,
            "transport_overhead": transport_overhead,
            "payload_size_bytes": payload_size_bytes,
            "target_ingestion_time": target_ingestion_time,
            "total_latency": total_end - total_start,
        }

class LipProtocolScenario(BenchmarkScenario):
    def __init__(self, source_tokenizer, target_tokenizer, adapter, *, gain=10.0, layer_idx=-2, inject_pos_mode="last", target_max_new_tokens=64):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.adapter = adapter
        self.gain = gain
        self.layer_idx = layer_idx
        self.inject_pos_mode = inject_pos_mode
        self.target_max_new_tokens = target_max_new_tokens
        self.source_model = None
        self.target_model = None
        self.ref_energy = 1.0

    def prepare(self, source_model, target_model):
        self.source_model = source_model
        self.target_model = target_model
        if hasattr(self.source_model, "eval"): self.source_model.eval()
        if hasattr(self.target_model, "eval"): self.target_model.eval()
        if hasattr(self.adapter, "eval"): self.adapter.eval()
        
        print("Calculating Reference Energy (safely)...")
        with torch.no_grad():
            embeddings = self.target_model.get_input_embeddings().weight
            self.ref_energy = embeddings.float().norm(p=2, dim=-1).mean().item()

    def _build_vec_injected(self, vec):
        current_norm = vec.norm(p=2, dim=-1)
        scale = (self.ref_energy / (current_norm + 1e-6)) * float(self.gain)
        return (vec * scale).to(self.target_model.device).to(self.target_model.dtype)

    def execute(self, prompt: str) -> Dict[str, Any]:
        cleanup()
        total_start = time.perf_counter_ns()

        # SOURCE: Read the full prompt (RAG simulation).
        source_start = time.perf_counter_ns()
        source_inputs = self.source_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16000)
        source_inputs = {k: v.to(self.source_model.device) for k, v in source_inputs.items()}
        with torch.no_grad():
            out = self.source_model(**source_inputs, output_hidden_states=True)
        vec = out.hidden_states[-1][:, -1, :].detach()
        # Free source memory immediately after use.
        del source_inputs
        del out
        cleanup()
        
        source_end = time.perf_counter_ns()
        source_processing_time = source_end - source_start

        # TRANSPORT
        transport_start = time.perf_counter_ns()
        adapter_device = next(self.adapter.parameters()).device
        vec_translated = self.adapter(vec.to(adapter_device))
        packet = LIPPacket(vec_translated, source_model="DeepSeek", intent="instruction")
        payload_json = packet.to_json()
        transport_end = time.perf_counter_ns()
        transport_overhead = transport_end - transport_start
        payload_size_bytes = len(payload_json.encode("utf-8"))

        # TARGET: Receive only the vector + anchor.
        ingestion_start = time.perf_counter_ns()
        vec_unpacked = packet.unpack_vector(device=self.target_model.device)
        vec_injected = self._build_vec_injected(vec_unpacked)

        # Use a static anchor instead of the full prompt.
        anchor_text = "Received instruction via LIP:"
        target_inputs = self.target_tokenizer(anchor_text, return_tensors="pt")
        target_inputs = {k: v.to(self.target_model.device) for k, v in target_inputs.items()}
        input_ids = target_inputs["input_ids"]
        
        # Inject at the last anchor position.
        inject_pos = input_ids.shape[1] - 1
        
        hook_fn = make_lip_hook(vec_injected=vec_injected, inject_pos=inject_pos, enable=True)
        handle = self.target_model.model.layers[self.layer_idx].register_forward_hook(hook_fn)
        ingestion_end = time.perf_counter_ns()
        target_ingestion_time = ingestion_end - ingestion_start

        with torch.no_grad():
            _ = self.target_model.generate(
                **target_inputs, max_new_tokens=self.target_max_new_tokens,
                pad_token_id=self.target_tokenizer.eos_token_id
            )
        handle.remove()
        total_end = time.perf_counter_ns()

        return {
            "source_processing_time": source_processing_time,
            "transport_overhead": transport_overhead,
            "payload_size_bytes": payload_size_bytes,
            "target_ingestion_time": target_ingestion_time,
            "total_latency": total_end - total_start,
        }

def _expand_heavy_context(prompt: str, multiplier: int = 15) -> str:
    lines = prompt.splitlines()
    start_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("context:"): start_idx = idx + 1
        if line.strip().lower().startswith("task:"): end_idx = idx; break
    if start_idx is None or end_idx is None or start_idx >= end_idx: return prompt * multiplier
    context_block = "\\n".join(lines[start_idx:end_idx]).strip()
    expanded_context = "\\n".join([context_block] * multiplier)
    return "\\n".join(lines[:start_idx]) + "\\n" + expanded_context + "\\n" + "\\n".join(lines[end_idx:])

def _collect_results(scenarios, dataset):
    results = []
    for scenario in scenarios:
        name = scenario.__class__.__name__
        for pid, prompt in enumerate(dataset):
            cleanup()
            try:
                metrics = scenario.execute(prompt)
                results.append(BenchmarkResult.from_metrics(scenario=name, prompt_id=pid, prompt=prompt, metrics=metrics))
            except Exception as e:
                print(f"⚠️ Erro prompt {pid} ({name}): {e}")
    return results

def _write_results_csv(output_path, results):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["scenario", "prompt_id", "prompt_len", "source_processing_time", "transport_overhead", "payload_size_bytes", "target_ingestion_time", "total_latency"])
        for i in results: w.writerow([i.scenario, i.prompt_id, len(i.prompt), i.source_processing_time, i.transport_overhead, i.payload_size_bytes, i.target_ingestion_time, i.total_latency])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.config) as f: cfg = yaml.safe_load(f)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, llm_int8_enable_fp32_cpu_offload=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    print("Loading Source...")
    s_tok = AutoTokenizer.from_pretrained(cfg["models"]["source"]["path"], trust_remote_code=True)
    s_mod = AutoModelForCausalLM.from_pretrained(cfg["models"]["source"]["path"], quantization_config=bnb, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
    
    print("Loading Target...")
    t_tok = AutoTokenizer.from_pretrained(cfg["models"]["target"]["path"], trust_remote_code=True)
    t_mod = AutoModelForCausalLM.from_pretrained(cfg["models"]["target"]["path"], quantization_config=bnb, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)

    print("Loading Adapter...")
    acfg = cfg["models"]["adapter"]
    adapter = LIPAdapter(input_dim=int(acfg.get("input_dim", 2048)), hidden_dim=int(acfg.get("bottleneck_dim", 1024)), output_dim=int(acfg.get("output_dim", 4096))).to(t_mod.device)
    adapter.load_state_dict(torch.load(acfg["path"], map_location=t_mod.device, weights_only=False))
    adapter.eval()

    prompts = []
    for item in cfg.get("test_cases", []):
        p = item["prompt"]
        if item.get("category") == "heavy_context_rag_simulation": p = _expand_heavy_context(p)
        prompts.append(p)

    json_s = JsonBaselineScenario(s_tok, t_tok, device=s_mod.device)
    lip_s = LipProtocolScenario(s_tok, t_tok, adapter)

    json_s.prepare(s_mod, t_mod)
    lip_s.prepare(s_mod, t_mod)

    iterations = int(cfg["experiment"].get("iterations", 1))
    warmup = int(cfg["experiment"].get("warmup", 0))
    
    all_res = []
    print(f"Starting Loop ({len(prompts)} prompts)...")
    for idx in range(warmup + iterations):
        print(f"   Iter {idx+1}/{warmup+iterations}")
        res = _collect_results([json_s, lip_s], prompts)
        if idx >= warmup: all_res.extend(res)

    _write_results_csv(Path(args.output), all_res)
    print(f"Done! Saved to {args.output}")

if __name__ == "__main__": main()
