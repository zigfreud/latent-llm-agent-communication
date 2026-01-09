"""Benchmark suite for LLM-to-LLM communication methods."""

from __future__ import annotations

import os
import sys
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.models import LIPAdapter
from src.integrations.hooks import make_lip_hook
from src.lip_protocol import LIPPacket


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
    def from_metrics(
        cls,
        *,
        scenario: str,
        prompt_id: int,
        prompt: str,
        metrics: Dict[str, Any],
    ) -> "BenchmarkResult":
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
    """Contract for benchmark scenarios."""

    @abc.abstractmethod
    def prepare(self, source_model: Any, target_model: Any) -> None:
        """Initialize the scenario with source/target models."""
        raise NotImplementedError

    @abc.abstractmethod
    def execute(self, prompt: str) -> Dict[str, Any]:
        """Run a single inference and return timing metrics."""
        raise NotImplementedError


class BenchmarkScenarioBase(abc.ABC):
    """Base contract for research benchmarking scenarios."""

    def __init__(self, source_model: Any, target_model: Any, tokenizer: Any, config: Dict[str, Any]) -> None:
        self.source_model = source_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.config = config

    @abc.abstractmethod
    def execute(self, prompt: str) -> Dict[str, Any]:
        """
        Execute a benchmark run and return metrics:
        - source_latency_ns
        - transport_latency_ns
        - payload_size_bytes
        - target_ingestion_ns
        - total_latency_ns
        """
        raise NotImplementedError


class JsonBaselineScenario(BenchmarkScenario):
    """Baseline JSON (Text -> JSON -> Text) communication simulation."""

    def __init__(
        self,
        source_tokenizer: Any,
        target_tokenizer: Any,
        *,
        device: Optional[torch.device] = None,
        source_max_new_tokens: int = 64,
        target_max_new_tokens: int = 64,
    ) -> None:
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.device = device
        self.source_max_new_tokens = source_max_new_tokens
        self.target_max_new_tokens = target_max_new_tokens
        self.source_model: Optional[Any] = None
        self.target_model: Optional[Any] = None

    def prepare(self, source_model: Any, target_model: Any) -> None:
        self.source_model = source_model
        self.target_model = target_model
        if hasattr(self.source_model, "eval"):
            self.source_model.eval()
        if hasattr(self.target_model, "eval"):
            self.target_model.eval()

    def execute(self, prompt: str) -> Dict[str, Any]:
        if self.source_model is None or self.target_model is None:
            raise RuntimeError("Scenario not prepared. Call prepare() first.")

        total_start = time.perf_counter_ns()

        source_start = time.perf_counter_ns()
        source_inputs = self.source_tokenizer(prompt, return_tensors="pt")
        if self.device is not None:
            source_inputs = {k: v.to(self.device) for k, v in source_inputs.items()}
        with torch.no_grad():
            source_output_ids = self.source_model.generate(
                **source_inputs,
                max_new_tokens=self.source_max_new_tokens,
            )
        source_end = time.perf_counter_ns()

        input_len = int(source_inputs["input_ids"].shape[-1])
        generated_ids = source_output_ids[0][input_len:]
        generated_text = self.source_tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        source_processing_time = source_end - source_start

        transport_start = time.perf_counter_ns()
        payload = {"role": "user", "content": generated_text}
        payload_json = json.dumps(payload, ensure_ascii=False)
        transport_end = time.perf_counter_ns()
        payload_size_bytes = len(payload_json.encode("utf-8"))
        transport_overhead = transport_end - transport_start

        ingestion_start = time.perf_counter_ns()
        decoded = json.loads(payload_json)
        target_text = decoded["content"]
        target_inputs = self.target_tokenizer(target_text, return_tensors="pt")
        if self.device is not None:
            target_inputs = {k: v.to(self.device) for k, v in target_inputs.items()}
        ingestion_end = time.perf_counter_ns()
        target_ingestion_time = ingestion_end - ingestion_start

        with torch.no_grad():
            _ = self.target_model.generate(
                **target_inputs,
                max_new_tokens=self.target_max_new_tokens,
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
    """LIP Protocol (Latent -> Adapter -> Injection) communication simulation."""

    def __init__(
        self,
        source_tokenizer: Any,
        target_tokenizer: Any,
        adapter: LIPAdapter,
        *,
        gain: float = 10.0,
        layer_idx: int = -2,
        inject_pos_mode: str = "last",
        target_max_new_tokens: int = 64,
    ) -> None:
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.adapter = adapter
        self.gain = gain
        self.layer_idx = layer_idx
        self.inject_pos_mode = inject_pos_mode
        self.target_max_new_tokens = target_max_new_tokens
        self.source_model: Optional[Any] = None
        self.target_model: Optional[Any] = None

    def prepare(self, source_model: Any, target_model: Any) -> None:
        self.source_model = source_model
        self.target_model = target_model
        if hasattr(self.source_model, "eval"):
            self.source_model.eval()
        if hasattr(self.target_model, "eval"):
            self.target_model.eval()
        if hasattr(self.adapter, "eval"):
            self.adapter.eval()

    def _build_vec_injected(self, vec: torch.Tensor) -> torch.Tensor:
        # Calibrate energy against target embeddings (project physics alignment).
        ref_energy = (
            self.target_model.get_input_embeddings()
            .weight.norm(p=2, dim=-1)
            .mean()
            .item()
        )
        current_norm = vec.norm(p=2, dim=-1)
        scale = (ref_energy / (current_norm + 1e-6)) * float(self.gain)
        return (vec * scale).to(self.target_model.device).to(self.target_model.dtype)

    def execute(self, prompt: str) -> Dict[str, Any]:
        if self.source_model is None or self.target_model is None:
            raise RuntimeError("Scenario not prepared. Call prepare() first.")

        total_start = time.perf_counter_ns()

        # LIP gains by avoiding source decoding; only a forward pass is timed here.
        source_start = time.perf_counter_ns()
        source_inputs = self.source_tokenizer(prompt, return_tensors="pt")
        source_inputs = {k: v.to(self.source_model.device) for k, v in source_inputs.items()}
        with torch.no_grad():
            out = self.source_model(**source_inputs, output_hidden_states=True)
        vec = out.hidden_states[-1][:, -1, :].detach()
        source_end = time.perf_counter_ns()
        source_processing_time = source_end - source_start

        transport_start = time.perf_counter_ns()
        adapter_device = next(self.adapter.parameters()).device
        vec_translated = self.adapter(vec.to(adapter_device))
        packet = LIPPacket(vec_translated, source_model="DeepSeek", intent="instruction")
        payload_json = packet.to_json()
        transport_end = time.perf_counter_ns()
        transport_overhead = transport_end - transport_start
        payload_size_bytes = len(payload_json.encode("utf-8"))

        ingestion_start = time.perf_counter_ns()
        vec_unpacked = packet.unpack_vector(device=self.target_model.device)
        vec_injected = self._build_vec_injected(vec_unpacked)

        target_inputs = self.target_tokenizer(prompt, return_tensors="pt")
        target_inputs = {k: v.to(self.target_model.device) for k, v in target_inputs.items()}
        input_ids = target_inputs["input_ids"]
        attn_mask = target_inputs.get("attention_mask", None)

        if self.inject_pos_mode == "last":
            inject_pos = input_ids.shape[1] - 1
        elif self.inject_pos_mode == "last_minus_1":
            inject_pos = input_ids.shape[1] - 2
        else:
            raise ValueError(f"Unknown inject_pos_mode: {self.inject_pos_mode}")

        hook_fn = make_lip_hook(vec_injected=vec_injected, inject_pos=inject_pos, enable=True)
        handle = self.target_model.model.layers[self.layer_idx].register_forward_hook(hook_fn)
        ingestion_end = time.perf_counter_ns()
        target_ingestion_time = ingestion_end - ingestion_start

        with torch.no_grad():
            _ = self.target_model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.target_max_new_tokens,
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


def run_benchmark(
    scenarios: Iterable[BenchmarkScenario],
    dataset: Iterable[str],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Execute all scenarios over the dataset and persist results to CSV.

    Each scenario.execute must return a dictionary with required keys:
    - source_processing_time
    - transport_overhead
    - payload_size_bytes
    - target_ingestion_time
    - total_latency
    """

    output_path = output_path or Path("benchmark_results.csv")
    results = _collect_results(scenarios, dataset)
    _write_results_csv(output_path, results)
    return output_path


def _collect_results(
    scenarios: Iterable[BenchmarkScenario],
    dataset: Iterable[str],
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    for scenario in scenarios:
        scenario_name = scenario.__class__.__name__
        for prompt_id, prompt in enumerate(dataset):
            metrics = scenario.execute(prompt)
            results.append(
                BenchmarkResult.from_metrics(
                    scenario=scenario_name,
                    prompt_id=prompt_id,
                    prompt=prompt,
                    metrics=metrics,
                )
            )
    return results


def _write_results_csv(output_path: Path, results: List[BenchmarkResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "prompt_id",
                "prompt",
                "source_processing_time",
                "transport_overhead",
                "payload_size_bytes",
                "target_ingestion_time",
                "total_latency",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.scenario,
                    item.prompt_id,
                    item.prompt,
                    item.source_processing_time,
                    item.transport_overhead,
                    item.payload_size_bytes,
                    item.target_ingestion_time,
                    item.total_latency,
                ]
            )


def _expand_heavy_context(prompt: str, multiplier: int = 200) -> str:
    lines = prompt.splitlines()
    start_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("context:"):
            start_idx = idx + 1
        if line.strip().lower().startswith("task:"):
            end_idx = idx
            break

    if start_idx is None or end_idx is None or start_idx >= end_idx:
        return prompt * multiplier

    context_block = "\n".join(lines[start_idx:end_idx]).strip()
    expanded_context = "\n".join([context_block] * multiplier)
    return "\n".join(lines[:start_idx]) + "\n" + expanded_context + "\n" + "\n".join(lines[end_idx:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LIP vs JSON benchmark suite.")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML config.")
    parser.add_argument("--output", required=True, help="Path to output CSV.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    source_cfg = cfg["models"]["source"]
    target_cfg = cfg["models"]["target"]
    adapter_cfg = cfg["models"]["adapter"]

    source_device = torch.device(source_cfg.get("device", "cpu"))
    target_device = torch.device(target_cfg.get("device", "cpu"))
    torch_dtype = torch.float16 if source_device.type == "cuda" else torch.float32

    source_tokenizer = AutoTokenizer.from_pretrained(source_cfg["path"], trust_remote_code=True)
    source_model = AutoModelForCausalLM.from_pretrained(
        source_cfg["path"],
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=True,
    ).to(source_device)
    source_model.eval()

    target_tokenizer = AutoTokenizer.from_pretrained(target_cfg["path"], trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_cfg["path"],
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=True,
    ).to(target_device)
    target_model.eval()

    adapter = LIPAdapter(
        input_dim=int(adapter_cfg.get("input_dim", 2048)),
        hidden_dim=int(adapter_cfg.get("bottleneck_dim", 512)),
        output_dim=int(adapter_cfg.get("output_dim", 4096)),
    ).to(target_device)
    state = torch.load(adapter_cfg["path"], map_location=target_device, weights_only=False)
    adapter.load_state_dict(state)
    adapter.eval()

    prompts: List[str] = []
    for item in cfg.get("test_cases", []):
        prompt = item["prompt"]
        if item.get("category") == "heavy_context_rag_simulation":
            prompt = _expand_heavy_context(prompt, multiplier=200)
        prompts.append(prompt)

    json_scenario = JsonBaselineScenario(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        device=source_device,
    )
    lip_scenario = LipProtocolScenario(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        adapter=adapter,
    )

    json_scenario.prepare(source_model, target_model)
    lip_scenario.prepare(source_model, target_model)

    iterations = int(cfg["experiment"].get("iterations", 1))
    warmup = int(cfg["experiment"].get("warmup", 0))

    all_results: List[BenchmarkResult] = []
    for idx in range(warmup + iterations):
        results = _collect_results([json_scenario, lip_scenario], prompts)
        if idx >= warmup:
            all_results.extend(results)

    _write_results_csv(Path(args.output), all_results)


if __name__ == "__main__":
    main()
