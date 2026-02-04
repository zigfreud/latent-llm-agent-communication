import os
import re
import ast
import sys
import csv
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from datasets import load_dataset

from src.pipelines.infer import (
    build_vec_injected,
    extract_thoughts,
    load_source,
    load_target,
    run_ab_no_text,
    _cleanup_cuda,
)
from src.core.models import LIPAdapter


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_prompt(example: dict) -> str:
    for key in ("prompt", "instruction", "text", "question", "input"):
        if key in example and example[key]:
            return str(example[key])
    # fallback: stringify whole record
    return str(example)


def extract_code(text: str) -> str:
    code_match = re.search(r"```python\\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()
    generic = re.search(r"```\\s*(.*?)```", text, re.DOTALL)
    if generic:
        return generic.group(1).strip()
    return text.strip()


def load_eval_samples(n_samples: int = 50):
    try:
        ds = load_dataset("flytech/python-codes-25k", split=f"test[:{n_samples}]")
    except Exception:
        ds = load_dataset("flytech/python-codes-25k", split=f"train[-{n_samples}:]")
    prompts = [pick_prompt(ex) for ex in ds]
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Evaluate syntax validity of LIP-injected generations.")
    parser.add_argument("--config", type=str, default="configs/infer.yaml", help="Path to infer config YAML.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    prompts = load_eval_samples(n_samples=50)

    # Load source only, extract thoughts, then free to avoid OOM on T4.
    src_model, src_tok = load_source(cfg["source_model"], cfg.get("runtime", {}).get("device_src", "cuda"))

    # Extract source vectors
    print("Extracting thought vectors from source...")
    thought_vectors = extract_thoughts(prompts, src_model, src_tok, cfg.get("runtime", {}).get("device_src", "cuda"))
    del src_model, src_tok
    _cleanup_cuda()

    # Load target + adapter after freeing source.
    tgt_model, tgt_tok = load_target(
        cfg["target_model"],
        cfg.get("runtime", {}).get("device_tgt", "auto"),
        cfg.get("runtime", {}).get("load_4bit", True),
    )
    adapter = LIPAdapter().to("cuda")
    state = __import__("torch").load(cfg["adapter_ckpt"], map_location="cuda", weights_only=False)
    adapter.load_state_dict(state)
    adapter.eval()

    # Build gen_kwargs after target is available.
    gen_kwargs = dict(
        max_new_tokens=int(cfg.get("generation", {}).get("max_new_tokens", 120)),
        do_sample=bool(cfg.get("generation", {}).get("do_sample", False)),
        temperature=float(cfg.get("generation", {}).get("temperature", 0.0)),
        repetition_penalty=float(cfg.get("generation", {}).get("repetition_penalty", 1.2)),
        pad_token_id=tgt_tok.eos_token_id,
    )

    anchor_prompt = str(cfg.get("lip", {}).get("anchor_prompt", "Received instruction via LIP:"))
    baseline_suffix = "\n\nReturn only Python code in a ```python``` block."

    results = []
    baseline_passes = 0
    no_text_passes = 0

    for prompt, vec_cpu in tqdm(zip(prompts, thought_vectors), total=len(prompts), desc="Evaluating"):
        vec_injected = build_vec_injected(
            vec_cpu,
            adapter=adapter,
            tgt_model=tgt_model,
            gain=float(cfg.get("lip", {}).get("gain", 10.0)),
        )

        baseline_prompt = f"{prompt}{baseline_suffix}"
        ab = run_ab_no_text(
            prompt_text=baseline_prompt,
            anchor_prompt=anchor_prompt,
            vec_injected=vec_injected,
            tgt_model=tgt_model,
            tgt_tok=tgt_tok,
            layer_idx=int(cfg.get("lip", {}).get("layer_idx", -2)),
            inject_pos_mode=str(cfg.get("lip", {}).get("inject_pos_mode", "last")),
            gen_kwargs=gen_kwargs,
        )

        baseline_text = ab["baseline_text"]
        no_text = ab["no_text"]

        baseline_code = extract_code(baseline_text)
        baseline_ok = True
        try:
            ast.parse(baseline_code)
        except SyntaxError:
            baseline_ok = False

        no_text_code = extract_code(no_text)
        no_text_ok = True
        try:
            ast.parse(no_text_code)
        except SyntaxError:
            no_text_ok = False

        baseline_passes += int(baseline_ok)
        no_text_passes += int(no_text_ok)
        results.append({
            "prompt": prompt,
            "baseline_completion": baseline_text,
            "baseline_ok": baseline_ok,
            "no_text_completion": no_text,
            "no_text_ok": no_text_ok,
        })

    baseline_rate = (baseline_passes / len(results)) * 100 if results else 0.0
    no_text_rate = (no_text_passes / len(results)) * 100 if results else 0.0
    print(f"Syntax Pass Rate (baseline): {baseline_rate:.2f}% ({baseline_passes}/{len(results)})")
    print(f"Syntax Pass Rate (no_text): {no_text_rate:.2f}% ({no_text_passes}/{len(results)})")

    output_csv = Path("results_syntax_eval.csv")
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt",
                "baseline_completion",
                "baseline_ok",
                "no_text_completion",
                "no_text_ok",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV to {output_csv.resolve()}")


if __name__ == "__main__":
    main()
    
