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
    format_prompt_like_dataset,
    load_source,
    load_target,
    run_ab,
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


def extract_code_block(text: str) -> str:
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

    # Load models
    src_model, src_tok = load_source(cfg["source_model"], cfg.get("runtime", {}).get("device_src", "cuda"))
    tgt_model, tgt_tok = load_target(
        cfg["target_model"],
        cfg.get("runtime", {}).get("device_tgt", "auto"),
        cfg.get("runtime", {}).get("load_4bit", True),
    )
    adapter = LIPAdapter().to("cuda")
    state = __import__("torch").load(cfg["adapter_ckpt"], map_location="cuda", weights_only=False)
    adapter.load_state_dict(state)
    adapter.eval()

    gen_kwargs = dict(
        max_new_tokens=int(cfg.get("generation", {}).get("max_new_tokens", 120)),
        do_sample=bool(cfg.get("generation", {}).get("do_sample", False)),
        temperature=float(cfg.get("generation", {}).get("temperature", 0.0)),
        repetition_penalty=float(cfg.get("generation", {}).get("repetition_penalty", 1.2)),
        pad_token_id=tgt_tok.eos_token_id,
    )

    # Extract source vectors
    print("Extracting thought vectors from source...")
    thought_vectors = extract_thoughts(prompts, src_model, src_tok, cfg.get("runtime", {}).get("device_src", "cuda"))

    results = []
    success = 0

    for prompt, vec_cpu in tqdm(zip(prompts, thought_vectors), total=len(prompts), desc="Evaluating"):
        vec_injected = build_vec_injected(
            vec_cpu,
            adapter=adapter,
            tgt_model=tgt_model,
            gain=float(cfg.get("lip", {}).get("gain", 10.0)),
        )

        ab = run_ab(
            prompt_text=prompt,
            vec_injected=vec_injected,
            tgt_model=tgt_model,
            tgt_tok=tgt_tok,
            layer_idx=int(cfg.get("lip", {}).get("layer_idx", -2)),
            inject_pos_mode=str(cfg.get("lip", {}).get("inject_pos_mode", "last")),
            gen_kwargs=gen_kwargs,
        )

        generated = ab[True]
        code = extract_code_block(generated)
        is_valid = True
        try:
            ast.parse(code)
        except SyntaxError:
            is_valid = False

        success += int(is_valid)
        results.append({
            "prompt": prompt,
            "output": generated,
            "valid": is_valid,
        })

    pass_rate = (success / len(results)) * 100 if results else 0.0
    print(f"Syntax Pass Rate: {pass_rate:.2f}% ({success}/{len(results)})")

    output_csv = Path("results_syntax_eval.csv")
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "output", "valid"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV to {output_csv.resolve()}")


if __name__ == "__main__":
    main()
    
