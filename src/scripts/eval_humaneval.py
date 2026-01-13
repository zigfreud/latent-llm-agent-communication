import os
import sys
import argparse
from pathlib import Path

import json
import yaml
from tqdm import tqdm
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.pipelines.infer import (
    load_source,
    load_target,
    extract_thoughts,
    build_vec_injected,
    format_prompt_like_dataset,
)
from src.core.models import LIPAdapter
from src.integrations.hooks import make_lip_hook


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Setup LIP eval on HumanEval (no generation).")
    parser.add_argument("--config", type=str, default="configs/infer.yaml", help="Path to infer config YAML.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of examples.")
    parser.add_argument("--mode", type=str, default="lip", choices=["lip", "baseline"], help="Generation mode.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    ds = load_dataset("openai_humaneval", split="test")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    prompts = [ex["prompt"] for ex in ds]

    print("Loading target model...")
    tgt_model, tgt_tok = load_target(
        cfg["target_model"],
        cfg.get("runtime", {}).get("device_tgt", "auto"),
        cfg.get("runtime", {}).get("load_4bit", True),
    )

    if args.mode == "lip":
        print("Loading source model...")
        src_model, src_tok = load_source(cfg["source_model"], cfg.get("runtime", {}).get("device_src", "cuda"))

        print("Loading adapter...")
        adapter = LIPAdapter().to("cuda")
        state = __import__("torch").load(cfg["adapter_ckpt"], map_location="cuda", weights_only=False)
        adapter.load_state_dict(state)
        adapter.eval()

        print("Extracting thought vectors from source...")
        thought_vectors = extract_thoughts(prompts, src_model, src_tok, cfg.get("runtime", {}).get("device_src", "cuda"))
    else:
        adapter = None
        thought_vectors = None

    gen_kwargs = dict(
        max_new_tokens=int(cfg.get("generation", {}).get("max_new_tokens", 512)),
        do_sample=bool(cfg.get("generation", {}).get("do_sample", False)),
        temperature=float(cfg.get("generation", {}).get("temperature", 0.0)),
        repetition_penalty=float(cfg.get("generation", {}).get("repetition_penalty", 1.2)),
        pad_token_id=tgt_tok.eos_token_id,
    )

    layer_idx = int(cfg.get("lip", {}).get("layer_idx", -2))
    inject_pos_mode = str(cfg.get("lip", {}).get("inject_pos_mode", "last"))
    gain = float(cfg.get("lip", {}).get("gain", 10.0))

    out_path = Path(f"{args.mode}_humaneval_samples.jsonl")
    print(f"Generating completions to {out_path}...")

    with out_path.open("w", encoding="utf-8") as f:
        iterator = zip(ds, prompts) if args.mode == "baseline" else zip(ds, prompts, thought_vectors)
        for item in tqdm(iterator, total=len(prompts), desc="Generating"):
            if args.mode == "baseline":
                example, prompt = item
                prompt_text = format_prompt_like_dataset(prompt)
                inputs = tgt_tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(tgt_model.device)
                input_ids = inputs["input_ids"]
                attn_mask = inputs.get("attention_mask", None)

                with __import__("torch").no_grad():
                    gen = tgt_model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        **gen_kwargs,
                    )
            else:
                example, prompt, vec_cpu = item
                vec_injected = build_vec_injected(
                    vec_cpu,
                    adapter=adapter,
                    tgt_model=tgt_model,
                    gain=gain,
                )

                inputs = tgt_tok(prompt, return_tensors="pt", add_special_tokens=False).to(tgt_model.device)
                input_ids = inputs["input_ids"]
                attn_mask = inputs.get("attention_mask", None)

                if inject_pos_mode == "last":
                    inject_pos = input_ids.shape[1] - 1
                elif inject_pos_mode == "last_minus_1":
                    inject_pos = input_ids.shape[1] - 2
                else:
                    raise ValueError(f"Unknown inject_pos_mode: {inject_pos_mode}")

                hook_fn = make_lip_hook(
                    vec_injected=vec_injected,
                    inject_pos=inject_pos,
                    enable=True,
                )
                handle = tgt_model.model.layers[layer_idx].register_forward_hook(hook_fn)

                try:
                    with __import__("torch").no_grad():
                        gen = tgt_model.generate(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            **gen_kwargs,
                        )
                finally:
                    handle.remove()

            completion = tgt_tok.decode(gen[0], skip_special_tokens=True).replace("</s>", "")

            record = {"task_id": example.get("task_id"), "completion": completion}
            f.write(f"{json.dumps(record, ensure_ascii=True)}\n")


if __name__ == "__main__":
    main()
