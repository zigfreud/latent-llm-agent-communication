# scripts/run_infer.py
import argparse
import yaml

from src.pipelines.infer import run_inference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/infer.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_inference(cfg)


if __name__ == "__main__":
    main()
