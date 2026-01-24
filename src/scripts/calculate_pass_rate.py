import json
import os

from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness


def main():
    sample_file = "experiments/logs/lip_humaneval_samples.jsonl"
    print(f"Calculating Pass@1 for {sample_file}...")

    problems = read_problems()
    sample_task_ids = []
    with open(sample_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task_id = record.get("task_id")
            if task_id:
                sample_task_ids.append(task_id)

    sample_task_ids = list(dict.fromkeys(sample_task_ids))
    subset = [problems[task_id] for task_id in sample_task_ids if task_id in problems]
    subset_path = os.path.join("experiments", "logs", "humaneval_subset.jsonl")
    os.makedirs(os.path.dirname(subset_path), exist_ok=True)
    write_jsonl(subset_path, subset)

    results = evaluate_functional_correctness(sample_file, problem_file=subset_path)

    print("\nFinal results:")
    print(results)


if __name__ == "__main__":
    main()
