import os
import json
import matplotlib.pyplot as plt


DATA_DIR = os.path.join("experiments", "logs")
OUTPUT_DIR = os.path.join("paper", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [
    ("Gen1_MSE_Only.json", "Gen 1: Naive (MSE)", "red", "--"),
    ("Gen2_Cosine_Only.json", "Gen 2: Geometric (Cosine)", "orange", "-."),
    ("Gen4_LIP_Hybrid.json", "Gen 4: LIP Protocol (Hybrid)", "blue", "-")
]

plt.figure(figsize=(10, 6))
print("📊 Plotting Training Dynamics...")

for filename, label, color, style in files:
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            epochs = [d['epoch'] for d in data]
            sims = [d['similarity'] for d in data]
            plt.plot(epochs, sims, label=label, color=color, linestyle=style, linewidth=2)
    else:
        print(f"⚠️ Warning: File {filename} not found in {DATA_DIR}. Run benchmark_gen.py first.")

plt.title("Semantic Convergence Evolution (TinyLlama -> Llama-3)", fontsize=14)
plt.xlabel("Training Epochs", fontsize=12)
plt.ylabel("Mean Cosine Similarity", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "convergence_comparison.png")
plt.savefig(save_path, dpi=300)
print(f"✅ Graph saved to: {save_path}")