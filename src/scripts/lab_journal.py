import os
import json
import torch
import datetime
import matplotlib.pyplot as plt

class LabJournal:
    def __init__(self, experiment_name, config):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.exp_id = f"{timestamp}_{experiment_name}"
        base_dir = os.path.join("experiments", "experiments_log")
        self.dir = os.path.join(base_dir, self.exp_id)

        os.makedirs(self.dir, exist_ok=True)

        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        self.metrics = {"loss": [], "mse": [], "cos": []}
        print(f"📔 Lab Journal opened at: {self.dir}")

    def log_metric(self, epoch, loss, mse, cos):
        self.metrics["loss"].append(loss)
        self.metrics["mse"].append(mse)
        self.metrics["cos"].append(cos)
        self._plot_metrics()

    def _plot_metrics(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["loss"], label="Total Loss")
        plt.plot(self.metrics["mse"], label="MSE (Magnitude)")
        plt.plot(self.metrics["cos"], label="Cosine (Direction)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Training Dynamics: {self.exp_id}")
        plt.savefig(os.path.join(self.dir, "training_curve.png"))
        plt.close()

    def save_model(self, model, filename="adapter.pth"):
        torch.save(model.state_dict(), os.path.join(self.dir, filename))
        with open(os.path.join(self.dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f)