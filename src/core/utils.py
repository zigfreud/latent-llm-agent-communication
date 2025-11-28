import os
import torch
import random
import logging
import numpy as np


def get_device(device_str="auto"):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if device_str == "dml":
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            print("⚠️ DirectML is not installed. Use 'pip install torch-directml'. Using cpu.")
            return torch.device("cpu")

    if device_str == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available(): return torch.device("cuda")

    try:
        import torch_directml
        if torch_directml.is_available(): return torch_directml.device()
    except: pass

    return torch.device("cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir):
    logging.basicConfig(
        filename=os.path.join(output_dir, 'experiment.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    return logging.getLogger(__name__)