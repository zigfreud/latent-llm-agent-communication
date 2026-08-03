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
    except ImportError:
        pass

    return torch.device("cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir):
    logger_name = f"lip.{os.path.abspath(output_dir)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        os.path.join(output_dir, "experiment.log"),
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger
