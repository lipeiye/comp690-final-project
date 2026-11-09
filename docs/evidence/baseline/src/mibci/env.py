"""Software and hardware details recorded with every result."""

import importlib.metadata as md
import platform
import subprocess

import torch


def environment() -> dict:
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except FileNotFoundError:  # not macOS
        cpu = platform.processor()
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": cpu,
        "torch_threads": torch.get_num_threads(),
        "packages": {
            p: md.version(p) for p in ["torch", "braindecode", "numpy", "scipy", "scikit-learn"]
        },
    }
