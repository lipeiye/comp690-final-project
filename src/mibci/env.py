"""Software and hardware details recorded with every result."""

import importlib.metadata as md
import platform
import subprocess

import torch


def environment() -> dict:
    ram_bytes = None
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        ram = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
        ram_bytes = int(ram) if ram.isdigit() else None
    except FileNotFoundError:  # not macOS
        cpu = platform.processor()
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": cpu,
        "physical_memory_bytes": ram_bytes,
        "torch_threads": torch.get_num_threads(),
        "packages": {
            p: md.version(p) for p in ["torch", "braindecode", "numpy", "scipy", "scikit-learn"]
        },
    }
