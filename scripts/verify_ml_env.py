"""Verify the ECG V2 machine-learning environment and report package versions."""

import importlib
import os
import platform
import struct
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))


REQUIRED_MODULES = (
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "wfdb",
    "torch",
    "matplotlib",
    "seaborn",
    "tqdm",
    "yaml",
    "tensorboard",
)


def module_version(module) -> str:
    return str(getattr(module, "__version__", "installed"))


def main() -> int:
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("Executable: {}".format(sys.executable))
    print("Architecture: {}-bit".format(struct.calcsize("P") * 8))
    print("Platform: {}".format(platform.platform()))

    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("The ML environment must use a 64-bit Python runtime")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("The verified ML runtime is Python 3.11")

    loaded = {}
    for name in REQUIRED_MODULES:
        module = importlib.import_module(name)
        loaded[name] = module
        print("{:<12} {}".format(name, module_version(module)))

    torch = loaded["torch"]
    print("PyTorch device: {}".format("cuda" if torch.cuda.is_available() else "cpu"))

    required_paths = (
        PROJECT_ROOT / "database" / "mit_bih",
        PROJECT_ROOT
        / "database"
        / "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0",
        PROJECT_ROOT
        / "database"
        / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    )
    for path in required_paths:
        if not path.is_dir():
            raise FileNotFoundError("Dataset directory not found: {}".format(path))
        print("Dataset OK: {}".format(path.name))

    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = torch.sum(tensor).item()
    if result != 10.0:
        raise RuntimeError("Unexpected PyTorch tensor result: {}".format(result))
    print("PyTorch tensor check: OK")

    convolution = torch.nn.Conv1d(in_channels=12, out_channels=8, kernel_size=7, padding=3)
    mock_ecg = torch.randn(2, 12, 5000)
    output = convolution(mock_ecg)
    if tuple(output.shape) != (2, 8, 5000):
        raise RuntimeError("Unexpected Conv1d output shape: {}".format(tuple(output.shape)))
    print("PyTorch Conv1d check: OK ({})".format(tuple(output.shape)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
