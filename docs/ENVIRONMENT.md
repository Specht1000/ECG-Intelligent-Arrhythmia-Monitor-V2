# Machine-Learning Environment

## Supported runtime

- Python 3.11, 64-bit
- Windows CPU training with PyTorch
- Project-local virtual environment at `.venv/`

The legacy Python 3.7 32-bit installation is not suitable for current PyTorch
packages and must not be used for model training.

## Recreate the environment

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_ml_env.ps1
```

The setup installs the CPU-specific PyTorch wheel from the official PyTorch index,
then installs the remaining packages from `requirements-ml.txt`.

## Activate

```powershell
.\.venv\Scripts\Activate.ps1
```

All commands can also use the environment without activation:

```powershell
.\.venv\Scripts\python.exe host/ai/plot_random_ecg.py --show
```

## Verify

```powershell
.\.venv\Scripts\python.exe scripts/verify_ml_env.py
.\.venv\Scripts\python.exe -m unittest discover -s host/tests -v
```

`requirements-lock.txt` records the exact resolved package versions from the last
verified setup. `requirements-ml.txt` remains the human-maintained compatibility
manifest.
