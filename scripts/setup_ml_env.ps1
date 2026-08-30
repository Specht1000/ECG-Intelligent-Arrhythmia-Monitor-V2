param(
    [string]$PythonExecutable = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VirtualEnvironment = Join-Path $ProjectRoot ".venv"
$VirtualPython = Join-Path $VirtualEnvironment "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Python 3.11 executable not found: $PythonExecutable"
}

& $PythonExecutable -m venv $VirtualEnvironment
& $VirtualPython -m pip install --upgrade pip setuptools wheel
& $VirtualPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& $VirtualPython -m pip install --requirement (Join-Path $ProjectRoot "requirements-ml.txt")

Write-Output "Environment ready: $VirtualEnvironment"
& $VirtualPython -c "import platform, sys, torch; print(sys.version); print(platform.architecture()); print('torch', torch.__version__); print('device', 'cuda' if torch.cuda.is_available() else 'cpu')"
