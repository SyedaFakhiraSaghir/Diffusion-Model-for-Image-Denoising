#!/bin/bash
echo "Setting up proper environment for Diffusion Model..."

# Clean up any existing environment
deactivate 2>/dev/null || true
rm -rf diffusion-env

# Create fresh virtual environment
python3 -m venv diffusion-env

# Activate environment
source diffusion-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install compatible versions - FORCE numpy 1.x
pip install "numpy<2"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib scikit-learn pillow

echo "=== Environment Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); import numpy as np; print(f'NumPy: {np.__version__}')"

echo "Environment ready! Run: source diffusion-env/bin/activate && python fixed_main.py"