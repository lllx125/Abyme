#!/bin/bash

set -e  # Exit on error

echo "==========================================  "
echo "Abyme Training Environment Setup"
echo "=========================================="

# Check GPU
echo "Checking GPU availability..."
nvidia-smi

# 1. Install standard system dependencies for virtual environments
echo "Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install python3-venv python3-pip tmux -y

# 2. Create the Virtual Environment
ENV_NAME="venv"
echo "Creating virtual environment: $ENV_NAME"
python3 -m venv $ENV_NAME

# 3. Activate the environment
echo "Activating virtual environment..."
source $ENV_NAME/bin/activate

# 4. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 5. Install PyTorch configured for CUDA 12.1 (Optimal for A100)
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. Install Unsloth and Hugging Face Dataset tools
echo "Installing Unsloth and Hugging Face tools..."
pip install unsloth unsloth_zoo datasets huggingface_hub

# 7. Pin the EXACT library versions required for Uncontaminated Packing
echo "Installing specific versions for uncontaminated packing..."
pip install transformers==4.56.2
pip install --no-deps "xformers<0.0.29" trl==0.22.2 peft accelerate bitsandbytes

# 8. Install the abyme package in editable mode
echo "Installing abyme package..."
pip install -e .

echo "=========================================="
echo "Setup complete!"
echo "To activate the environment, run:"
echo "  source $ENV_NAME/bin/activate"
echo "=========================================="
