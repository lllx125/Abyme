#!/bin/bash

set -e  # Exit on error

echo "==========================================  "
echo "Abyme Training Environment Setup"
echo "=========================================="

# Check GPU
echo "Checking GPU availability..."
nvidia-smi

# 1. Install Python 3.12 and system dependencies
echo "Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip tmux

# 2. Create the Virtual Environment
ENV_NAME="venv"
echo "Creating virtual environment: $ENV_NAME"
python3.12 -m venv $ENV_NAME

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

# 7. other dependencies
pip install --no-deps xformers trl peft accelerate bitsandbytes
pip install flash-attn --no-build-isolation
pip install vllm
pip install --upgrade transformers  # must come after vllm to override its pinned version

# 8. Install the abyme package in editable mode
echo "Installing abyme package..."
cd abyme-rllm/
pip install -e .

# auto activate the environment when opening a new terminal
echo "source $ENV_NAME/bin/activate" >> ~/.bashrc

# 9 Install looong dependencies
pip install "causal-conv1d>=1.2.0"
pip install flash-linear-attention