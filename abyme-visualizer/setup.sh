#!/bin/bash

# Abyme Visualizer - Setup Script
# Installs dependencies and prepares the environment

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo "Abyme Visualizer - Setup"
echo "========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Using Python: $PYTHON_CMD"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtualenv
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install -q --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Install abyme-rllm in development mode
echo ""
echo "Installing abyme-rllm..."
cd "$SCRIPT_DIR/../abyme-rllm"
pip install -q -e .

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "You can now run: ./visualize.sh"
echo ""
