#!/bin/bash

# Abyme Visualizer - Quick Setup Script

echo "========================================="
echo "Abyme Visualizer Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo ""
echo "Installing Python dependencies in venv..."
source venv/bin/activate
pip install -r requirements.txt
pip install openai  # For DeepSeek API

# Check for DeepSeek API key
echo ""
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "⚠️  WARNING: DEEPSEEK_API_KEY not found in environment"
    echo "Please set your API key:"
    echo "  export DEEPSEEK_API_KEY='your-api-key-here'"
    echo "  or add it to ../abyme-rllm/.env"
else
    echo "✓ DEEPSEEK_API_KEY found"
fi

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To start the visualizer:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Or use the run.sh script:"
echo "  ./run.sh"
echo ""
echo "Then open in your browser:"
echo "  http://localhost:5000"
echo ""
