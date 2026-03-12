#!/bin/bash

# Quick run script for Abyme Visualizer

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup..."
    ./setup.sh
fi

echo "Starting Abyme Visualizer..."
echo ""
source venv/bin/activate
python app.py
