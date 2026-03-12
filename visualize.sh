#!/bin/bash
# One-click launcher for Abyme Visualizer
cd "$(dirname "$0")/abyme-visualizer" && source venv/bin/activate 2>/dev/null || (echo "Installing..." && ./setup.sh && source venv/bin/activate) && python app.py
