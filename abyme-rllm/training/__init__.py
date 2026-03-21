"""
Iterative Training Loop for MATH Dataset

This package implements an iterative training loop for the MATH dataset using:
- Data generation with Abyme recursive models
- KTO (Kahneman-Tversky Optimization) training with unsloth
- Curriculum learning with level-based progression
"""

from .data_generation import DataManagement
from .run_training import (
    run_kto_training,
    load_trained_model,
    create_inference_model,
    prepare_kto_dataset
)
from .training_loop import TrainingLoop

__all__ = [
    "DataManagement",
    "run_kto_training",
    "load_trained_model",
    "create_inference_model",
    "prepare_kto_dataset",
    "TrainingLoop",
]
