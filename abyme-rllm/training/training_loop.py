"""
Training Loop Pipeline

This module orchestrates the complete iterative training loop:
Generation -> Rating -> KTO Training -> Testing -> Scoring -> Level Upgrade
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime

from training.data_generation import DataManager
from training.run_training import run_training
from abyme.model import Model
from abyme.core import Abyme_API_Model

NUM_QUESTIONS = 50
NUM_GEN_PER_QUESTION = 10


def run_single_interation(base_model:str, iteration:int, sample:Tuple[float,float,float,float,float]):
    # Generate the dataset
    if not sum(sample) == 1:
        raise ValueError("Sample frequencies must sum to 1.")
    training_sample = (int(sample[0]*NUM_QUESTIONS), int(sample[1]*NUM_QUESTIONS), int(sample[2]*NUM_QUESTIONS), int(sample[3]*NUM_QUESTIONS), int(sample[4]*NUM_QUESTIONS))
    testing_sample = (int(sample[0]*NUM_QUESTIONS*0.5), int(sample[1]*NUM_QUESTIONS*0.5), int(sample[2]*NUM_QUESTIONS*0.5), int(sample[3]*NUM_QUESTIONS*0.5), int(sample[4]*NUM_QUESTIONS*0.5))
    data_manager = DataManager( iteration=iteration, sample=training_sample, test = testing_sample, num_gen_per_question=NUM_GEN_PER_QUESTION)
