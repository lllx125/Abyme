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
from abyme.vllm_models import LocalVLLMModel

NUM_QUESTIONS = 50
NUM_GEN_PER_QUESTION = 10


def run_single_interation(base_model:str, iteration:int, sample:Tuple[float,float,float,float,float]):
    # Generate the dataset
    if not sum(sample) == 1:
        raise ValueError("Sample frequencies must sum to 1.")
    training_sample = (int(sample[0]*NUM_QUESTIONS), int(sample[1]*NUM_QUESTIONS), int(sample[2]*NUM_QUESTIONS), int(sample[3]*NUM_QUESTIONS), int(sample[4]*NUM_QUESTIONS))
    testing_sample = (int(sample[0]*NUM_QUESTIONS*0.5), int(sample[1]*NUM_QUESTIONS*0.5), int(sample[2]*NUM_QUESTIONS*0.5), int(sample[3]*NUM_QUESTIONS*0.5), int(sample[4]*NUM_QUESTIONS*0.5))
    # define base model
    model = LocalVLLMModel(model_name=base_model)
   
    # define the dataset manager
    data_manager = DataManager(iteration=iteration, samples=training_sample, tests = testing_sample, num_gen_per_question=NUM_GEN_PER_QUESTION)
    # run the model on the questions
    data_manager.generate_all(model)
    # create the dataset for training by rating the generated data
    data_manager.rate_all()
    
    # run the KTO training
    run_training(model_name=base_model, dataset_id=data_manager.hf_repo_rated, hub_repo_id=data_manager.hf_repo_trained_model)
    
    # test the trained model and score the results
    trained_model = LocalVLLMModel(model_name=data_manager.hf_repo_trained_model)
    
    data_manager.test_all(trained_model)
    #data_manager.score_all()
    

if __name__ == "__main__":
    base_model = "Lixing-Li/Abyme-Base-Model"
    iteration = 1
    sample = (1.0,0.0,0.0,0.0,0.0) 
    run_single_interation(base_model, iteration, sample)
    
    
    
