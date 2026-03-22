"""
Training Loop Pipeline

This module orchestrates the complete iterative training loop:
Generation -> Rating -> KTO Training -> Testing -> Scoring -> Level Upgrade
"""

from typing import Tuple

from training.data_generation import DataManager
from training.run_training import run_training
from abyme.vllm_model import LocalVLLMModel
from training.notifier import mailman

NUM_QUESTIONS = 50
NUM_GEN_PER_QUESTION = 10

BASE_LINE_SCORE = [0.883,0.878,0.851,0.766,0.574]
LEVELS = [
    (0.2,0.2,0.2,0.2,0.2),
    (1.0,0.0,0.0,0.0,0.0),  
    (0.2,0.8,0.0,0.0,0.0),
    (0.1,0.2,0.7,0.0,0.0),
    (0.1,0.1,0.2,0.6,0.0),
    (0.1,0.1,0.1,0.2,0.5),
]


def run_single_interation(base_model:str, iteration:int, sample:Tuple[float,float,float,float,float], **recursive_kwargs):
    # Generate the dataset
    if not sum(sample) == 1:
        raise ValueError("Sample frequencies must sum to 1.")
    training_sample = (int(sample[0]*NUM_QUESTIONS), int(sample[1]*NUM_QUESTIONS), int(sample[2]*NUM_QUESTIONS), int(sample[3]*NUM_QUESTIONS), int(sample[4]*NUM_QUESTIONS))
    test_samples = training_sample
    mailman.send(f"Training Loop Started: Iteration {iteration} has started with sample distribution: {sample}")
    # define base model
    # model = LocalVLLMModel(model_path=base_model)
    # mailman.send(f"Model {base_model} loaded successfully!")
    # define the dataset manager
    data_manager = DataManager(iteration=iteration, samples=training_sample, tests = test_samples, num_gen_per_question=NUM_GEN_PER_QUESTION)
    # run the model on the questions
    # data_manager.generate_all(model,**recursive_kwargs)
    # # create the dataset for training by rating the generated data
    # data_manager.rate_all()
    # mailman.send(f"Data generation and rating completed for iteration {iteration}. Starting KTO training.")
    
    # # run the KTO training
    # run_training(model_name=base_model, dataset_id=data_manager.hf_repo_rated, hub_repo_id=data_manager.hf_repo_trained_model)
    # mailman.send(f"KTO Training completed for iteration {iteration}. Starting testing and scoring.")
    
    # test the trained model and score the results
    trained_model = LocalVLLMModel(model_path=data_manager.hf_repo_trained_model)
    
    data_manager.test_all(trained_model, **recursive_kwargs)
    data_manager.score_all()
    results = data_manager.check_scores_by_level()
    mailman.send(f"Iteration {iteration} completed. Average Scores: {results[0]}.")

if __name__ == "__main__":
    base_model = "Lixing-Li/Abyme-Qwen3.5-9B-SFT"
    recursive_kwargs = {
        "max_depth": 5,
        "max_call": 50,
        "max_chain_length": 5
    }
    run_single_interation(base_model=base_model, iteration=1, sample=(1.0,0.0,0.0,0.0,0.0), **recursive_kwargs)
    
    
    
    
