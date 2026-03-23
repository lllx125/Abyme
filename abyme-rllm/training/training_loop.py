"""
Training Loop Pipeline

This module orchestrates the complete iterative training loop:
Generation -> Rating -> KTO Training -> Testing -> Scoring -> Level Upgrade
"""

import multiprocessing
from typing import Tuple

from training.data_generation import DataManager
from training.notifier import mailman

NUM_GEN_PER_QUESTION = 10
TEST_SAMPLES = (20, 20, 20, 20, 20)

BASE_LINE_SCORE = [0.883,0.878,0.851,0.766,0.574] # from Qwen3.5-9B on MATH-500 benchmark, by level
LEVELS = [
    (20, 20, 20, 20, 20), # Level 0
    (80, 5, 5, 5, 5),     # Level 1
    (20, 65, 5, 5, 5),    # Level 2
    (5, 20, 65, 5, 5),    # Level 3
    (5, 5, 20, 65, 5),    # Level 4
    (5, 5, 5, 20, 65)     # Level 5
]

BASE_MODEL = "Lixing-Li/Abyme-Qwen3.5-9B-SFT"


def _generate(dm, base_model_path, recursive_kwargs):
    from abyme.vllm_model import LocalVLLMModel
    model = LocalVLLMModel(model_path=base_model_path)
    dm.generate_all(model, **recursive_kwargs)


def _train(model_name, dataset_id, hub_repo_id):
    from training.run_training import run_training
    run_training(model_name=model_name, dataset_id=dataset_id, hub_repo_id=hub_repo_id)


def _test(dm, trained_model_path, recursive_kwargs):
    from abyme.vllm_model import LocalVLLMModel
    trained_model = LocalVLLMModel(model_path=trained_model_path)
    dm.test_all(trained_model, **recursive_kwargs)


def _spawn(target, args):
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=target, args=args)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Subprocess {target.__name__} failed with exit code {p.exitcode}")


def run_single_interation(base_model:str, iteration:int, sample:Tuple[int, int, int, int, int], **recursive_kwargs) -> Tuple[str, float, float, float, float, float, float]:
    training_sample = sample
    test_samples = TEST_SAMPLES
    mailman.send(f"Training Loop Started: Iteration {iteration} has started with sample distribution: {sample}")

    dm = DataManager(iteration=iteration, samples=training_sample, tests=test_samples, num_gen_per_question=NUM_GEN_PER_QUESTION)

    # generation subprocess — GPU memory fully freed on exit
    _spawn(_generate, (dm, base_model, recursive_kwargs))
    mailman.send(f"Model {base_model} generation completed!")

    dm.rate_all()
    mailman.send(f"Data generation and rating completed for iteration {iteration}. Starting KTO training.")

    # training subprocess — GPU memory fully freed on exit
    _spawn(_train, (base_model, dm.hf_repo_rated, dm.hf_repo_trained_model))
    mailman.send(f"KTO Training completed for iteration {iteration}. Starting testing and scoring.")

    # inference subprocess — GPU memory fully freed on exit
    _spawn(_test, (dm, dm.hf_repo_trained_model, recursive_kwargs))

    dm.score_all()
    results = dm.check_scores_by_level()
    mailman.send(f"Iteration {iteration} completed. Average: {results[0]:.4f} | L1: {results[1]:.4f} L2: {results[2]:.4f} L3: {results[3]:.4f} L4: {results[4]:.4f} L5: {results[5]:.4f}.")
    return (dm.hf_repo_trained_model, *results)


def run_training_loop(start_model: str, start_iteration: int, start_level: int, max_iterations_per_level: int = 5):
    """
    Run the training until the model at each level reaches the baseline score or we hit the max iterations per level.
    """
    current_model = start_model
    next_iteration = start_iteration + 1
    level = start_level
    iterations_this_level = 0
    while level <= 5 and iterations_this_level < max_iterations_per_level:
        mailman.send(f"Starting iteration {next_iteration} at level {level} with model {current_model}")
        next_model, scores = run_single_interation(base_model=current_model, iteration=next_iteration, sample=LEVELS[level], **recursive_kwargs)
        if scores[level] >= BASE_LINE_SCORE[level]:
            mailman.send(f"Level {level} passed with score {scores[level]:.4f}! Moving to next level.")
            level += 1
            iterations_this_level = 0
        else:
            mailman.send(f"Level {level} not passed. Current score: {scores[level]:.4f}. Retrying with new model.")
            iterations_this_level += 1
        current_model = next_model
        next_iteration += 1

    mailman.send(f"Training loop completed. Final model: {current_model}, final level: {level}")

def score_base_model(base_model:str, recursive_kwargs):
    dm = DataManager(iteration=0, samples=(20,20,20,20,20), tests=(20,20,20,20,20), num_gen_per_question=NUM_GEN_PER_QUESTION)
    _spawn(_test, (dm, base_model, recursive_kwargs))
    dm.score_all()
    results = dm.check_scores_by_level()

def clean_score_from_hub(repo_id: str = "Lixing-Li/Model-Scores", filter_test_name: str = "SFT-base"):
    """Delete all records except for SFT-base"""
    from datasets import load_dataset

    try:
        ds = load_dataset(repo_id, split="train")
        filtered = ds.filter(lambda x: x["test_name"] == filter_test_name)
        filtered.push_to_hub(repo_id, private=True)
        print(f"Cleaned dataset, only {filter_test_name} records remain.")
    except Exception as e:
        print(f"Failed to clean dataset: {e}")

if __name__ == "__main__":
    #base_model = "Lixing-Li/Abyme-Trained-Iteration-1"
    base_model = BASE_MODEL
    recursive_kwargs = {
        "max_depth": 5,
        "max_call": 70,
        "max_chain_length": 5
    }
    #clean_score_from_hub()
    #score_base_model(base_model=base_model, recursive_kwargs=recursive_kwargs)
    #run_single_interation(base_model=base_model, iteration=2, sample=(1.0,0.0,0.0,0.0,0.0), **recursive_kwargs)
    run_training_loop(start_model=base_model, start_iteration=0, start_level=1, max_iterations_per_level=5)
