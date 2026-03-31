"""
training_curriculum.py
Master Orchestrator. 
Delegates dataset management to DataManager, guarantees reproducibility,
and isolates vLLM/Unsloth processes to prevent OOM.
"""

import os
import multiprocessing as mp
from typing import List, Dict, Any
from training.notifier import mailman
from data_generation import DataManager
import traceback
import sys

# ==========================================
# CONFIGURATION
# ==========================================
BASE_MODEL = "Lixing-Li/Abyme-Qwen3.5-9B-SFT"
ADAPTER_DIR = "checkpoints/abyme-grpo-current"
HUB_REPO_PREFIX = "Lixing-Li/Abyme_GRPO_Iteration_"

CHECKPOINT_INTERVAL = 20 
GROUP_SIZE = 64
MAX_WORKERS = 50

# recusive arg
MAX_CALL = 20
MAX_DEPTH = 5
MAX_CHAIN_LENGTH = 5

# hyperparameters
TEMPERATURE = 0.05
ALPHA = 1.0

manager = DataManager(group_size=GROUP_SIZE, temperature=TEMPERATURE, alpha=ALPHA)

# ==========================================
# PROCESS ISOLATION WRAPPERS
# ==========================================


def _error_catching_wrapper(func, error_queue, *args, **kwargs):
    """
    Wraps target functions to catch exceptions and pass the full 
    traceback string back to the main process via a queue.
    """
    try:
        func(*args, **kwargs)
    except Exception as e:
        # Capture the full traceback string
        tb_str = traceback.format_exc()
        # Put the traceback into the queue for the main process
        error_queue.put(tb_str)
        # Exit forcefully so the main process registers exitcode != 0
        sys.exit(1)


def _generate_data_process(problem: Dict[str, Any], output_file: str, base_model: str, adapter_dir: str):
    from pathlib import Path
    from abyme.vllm_model import LocalVLLMModel
    from abyme.recursive_engine import RecursiveEngine
    
    has_adapter = os.path.exists(adapter_dir)
    model = LocalVLLMModel(model_path=base_model, lora_path=adapter_dir if has_adapter else None)
    engine = RecursiveEngine(base_model=model, max_workers=MAX_WORKERS, max_depth=MAX_DEPTH, max_call=MAX_CALL, max_chain_length=MAX_CHAIN_LENGTH)

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("") 
    
    manager.generate(problem, engine, path)
    model.shutdown()

def _train_model_process(base_model: str, data_file: str, adapter_dir: str):
    from run_training import run_training
    
    if not os.path.exists(data_file) or os.path.getsize(data_file) == 0:
        print("    [Info] No valid data generated for this step. Skipping training update.")
        return
        
    run_training(
        base_model_path=base_model, 
        local_jsonl_path=data_file, 
        output_adapter_dir=adapter_dir, 
        resume=os.path.exists(adapter_dir)
    )

def score_by_level(test_data: List[Dict[str, Any]], scores: List[float]) -> tuple:
    """Compute per-level averages from a flat scores list aligned with test_data.

    Returns:
        (total_avg, l1_avg, l2_avg, l3_avg, l4_avg, l5_avg)
        Missing levels default to 0.0.
    """
    level_totals: Dict[int, List[float]] = {l: [] for l in range(1, 6)}
    for item, s in zip(test_data, scores):
        lvl = item.get("level_num", 1)
        if lvl in level_totals:
            level_totals[lvl].append(s)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    total = _avg(scores)
    return (total, _avg(level_totals[1]), _avg(level_totals[2]), _avg(level_totals[3]), _avg(level_totals[4]), _avg(level_totals[5]))


def append_score_to_hub(
    scores: tuple,
    test_name: str,
    repo_id: str = "Lixing-Li/Model-Scores",
):
    """Append a score record to a HuggingFace dataset.

    Args:
        scores: (total_avg, level1, level2, level3, level4, level5)
        test_name: Identifier stored as metadata.
        repo_id: HuggingFace dataset repo to push to.
    """
    import os
    import dotenv
    from huggingface_hub import login
    from datasets import Dataset, concatenate_datasets, load_dataset

    dotenv.load_dotenv()
    login(token=os.getenv("HF_TOKEN", ""), add_to_git_credential=True)

    total, l1, l2, l3, l4, l5 = scores
    new_record = {
        "test_name": test_name,
        "total": total,
        "level_1": l1,
        "level_2": l2,
        "level_3": l3,
        "level_4": l4,
        "level_5": l5,
    }
    new_ds = Dataset.from_list([new_record])
    try:
        existing_ds = load_dataset(repo_id, split="train")
        combined = concatenate_datasets([existing_ds, new_ds])
    except Exception:
        combined = new_ds
    combined.push_to_hub(repo_id)
    print(f"Score record pushed to https://huggingface.co/datasets/{repo_id}")


def clean_score_from_hub(repo_id: str = "Lixing-Li/Model-Scores", filter_test_name: str = ""):
    """Delete the first row matching filter_test_name from the HuggingFace dataset.

    Args:
        repo_id: HuggingFace dataset repo.
        filter_test_name: test_name value to remove.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(repo_id, split="train")
        first_match = next((i for i, x in enumerate(ds) if x["test_name"] == filter_test_name), None)
        if first_match is None:
            print(f"No record found with test_name='{filter_test_name}'")
            return
        indices = [i for i in range(len(ds)) if i != first_match]
        filtered = ds.select(indices)
        filtered.push_to_hub(repo_id, private=True)
        print(f"Deleted one record with test_name='{filter_test_name}'.")
    except Exception as e:
        print(f"Failed to clean dataset: {e}")


def _checkpoint_process(base_model: str, adapter_dir: str, iteration: int):
    from run_training import merge_and_upload
    from abyme.vllm_model import LocalVLLMModel

    hub_repo_id = f"{HUB_REPO_PREFIX}{iteration}"
    print(f"\n{'='*50}\nREACHED CHECKPOINT: ITERATION {iteration}\n{'='*50}")

    merge_and_upload(base_model, adapter_dir, hub_repo_id)

    print(f"\nTesting {hub_repo_id} on the official test set...")

    # Reload the exact 50-per-level test set from disk
    test_data = manager.load_test_set()

    model = LocalVLLMModel(model_path=hub_repo_id)
    avg, scores = manager.score_all(
        model=model,
        test_data=test_data,
        iteration=iteration,
        max_workers=MAX_WORKERS,
        max_depth=MAX_DEPTH,
        max_call=MAX_CALL,
        max_chain_length=MAX_CHAIN_LENGTH
    )
    model.shutdown()

    level_scores = score_by_level(test_data, scores)
    total, l1, l2, l3, l4, l5 = level_scores
    mailman.send(
        f"Checkpoint {iteration} | {hub_repo_id}\n"
        f"Avg: {avg:.4f} | Total: {total:.4f}\n"
        f"L1: {l1:.4f} | L2: {l2:.4f} | L3: {l3:.4f} | L4: {l4:.4f} | L5: {l5:.4f}"
    )
    append_score_to_hub(scores=level_scores, test_name=f"grpo_iteration_{iteration}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def run_curriculum(base_model: str = "Lixing-Li/Abyme-Qwen3.5-9B-SFT", starting_problem: int = 0):
    # Guarantee OS-level memory clearance
    mp.set_start_method('spawn', force=True)
    os.makedirs("results/step_data", exist_ok=True)

    # Delete existing LoRA adapter checkpoint before starting
    import shutil
    if os.path.exists(ADAPTER_DIR):
        print(f"Deleting existing adapter checkpoint: {ADAPTER_DIR}")
        shutil.rmtree(ADAPTER_DIR)
    
    print("Initializing Data Manager...")
    master_manager = DataManager()
    
    # Prepare datasets (will skip if already exists on disk)
    master_manager.prepare_datasets(raw_data_file="data/math_full.jsonl", test_samples_per_level=50)
    
    # Load exact state from disk
    curriculum = master_manager.load_curriculum()
    test_data = master_manager.load_test_set()
    
    print(f"Curriculum ready: {len(curriculum)} problems.")
    print(f"Test Set ready: {len(test_data)} problems.")
    
    if starting_problem > 0:
        print(f"\nResuming curriculum from problem index: {starting_problem}")

    # Calculate current checkpoint iteration based on the starting problem
    iteration = starting_problem // CHECKPOINT_INTERVAL

    for step in range(starting_problem, len(curriculum)):
        problem = curriculum[step]
        
        prompt_preview = problem.get('problem', '')
        if len(prompt_preview) > 200:
            prompt_preview = prompt_preview[:200] + "..."

        print(f"\n{'='*80}")
        mailman.send(f"TRAINING PROBLEM: {step+1}/{len(curriculum)}")
        print(f"CHECKPOINT    : Iteration {iteration}")
        print(f"TYPE          : {problem.get('type', '?')}")
        print(f"LEVEL         : {problem.get('level_num', '?')}")
        print(f"PROMPT        : {prompt_preview}")
        print(f"{'='*80}")

        step_data_file = f"results/step_data/step_{step}.jsonl"

        # ==========================================
        # 1. GENERATE
        # ==========================================
        error_queue = mp.Queue()
        p_gen = mp.Process(
            target=_error_catching_wrapper, 
            args=(_generate_data_process, error_queue, problem, step_data_file, base_model, ADAPTER_DIR)
        )
        p_gen.start()
        p_gen.join()

        if p_gen.exitcode != 0:
            tb = error_queue.get() if not error_queue.empty() else "Unknown Error (No traceback captured)"
            msg = f"[FATAL] Generation crashed at step {step}.\n\nTraceback:\n{tb}"
            print(msg)
            mailman.send(msg)
            raise RuntimeError(msg)

        if not os.path.exists(step_data_file) or os.path.getsize(step_data_file) == 0:
            print(f"    [Skip] No valid data generated for step {step} (all generations failed). Skipping training.")
            continue

        # ==========================================
        # 2. TRAIN
        # ==========================================
        error_queue = mp.Queue()
        p_train = mp.Process(
            target=_error_catching_wrapper, 
            args=(_train_model_process, error_queue, base_model, step_data_file, ADAPTER_DIR)
        )
        p_train.start()
        p_train.join()

        if p_train.exitcode != 0:
            tb = error_queue.get() if not error_queue.empty() else "Unknown Error (No traceback captured)"
            msg = f"[FATAL] Training crashed at step {step}.\n\nTraceback:\n{tb}"
            print(msg)
            mailman.send(msg)
            raise RuntimeError(msg)

        # ==========================================
        # 3. CHECKPOINT & TEST
        # ==========================================
        if (step + 1) % CHECKPOINT_INTERVAL == 0:
            iteration += 1
            error_queue = mp.Queue()
            p_check = mp.Process(
                target=_error_catching_wrapper, 
                args=(_checkpoint_process, error_queue, base_model, ADAPTER_DIR, iteration)
            )
            p_check.start()
            p_check.join()

            if p_check.exitcode != 0:
                tb = error_queue.get() if not error_queue.empty() else "Unknown Error (No traceback captured)"
                msg = f"[FATAL] Checkpoint {iteration} crashed.\n\nTraceback:\n{tb}"
                print(msg)
                mailman.send(msg)
                raise RuntimeError(msg)

            # Checkpoint succeeded — merged model is the new base; reset adapter.
            base_model = f"{HUB_REPO_PREFIX}{iteration}"
            import shutil
            if os.path.exists(ADAPTER_DIR):
                shutil.rmtree(ADAPTER_DIR)
            print(f"Base model updated to: {base_model}")

if __name__ == "__main__":
    run_curriculum(
        #base_model="Lixing-Li/Abyme-Qwen3.5-9B-SFT", 
        #base_model="Lixing-Li/Abyme-Trained-Iteration-5",
        base_model="Lixing-Li/Abyme_GRPO_Iteration_2",
        starting_problem=43
    )