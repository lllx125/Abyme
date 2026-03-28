"""
training_curriculum.py

Master Orchestrator for Recursive GRPO.
Features:
1. OS-Level Process Isolation (Prevents vLLM/Unsloth OOMs).
2. Dynamic Curriculum (Progressive mixing of Levels 1-5).
3. Automated Checkpointing, Merging, and Testing.
"""

import os
import json
import random
import multiprocessing as mp
from typing import List, Dict, Any

# ==========================================
# CONFIGURATION
# ==========================================
BASE_MODEL = "Lixing-Li/Abyme-Qwen3.5-9B-SFT"
ADAPTER_DIR = "checkpoints/abyme-grpo-current"
HUB_REPO_ID = "Lixing-Li/Abyme-Qwen3.5-9B-GRPO-Live"

CHECKPOINT_INTERVAL = 50  # Merge, upload, and test every 50 problems
GROUP_SIZE = 64
MAX_WORKERS = 60

# ==========================================
# PROCESS ISOLATION WRAPPERS
# ==========================================
def _generate_data_process(problem: Dict[str, Any], output_file: str, base_model: str, adapter_dir: str):
    """Isolated process to run vLLM and generate traces."""
    from abyme.vllm_model import LocalVLLMModel
    from data_generation import DataManager
    from abyme.recursive_engine import RecursiveEngine
    import threading

    # 1. Load vLLM (with LoRA if it exists)
    has_adapter = os.path.exists(adapter_dir)
    model = LocalVLLMModel(
        model_path=base_model,
        lora_path=adapter_dir if has_adapter else None
    )

    engine = RecursiveEngine(base_model=model, max_workers=MAX_WORKERS)
    
    # 2. Setup DataManager
    manager = DataManager(
        samples=(0,0,0,0,0), tests=(0,0,0,0,0), iteration=999, group_size=GROUP_SIZE
    )

    # 3. Generate
    from pathlib import Path
    lock = threading.Lock()
    manager.generate_group(problem, engine, Path(output_file), lock)
    
    # Graceful shutdown to ensure GPU memory is released
    model.shutdown()

def _train_model_process(base_model: str, data_file: str, adapter_dir: str):
    """Isolated process to run Unsloth backpropagation."""
    from run_training import run_training
    run_training(
        base_model_path=base_model,
        local_jsonl_path=data_file,
        output_adapter_dir=adapter_dir,
        resume=os.path.exists(adapter_dir) # Tell trainer to resume if adapter exists
    )

def _checkpoint_process(base_model: str, adapter_dir: str, hub_repo_id: str, iteration: int):
    """Isolated process to merge, upload, and run the test set."""
    from run_training import merge_and_upload
    from abyme.vllm_model import LocalVLLMModel
    from data_generation import DataManager
    
    print(f"\n--- REACHED CHECKPOINT {iteration} ---")
    
    # 1. Merge and Upload
    merge_and_upload(base_model, adapter_dir, hub_repo_id)

    # 2. Run Test Set Evaluator
    model = LocalVLLMModel(model_path=hub_repo_id)
    manager = DataManager(
        samples=(0,0,0,0,0), tests=(5,5,5,5,5), iteration=iteration, group_size=GROUP_SIZE
    )
    
    avg_score, _ = manager.score_all(model=model, max_workers=MAX_WORKERS)
    manager.check_scores_by_level()
    model.shutdown()


# ==========================================
# CURRICULUM BUILDER
# ==========================================
def build_curriculum(data_file: str = "data/math_full.jsonl") -> List[Dict[str, Any]]:
    """
    Builds a progressive curriculum.
    Mixes easy to hard problems using shifting probability distributions.
    """
    levels = {1: [], 2: [], 3: [], 4: [], 5: []}
    with open(data_file, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                levels[item["level_num"]].append(item)

    for l in levels.values():
        random.shuffle(l)

    total_problems = sum(len(l) for l in levels.values())
    curriculum = []

    # Phase distributions: (L1, L2, L3, L4, L5) probabilities
    phases = [
        (0.7, 0.2, 0.1, 0.0, 0.0), # Phase 1: Mostly Easy
        (0.3, 0.4, 0.2, 0.1, 0.0), # Phase 2: Early Mid
        (0.1, 0.2, 0.4, 0.2, 0.1), # Phase 3: Mid
        (0.0, 0.1, 0.3, 0.4, 0.2), # Phase 4: Late Mid
        (0.0, 0.0, 0.1, 0.4, 0.5), # Phase 5: Hard
    ]

    phase_length = total_problems // len(phases)

    for step in range(total_problems):
        phase_idx = min(step // phase_length, len(phases) - 1)
        weights = phases[phase_idx]
        
        # Pick a level based on phase weights, falling back if a level is empty
        available_levels = [l for l in range(1, 6) if levels[l]]
        if not available_levels: break
        
        active_weights = [weights[l-1] for l in available_levels]
        # Normalize weights if some levels are empty
        weight_sum = sum(active_weights)
        if weight_sum == 0:
            active_weights = [1.0 / len(available_levels)] * len(available_levels)
        else:
            active_weights = [w / weight_sum for w in active_weights]

        chosen_level = random.choices(available_levels, weights=active_weights, k=1)[0]
        curriculum.append(levels[chosen_level].pop())

    return curriculum


# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    # Force 'spawn' to guarantee zero CUDA context inheritance between processes
    mp.set_start_method('spawn', force=True)

    print("Building Curriculum...")
    curriculum = build_curriculum()
    print(f"Curriculum built with {len(curriculum)} problems.")

    os.makedirs("results/step_data", exist_ok=True)

    for step, problem in enumerate(curriculum):
        print(f"\n{'='*50}")
        print(f"STEP {step+1}/{len(curriculum)} | Level {problem['level_num']} | Subject: {problem['type']}")
        print(f"{'='*50}")

        step_data_file = f"results/step_data/step_{step}.jsonl"

        # ----------------------------------------------------
        # 1. Rollout Process (vLLM)
        # ----------------------------------------------------
        p_gen = mp.Process(target=_generate_data_process, args=(problem, step_data_file, BASE_MODEL, ADAPTER_DIR))
        p_gen.start()
        p_gen.join()

        if p_gen.exitcode != 0:
            print(f"Generation failed at step {step}. Skipping training.")
            continue

        # ----------------------------------------------------
        # 2. Training Process (Unsloth)
        # ----------------------------------------------------
        p_train = mp.Process(target=_train_model_process, args=(BASE_MODEL, step_data_file, ADAPTER_DIR))
        p_train.start()
        p_train.join()

        if p_train.exitcode != 0:
            print(f"Training failed at step {step}.")
            continue

        # ----------------------------------------------------
        # 3. Checkpoint Process
        # ----------------------------------------------------
        if (step + 1) % CHECKPOINT_INTERVAL == 0:
            p_check = mp.Process(target=_checkpoint_process, args=(BASE_MODEL, ADAPTER_DIR, HUB_REPO_ID, step+1))
            p_check.start()
            p_check.join()