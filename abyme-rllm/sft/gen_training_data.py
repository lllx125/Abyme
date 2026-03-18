import os, sys, json
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

MAX_WORKERS = 50
DATA_PATH = "data/sft_training_data.jsonl"
MAX_CHAR = 13000  # Maximum combined character length for main_task + boss_task + prompt + fragment + output
TARGET_FIELD = ["Calculus","Algebra","Coding","Physics","Lean", "Competition Math", "Travel", "Finance", "Event Planning"] # "Calculus", "Algebra", "Coding", "Physics", "Lean", "Competition Math", "Travel", "Finance", "Event Planning"
TARGET_HIERARCHY = ["main","sub","sub-sub"] # "main","sub" or "sub-sub"
TARGET_ACTION = ["ANSWER","AND","OR"]
TARGET_STATE = ["first","continue","fail"]
NUM_SAMPLES_NEEDED = 25

# 1. Get the absolute path to the directory containing 'main.py'
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the path to the parent directory (project_root)
parent_dir = os.path.dirname(current_dir)

# 3. Add the parent directory to sys.path
sys.path.append(parent_dir)

from abyme.model import DeepSeekModel
from abyme.utils import verify_format
from prompts import *
import time

def clean_and_parse_llm_json(llm_output: str) -> Dict[str, str]:
    """Cleans markdown artifacts and parses the JSON output."""
    if llm_output.startswith("```json"):
        llm_output = llm_output.replace("```json", "", 1)
    if llm_output.startswith("```"):
        llm_output = llm_output.replace("```", "", 1)
    if llm_output.endswith("```"):
        llm_output = llm_output.rsplit("```", 1)[0]
        
    llm_output = llm_output.replace('\xa0', ' ').strip()
    return json.loads(llm_output)


def generate_single_sample(
    field: str, 
    problem_type: str, 
    answer_type: str, 
    continue_type: str, 
    planner_model: DeepSeekModel, 
    fragment_model: DeepSeekModel, 
    output_model: DeepSeekModel
) -> Optional[Dict[str, str]]:
    
    # ==========================================
    # STEP 1: PLANNER
    # ==========================================
    planner_prompt = f"""
# FIELD
{FIELD_PROMPTS[field]}
# HIERARCHY
{HIERARCHY_PROMPTS[problem_type]}
# STATE
{STATE_PROMPTS[continue_type]}
# ACTION
{ACTION_PROMPTS[answer_type]}

Generate the JSON object now.
"""
    planner_res = planner_model.generate(planner_prompt)
    
    try:
        plan_data = clean_and_parse_llm_json(planner_res)
    except json.JSONDecodeError as e:
        print(f"[-] Step 1 Parsing Error: {e}\nRaw Output: {planner_res}")
        return None

    # ==========================================
    # STEP 2: FRAGMENT WRITER
    # ==========================================
    fragment_text = None
    if continue_type != "first" and plan_data.get("fragment_plan"):
        fragment_prompt = f"""
# FRAGMENT PLAN
{plan_data["fragment_plan"]}

{FORMAT_INSTRUCTIONS_FRAGMENT[continue_type]}

Write the exact fragment text now. Do NOT wrap in JSON or code blocks.
"""
        fragment_text = fragment_model.generate(fragment_prompt).strip()


    # ==========================================
    # STEP 3: OUTPUT WRITER
    # ==========================================
    output_prompt = f"""
# CONTEXT
Main Task: {plan_data.get('main_task')}
Boss Task: {plan_data.get('boss_task')}
Prompt (Your Task): {plan_data.get('prompt')}

# PREVIOUS FRAGMENT
{fragment_text if fragment_text else "None (Fresh start)"}

# OUTPUT PLAN
{plan_data.get('output_plan')}

{FORMAT_INSTRUCTIONS_OUTPUT[answer_type]}

Write the exact output text now. Do NOT wrap in JSON or code blocks.
"""
    output_text = output_model.generate(output_prompt).strip()

    # ==========================================
    # ASSEMBLE FINAL SCHEMA
    # ==========================================
    final_sample = {
        "field": field,
        "hierarchy": problem_type,
        "action": answer_type,
        "state": continue_type,
        "prompt": plan_data.get("prompt"),
        "main_task": plan_data.get("main_task"),
        "boss_task": plan_data.get("boss_task"),
        "fragment": fragment_text,
        "output": output_text
    }

    return final_sample


def main():
    start_time = time.time()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # Instantiate the 3 distinct roles (you can use the same model class, but distinct system prompts)
    print("Loading models...")
    planner_model = DeepSeekModel(reasoning=True, system_prompt=PLANNER_SYSTEM_PROMPT, temperature=1.3)
    fragment_model = DeepSeekModel(reasoning=True, system_prompt=FRAGMENT_SYSTEM_PROMPT)
    output_model = DeepSeekModel(reasoning=True, system_prompt=OUTPUT_SYSTEM_PROMPT)
    print("Models loaded successfully.")
    # ----------------------------------------------

    # Validate that all values are valid keys in their respective dictionaries
    print("Validating parameters...")
    validation_errors = []

    for field in TARGET_FIELD:
        if field not in FIELD_PROMPTS:
            validation_errors.append(f"Invalid FIELD: '{field}' not found in FIELD_PROMPTS")

    for hierarchy in TARGET_HIERARCHY:
        if hierarchy not in HIERARCHY_PROMPTS:
            validation_errors.append(f"Invalid HIERARCHY: '{hierarchy}' not found in HIERARCHY_PROMPTS")

    for action in TARGET_ACTION:
        if action not in ACTION_PROMPTS:
            validation_errors.append(f"Invalid ACTION: '{action}' not found in ACTION_PROMPTS")

    for state in TARGET_STATE:
        if state not in STATE_PROMPTS:
            validation_errors.append(f"Invalid STATE: '{state}' not found in STATE_PROMPTS")

    if validation_errors:
        print("\n❌ VALIDATION FAILED:")
        for error in validation_errors:
            print(f"  - {error}")
        print("\nAvailable keys:")
        print(f"  FIELD_PROMPTS: {list(FIELD_PROMPTS.keys())}")
        print(f"  HIERARCHY_PROMPTS: {list(HIERARCHY_PROMPTS.keys())}")
        print(f"  ACTION_PROMPTS: {list(ACTION_PROMPTS.keys())}")
        print(f"  STATE_PROMPTS: {list(STATE_PROMPTS.keys())}")
        print(f"  FORMAT_INSTRUCTIONS_FRAGMENT: {list(FORMAT_INSTRUCTIONS_FRAGMENT.keys())}")
        print(f"  FORMAT_INSTRUCTIONS_OUTPUT: {list(FORMAT_INSTRUCTIONS_OUTPUT.keys())}")
        return

    print("✓ All parameters validated successfully!\n")

    # Iterate through all combinations
    all_combinations = list(product(TARGET_FIELD, TARGET_HIERARCHY, TARGET_ACTION, TARGET_STATE))
    total_combinations = len(all_combinations)

    print(f"\nTotal combinations to process: {total_combinations}")
    print(f"Samples needed per combination: {NUM_SAMPLES_NEEDED}")
    print(f"Using {MAX_WORKERS} workers for multithreaded generation across all combinations\n")

    # Track successful samples per combination
    successful_samples_per_combo = {combo: 0 for combo in all_combinations}
    attempts_per_combo = {combo: 0 for combo in all_combinations}
    overall_attempts = 0

    # Worker function that includes combination parameters
    def worker(combo, attempt_num):
        """Worker function that generates a single sample and returns (combo, attempt_num, result, latency)"""
        field, hierarchy, action, state = combo
        print(f"\n{'='*50}")
        print(f"Attempt {attempt_num} | Combo: ({field}, {hierarchy}, {action}, {state})")
        print(f"{'='*50}")
        start = time.time()
        result = generate_single_sample(
            field=field,
            problem_type=hierarchy,
            answer_type=action,
            continue_type=state,
            planner_model=planner_model,
            fragment_model=fragment_model,
            output_model=output_model
        )
        latency = time.time() - start
        return (combo, attempt_num, result, latency)

    # Use ThreadPoolExecutor to parallelize across all combinations
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit initial batch of jobs distributed across combinations
        futures = {}
        total_samples_needed = total_combinations * NUM_SAMPLES_NEEDED

        # Round-robin initial job submission across combinations
        combo_idx = 0
        for _ in range(min(MAX_WORKERS, total_samples_needed)):
            combo = all_combinations[combo_idx % total_combinations]
            if successful_samples_per_combo[combo] < NUM_SAMPLES_NEEDED:
                attempts_per_combo[combo] += 1
                overall_attempts += 1
                future = executor.submit(worker, combo, attempts_per_combo[combo])
                futures[future] = combo
            combo_idx += 1

        # Process completed jobs and submit new ones as needed
        total_successful = sum(successful_samples_per_combo.values())
        while total_successful < total_samples_needed:
            # Wait for at least one job to complete
            for future in as_completed(futures):
                combo, attempt_num, result, latency = future.result()
                field, hierarchy, action, state = combo

                print(f"Attempt {attempt_num} for ({field}, {hierarchy}, {action}, {state}) - latency: {latency:.2f}s")

                # Track if this attempt failed
                attempt_failed = False

                if result:
                    # Validate output is not empty
                    output_valid = result['output'] and result['output'].strip()

                    # Validate fragment is not empty if state is not "first"
                    fragment_valid = True
                    if state != "first":
                        fragment_valid = result['fragment'] and result['fragment'].strip()

                    # Validate format
                    format_valid = verify_format(result['output']) and verify_format(result['fragment'])

                    # Validate character length
                    main_len = len(result.get('main_task') or "")
                    boss_len = len(result.get('boss_task') or "")
                    prompt_len = len(result.get('prompt') or "")
                    fragment_len = len(result.get('fragment') or "")
                    output_len = len(result.get('output') or "")
                    total_len = main_len + boss_len + prompt_len + fragment_len + output_len
                    length_valid = total_len <= MAX_CHAR

                    if output_valid and fragment_valid and format_valid and length_valid:

                        # Append JSON object to JSONL file immediately
                        with open(DATA_PATH, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')

                        successful_samples_per_combo[combo] += 1
                        total_successful += 1

                        print(f"\n[+] Sample for ({field}, {hierarchy}, {action}, {state}) saved! "
                              f"({successful_samples_per_combo[combo]}/{NUM_SAMPLES_NEEDED} for this combo, "
                              f"{total_successful}/{total_samples_needed} total)")

                        # Print the outputs so you can inspect them live
                        print("\n--- FINAL SAMPLE PREVIEW ---")
                        print(f"Sample {total_successful}/{total_samples_needed}")
                        print(f"Parameters: Field={field}, Hierarchy={hierarchy}, Action={action}, State={state}")
                        print(f"Character Count: {total_len}/{MAX_CHAR} chars")
                        print(f"Prompt: {result['prompt']}")
                        print(f"Main Task: {result['main_task']}")
                        print(f"Boss Task: {result['boss_task']}")
                        print(f"Fragment:\n{result['fragment']}")
                        print(f"Output:\n{result['output']}")
                        print("----------------------------\n")
                    else:
                        validation_reasons = []
                        if not output_valid:
                            validation_reasons.append("output is empty")
                        if not fragment_valid:
                            validation_reasons.append("fragment is empty (required for non-first state)")
                        if not format_valid:
                            validation_reasons.append("format check failed")
                        if not length_valid:
                            validation_reasons.append(f"sample too long ({total_len} > {MAX_CHAR} chars)")
                        print(f"\n[-] Attempt {attempt_num} for ({field}, {hierarchy}, {action}, {state}) failed validation: {', '.join(validation_reasons)}")
                        attempt_failed = True
                else:
                    print(f"\n[-] Attempt {attempt_num} for ({field}, {hierarchy}, {action}, {state}) failed generation.")
                    attempt_failed = True

                # Remove completed future
                del futures[future]

                # Count in-flight jobs per combo
                in_flight_per_combo = {}
                for f in futures.values():
                    in_flight_per_combo[f] = in_flight_per_combo.get(f, 0) + 1

                # Submit replacement job if this attempt failed and this combo still needs samples
                if attempt_failed:
                    in_flight_for_this_combo = in_flight_per_combo.get(combo, 0)
                    if successful_samples_per_combo[combo] + in_flight_for_this_combo < NUM_SAMPLES_NEEDED:
                        attempts_per_combo[combo] += 1
                        overall_attempts += 1
                        new_future = executor.submit(worker, combo, attempts_per_combo[combo])
                        futures[new_future] = combo
                # If this succeeded, find another combo that needs samples
                elif total_successful < total_samples_needed:
                    # Find a combination that still needs samples (accounting for in-flight jobs)
                    for next_combo in all_combinations:
                        in_flight_for_next = in_flight_per_combo.get(next_combo, 0)
                        if successful_samples_per_combo[next_combo] + in_flight_for_next < NUM_SAMPLES_NEEDED:
                            attempts_per_combo[next_combo] += 1
                            overall_attempts += 1
                            new_future = executor.submit(worker, next_combo, attempts_per_combo[next_combo])
                            futures[new_future] = next_combo
                            break

                # Break out of as_completed loop to check if we're done
                break

    # Print summary per combination
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ ALL COMBINATIONS COMPLETE!")
    print(f"{'='*60}")
    for combo_idx, combo in enumerate(all_combinations, 1):
        field, hierarchy, action, state = combo
        print(f"Combo {combo_idx}: ({field}, {hierarchy}, {action}, {state}) - "
              f"{successful_samples_per_combo[combo]}/{NUM_SAMPLES_NEEDED} samples, "
              f"{attempts_per_combo[combo]} attempts")
    print(f"\nTotal combinations processed: {total_combinations}")
    print(f"Total samples generated: {total_successful}")
    print(f"Total attempts: {overall_attempts}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Saved to {DATA_PATH}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()