'''
We will use a DeepSeek model to generate a question.
We will use the recursive engine to generate a trace.
feed the trace to our model for output and test the format of the output.
'''

from abyme.model import DeepSeekModel
from abyme.core import Abyme_API_Model
from typing import Tuple, List, Dict, Any
from sft.prompts import FIELD_PROMPTS
from abyme.tree_trace import flatten_trace
from abyme.magic import magic_formatter
from sft.load_sft_model import AbymeSFTHuggingFaceModel
from abyme.utils import verify_output_format_strict
import json
import os
from datetime import datetime

deepseek = DeepSeekModel()
teacher = Abyme_API_Model("deepseek",print_progress=True, max_depth = 3, max_call = 20, max_chain_length=3, max_subproblem_retry=1, max_parallel_workers=10)
abyme = AbymeSFTHuggingFaceModel()
DATA_PATH = "../data/test_sft_syntax_data.jsonl"
FAILED_OUTPUT_PATH = "../data/test_sft_failed_outputs.jsonl"
FIELDS = ["Calculus", "Algebra", "Coding", "Physics", "Finance", "Travel", "Event Planning","Competition Math", "Lean"]
NUM_SAMPLES_PER_FIELD = 10

# total data size should be 9*10*20 = 1800

print("Models loaded successfully.")

# Test a single question from a field and return the number of succuessful generations and total generations for that field
def test_single_sample(field:str)->Tuple[List[str],int,int]:
    if field in FIELD_PROMPTS:
        question = deepseek.generate(f"Generate a question that requires extensive reasoning in the field of {FIELD_PROMPTS[field]}.")
    else:
        question = deepseek.generate(f"Generate a question that requires extensive reasoning in the field of {field}.")
    
    print(f"Generated question: {question}")
    print("=="*50)
    
    try:
        teacher.generate(question)
    except Exception as e:
        pass
    
    print("=="*50)
    
    trace_list = flatten_trace(teacher.trace)
    
    print(f"trace length: {len(trace_list)}")
    
    successful_generations = 0
    total_generations = 0
    failed_output = []
    
    for trace in trace_list:
        total_generations += 1
        output = abyme.generate(magic_formatter(trace[0], trace[1], trace[2], trace[3]))
        print("=="*50)
        print(f"prompt: {magic_formatter(trace[0], trace[1], trace[2], trace[3])}")
        print(f"output: {output}")
        print(f"counter: {total_generations}")
        if verify_output_format_strict(output):
            successful_generations += 1
        else:
            failed_output.append(output)
    
    return failed_output, successful_generations, total_generations

def generate_and_save_data():
    """
    Generate training data for all fields and append to DATA_PATH.
    Each result is saved incrementally as a complete list entry.
    Outer loop: num_samples, Inner loop: fields
    """
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    print(f"\n{'='*50}")
    print("GENERATING DATA FOR ALL FIELDS")
    print(f"{'='*50}\n")

    for sample_idx in range(NUM_SAMPLES_PER_FIELD):
        print(f"\n{'='*50}")
        print(f"Sample iteration {sample_idx + 1}/{NUM_SAMPLES_PER_FIELD}")
        print(f"{'='*50}\n")

        for field in FIELDS:
            print(f"\n{'-'*50}")
            print(f"Generating sample {sample_idx + 1}/{NUM_SAMPLES_PER_FIELD} for field: {field}")
            print(f"{'-'*50}\n")

            # Generate question
            if field in FIELD_PROMPTS:
                question = deepseek.generate(f"Generate a question that requires extensive reasoning in the field of {FIELD_PROMPTS[field]}.")
            else:
                question = deepseek.generate(f"Generate a question that requires extensive reasoning in the field of {field}.")

            print(f"Generated question: {question}")
            print("=="*50)

            # Generate trace using teacher model
            try:
                teacher.generate(question)
            except Exception as e:
                print(f"Teacher model error: {e}")

            print("=="*50)

            trace_list = flatten_trace(teacher.trace)
            print(f"Trace length: {len(trace_list)}")

            # Prepare data entry
            data_entry = {
                "field": field,
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "trace_length": len(trace_list),
                "sample_index": sample_idx,
                "traces": []
            }

            # Process each trace
            for trace_idx, trace in enumerate(trace_list):
                trace_data = {
                    "trace_index": trace_idx,
                    "parent_text": trace[0],
                    "parent_output": trace[1],
                    "child_text": trace[2],
                    "child_output": trace[3],
                    "prompt": magic_formatter(trace[0], trace[1], trace[2], trace[3])
                }
                data_entry["traces"].append(trace_data)

            # Append to file as a complete list entry
            with open(DATA_PATH, 'a') as f:
                f.write(json.dumps(data_entry) + '\n')

            print(f"Saved data entry for {field} (sample {sample_idx + 1}) to {DATA_PATH}")

    print(f"\n{'='*50}")
    print("DATA GENERATION COMPLETE")
    print(f"Data saved to: {DATA_PATH}")
    print(f"{'='*50}\n")

def run_sft_tests_on_all_data():
    """
    Run the SFT model on all tests from the data file.
    Save failed outputs with full trace information.
    """
    if not os.path.exists(DATA_PATH):
        print(f"No data file found at {DATA_PATH}")
        return

    os.makedirs(os.path.dirname(FAILED_OUTPUT_PATH), exist_ok=True)

    successful_generations = 0
    total_generations = 0
    failed_count = 0

    with open(DATA_PATH, 'r') as f:
        for line_idx, line in enumerate(f):
            data_entry = json.loads(line.strip())

            print(f"\n{'='*50}")
            print(f"Processing entry {line_idx + 1}: {data_entry['field']}")
            print(f"Question: {data_entry['question']}")
            print(f"{'='*50}\n")

            # Test each trace
            for trace_data in data_entry["traces"]:
                total_generations += 1
                prompt = trace_data["prompt"]

                # Generate output using SFT model
                output = abyme.generate(prompt)

                print(f"Trace {trace_data['trace_index'] + 1}/{len(data_entry['traces'])}")
                print(f"Output: {output}")

                # Verify output format
                is_valid = verify_output_format_strict(output,allow_extra_content=True)

                if is_valid:
                    successful_generations += 1
                    print("✓ Valid format")
                else:
                    failed_count += 1
                    print("✗ Invalid format")

                    # Save failed output with full trace information
                    failed_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "field": data_entry["field"],
                        "question": data_entry["question"],
                        "trace_index": trace_data["trace_index"],
                        "parent_text": trace_data["parent_text"],
                        "parent_output": trace_data["parent_output"],
                        "child_text": trace_data["child_text"],
                        "child_output": trace_data["child_output"],
                        "prompt": prompt,
                        "model_output": output,
                        "error_type": "invalid_format"
                    }

                    with open(FAILED_OUTPUT_PATH, 'a') as failed_f:
                        failed_f.write(json.dumps(failed_entry) + '\n')
                
                print(f"Total generations so far: {total_generations}")
                print(f"Successful generations so far: {successful_generations}")
                print("=="*50)

    # Print summary
    print(f"\n{'='*50}")
    print("TESTING SUMMARY")
    print(f"{'='*50}")
    print(f"Total generations: {total_generations}")
    print(f"Successful generations: {successful_generations}")
    print(f"Failed generations: {failed_count}")
    print(f"Success rate: {successful_generations/total_generations*100:.2f}%")
    print(f"Failed outputs saved to: {FAILED_OUTPUT_PATH}")
    print(f"{'='*50}\n")

def main():
    """
    Legacy main function - runs full pipeline with test_single_sample.
    """
    fields = FIELDS
    successful_generations = 0
    total_generations = 0
    failed_output = []
    for _ in range(NUM_SAMPLES_PER_FIELD):
        for f in fields:
            failed, s ,t = test_single_sample(f)
            failed_output.extend(failed)
            successful_generations += s
            total_generations += t
    print("=="*50)
    print("Failed outputs:")
    for output in failed_output:
        print(output)
    print("=="*50)
    print(f"Overall Successful generations: {successful_generations}")
    print(f"Overall Total generations: {total_generations}")

if __name__ == "__main__":
    
    #generate_and_save_data()
    
    run_sft_tests_on_all_data()

'''
================================================================================
FINAL ANALYSIS: Failed Output Format Validation
================================================================================

OVERALL STATISTICS:
------------------
• Total generations tested: 900
• Fails: 13

Success Rate: 98.56%

FAILURE BREAKDOWN (13 remaining failures):
------------------------------------------

1. TRUNCATION FAILURES: 8 cases (61.5%) ⚠️ PRIMARY ISSUE
   Root Cause: OUTPUT TOKEN LIMIT reached during generation
   
   Evidence:
   - More opening tags than closing tags (e.g., 8 <do> but 7 </do>)
   - Text ends mid-sentence without punctuation
   - Average output length: ~7,000 characters
   
   Affected Cases:
   #0:  Event Planning   (11,417 chars, missing 1 </do>)
   #3:  Finance          (6,143 chars, missing 1 </do>)
   #4:  Finance          (6,542 chars, missing 1 </do>)
   #5:  Finance          (6,679 chars, missing 1 </do>)
   #8:  Competition Math (5,305 chars, missing 1 </do>)
   #9:  Competition Math (4,226 chars, missing 1 </try>)
   #10: Algebra          (5,226 chars, missing 1 </try>)
   #11: Event Planning   (10,439 chars, missing 1 </do>)

2. FORMAT ISSUES: 3 cases (23.1%)
   Root Cause: Missing '>' description lines after ## DO/TRY headers
   
   Details:
   #2: Calculus       - 9 headers, 9 tags, 0 complete pattern matches
   #6: Event Planning - 10 headers, 10 tags, 4 complete pattern matches  
   #7: Event Planning - 12 headers, 11 tags, 11 complete pattern matches

3. NO DELEGATION TAGS: 2 cases (15.4%)
   Root Cause: Model generated reasoning text without delegation structure
   
   Details:
   #1:  Algebra (9,024 chars)
   #12: Coding  (2,996 chars)

================================================================================
'''
