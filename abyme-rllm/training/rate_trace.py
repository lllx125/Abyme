from pathlib import Path
from abyme.tree_trace import TreeTraceNode, fold, length, dict_to_node, future_length, collect_all_nodes
from typing import Tuple, Callable, Dict, Any, List, Optional
from abyme.utils import verify_output_format_strict
from abyme.magic import magic_formatter
from collections import defaultdict
import json

GOOD = True
BAD = False

def _rate_all(input_path: Path, output_path: Path, score_function: Callable[[str, Dict[str, Any]], float], verbose: bool = True):
    """
    Rate all nodes from generated traces and output labeled training data.

    Reads trace results from input_path JSONL, extracts all nodes, computes future_length
    for each node, and labels based on future_length percentiles.

    Labeling Strategy:
    1. For each question, label all nodes of the trace with correct final answer and minimum length as GOOD
    2. Label all nodes with incorrect format, parent of FAIL node as BAD
    3. Label failed/wrong answer trace level 0 as BAD
   
   
    Args:
        input_path: Path to JSONL file with trace results (from ParallelTreeOrchestrator)
        output_path: Path to output JSONL file with labeled nodes
        score_function: Function to score final answers (e.g., MATH500Benchmark.score)
        verbose: If True, print label counts per category after rating.
    """
    # Dict key in the question index

    correct_traces: Dict[int, List[TreeTraceNode]] = defaultdict(list)
    
    incorrect_traces: Dict[int, List[TreeTraceNode]] = defaultdict(list) # but not failed
    
    failed_traces: Dict[int, List[TreeTraceNode]] = defaultdict(list)
    
    all_traces: Dict[int, List[TreeTraceNode]] = defaultdict(list)
    
    # good nodes
    good_nodes: List[TreeTraceNode] = []
    # bad nodes
    bad_nodes: List[TreeTraceNode] = []
    
    with input_path.open('r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line.strip())
            if 'trace_tree' not in record:
                continue

            trace = dict_to_node(record['trace_tree'])

            # Score at the trace level using the pre-extracted final output.
            # Individual nodes cannot determine answer correctness on their own.
            try:
                is_correct = score_function(record['output'], record) == 1.0
            except Exception:
                is_correct = False
            
            index = record.get("index", -1)
            
            if record.get("status") == "FAILED":
                failed_traces[index].append(trace)
            elif is_correct:
                correct_traces[index].append(trace)
            else:
                incorrect_traces[index].append(trace)
            all_traces[index].append(trace)
    
    # 1 Label GOOD nodes from correct traces with minimum length
    for index in correct_traces:
        if not correct_traces[index]:
            continue
        min_length_trace = correct_traces.index(min(correct_traces[index], key=lambda t: length(t)))
        good_nodes.extend(collect_all_nodes(min_length_trace))
    
    good_count = len(good_nodes)
    
    bad_format_count = 0
    bad_failed_count = 0
    
    # 2 Label BAD nodes with incorrect format, parent of FAIL node
    for index in all_traces:
        for trace in all_traces[index]:
            nodes = collect_all_nodes(trace)
            for node in nodes:
                if not node.output:
                    continue
                if not verify_output_format_strict(node.output):
                    bad_nodes.append(node)
                    bad_format_count += 1
                if node.status == "FAILED" and node.parent:
                    bad_nodes.append(node.parent)
                    bad_failed_count += 1

    # 3 Label BAD nodes at depth 0 of failed traces
    bad_failed_depth0_count = 0
    for index in failed_traces:
        for trace in failed_traces[index]:
            bad_nodes.append(trace)
            bad_nodes.extend(trace.past)
            bad_failed_depth0_count += len(trace.past) + 1
    
    # 3 Label BAD nodes at depth 0 of incorrect traces
    bad_incorrect_depth0_count = 0
    for index in incorrect_traces:
        for trace in incorrect_traces[index]:
            bad_nodes.append(trace)
            bad_nodes.extend(trace.past)
            bad_incorrect_depth0_count += len(trace.past) + 1

    if verbose:
        print(f"\n{'='*50}")
        print(f"Rate All: {input_path.name}")
        print(f"{'='*50}")
        print(f"  Total nodes          : {sum(len(collect_all_nodes(t)) for traces in all_traces.values() for t in traces)}")
        print(f"  --- GOOD ---")
        print(f"  Minimum length trace : {good_count}")
        print(f"  --- BAD ---")
        print(f"  Bad format           : {bad_format_count}")
        print(f"  Parent of FAIL       : {bad_failed_count}")
        print(f"  Failed depth-0       : {bad_failed_depth0_count}")
        print(f"  Incorrect depth-0    : {bad_incorrect_depth0_count}")
        print(f"  BAD total            : {len(bad_nodes)}")
        print(f"{'='*50}\n")
        
    # --- Write labeled nodes to output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        for node in good_nodes:
            record = {
                "input": magic_formatter(node.prompt, node.main_problem, node.parent_problem, node.fragment),
                "output": node.output,
                "label": GOOD,
            }
            f.write(json.dumps(record) + '\n')
        for node in bad_nodes:
            record = {
                "input": magic_formatter(node.prompt, node.main_problem, node.parent_problem, node.fragment),
                "output": node.output,
                "label": BAD,
            }
            f.write(json.dumps(record) + '\n')



def rate_all_deprecated(input_path: Path, output_path: Path, score_function: Callable[[str, Dict[str, Any]], float], verbose: bool = True):
    """
    Rate all nodes from generated traces and output labeled training data.

    Reads trace results from input_path JSONL, extracts all nodes, computes future_length
    for each node, and labels based on future_length percentiles.

    Labeling Strategy:
    1. Compute future_length for all nodes (see compute_future_length documentation)
    2. Filter nodes from traces with correct answers
    3. Sort filtered nodes by future_length
    4. Label lowest 25% (of total nodes) from correct traces as GOOD (most efficient correct nodes)
    5. Label all nodes with incorrect format as BAD
    6. Label all final node outputs that are incorrect as BAD
    7. Label all Failed node as BAD
    8. Label all past (level 0) nodes as BAD if the final answer is incorrect (if exceeds 25%, randomly drop some to maintain balance)
    9. Label the node with highest future_length of all nodes as BAD until we get 25% of total nodes

    Args:
        input_path: Path to JSONL file with trace results (from ParallelTreeOrchestrator)
        output_path: Path to output JSONL file with labeled nodes
        score_function: Function to score final answers (e.g., MATH500Benchmark.score)
        verbose: If True, print label counts per category after rating.
    """
    import random

    # Nodes from traces with correct final answers
    correct_node: Dict[int, List[TreeTraceNode]] = defaultdict(list)
    # all nodes
    all_node: Dict[int, List[TreeTraceNode]] = defaultdict(list)
    
    # good nodes
    good_nodes: List[TreeTraceNode] = []
    # bad nodes
    bad_nodes: List[TreeTraceNode] = []

    with input_path.open('r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line.strip())
            if 'trace_tree' not in record:
                continue

            trace = dict_to_node(record['trace_tree'])

            # Score at the trace level using the pre-extracted final output.
            # Individual nodes cannot determine answer correctness on their own.
            try:
                is_correct = score_function(record['output'], record) == 1.0
            except Exception:
                is_correct = False

            nodes = collect_all_nodes(trace)
            if is_correct:
                correct_node[record["index"]].extend(nodes)
            all_node[record["index"]].extend(nodes)

    total_nodes = sum(len(nodes) for nodes in all_node.values())
    if total_nodes == 0:
        return

    target_count = total_nodes // 8  # 12.5% each for GOOD and BAD

    # Step 2-4: GOOD — all nodes from correct traces, sorted by future_length, label lowest 12.5%
    good_per_index = target_count // len(correct_node) if correct_node else 0
    for _ , nodes in correct_node.items():
        nodes.sort(key=lambda n: future_length(n))
        if len(nodes) <= good_per_index:
            good_nodes.extend(nodes)
        else:
            good_nodes.extend(nodes[:good_per_index])
    good_count = len(good_nodes)
    
    target_count = good_count  # To maintain balance, we want equal GOOD and BAD counts

    # Steps 5-8: BAD — explicit bad signals
    bad_count = 0
    bad_format_count = 0
    bad_final_wrong_count = 0
    bad_failed_count = 0
    bad_depth0_count = 0
    bad_high_future_length_count = 0
    
    # GLobal pass to label explicit BAD nodes based on format, final answer correctness, and failures
    for nodes in all_node.values():
        for node in nodes:
            if node in good_nodes:
                continue  # Already labeled as GOOD
            
            if not node.output:
                continue

            # Bad format
            if not verify_output_format_strict(node.output):
                bad_nodes.append(node)
                bad_format_count += 1
                bad_count += 1
                continue

            # Final node with incorrect answer
            if node.type == "FINAL" and node not in correct_node:
                bad_nodes.append(node)
                bad_final_wrong_count += 1
                bad_count += 1
                continue

            # Failed node
            if node.status == "FAILED":
                bad_nodes.append(node)
                bad_failed_count += 1
                bad_count += 1
                continue
    
    # Step 8-9: BAD — depth-0 nodes with incorrect final answer + high future_length nodes until we reach target_count
    if bad_count < target_count:
        bad_depth0_per_index = target_count - bad_count // len(all_node) if all_node else 0
        for nodes in all_node.values():
            bad_depth0_count_this_index = 0
            for node in nodes:
                if node in good_nodes or node in bad_nodes:
                    continue  # Already labeled

                if node.depth == 0 and node not in correct_node:
                    bad_nodes.append(node)
                    bad_depth0_count += 1
                    bad_depth0_count_this_index += 1
                    bad_count += 1
                    continue
                if bad_depth0_count_this_index >= bad_depth0_per_index:
                    break  # Stop labeling depth-0 nodes for this index if we've reached the per-index target
        
    if bad_count < target_count:
        bad_high_future_length_count_per_index = (target_count - bad_count) // len(all_node) if all_node else 0
        for nodes in all_node.values():
            high_future_length_nodes = [n for n in nodes if n not in good_nodes and n not in bad_nodes]
            high_future_length_nodes.sort(key=lambda n: future_length(n), reverse=True)
            bad_high_future_length_count_this_index = 0
            for node in high_future_length_nodes:
                if bad_count >= target_count:
                    break
                if bad_high_future_length_count_this_index >= bad_high_future_length_count_per_index:
                    break
                bad_nodes.append(node)
                bad_high_future_length_count_this_index += 1
                bad_high_future_length_count += 1
                bad_count += 1

    if verbose:
        unlabeled_count = total_nodes - good_count - bad_count
        print(f"\n{'='*50}")
        print(f"Rate All: {input_path.name}")
        print(f"{'='*50}")
        print(f"  Total nodes          : {total_nodes}")
        print(f"  Target per class     : {target_count} (12.5%)")
        print(f"  --- GOOD ---")
        print(f"  Low future_length    : {good_count}")
        print(f"  --- BAD ---")
        print(f"  Bad format           : {bad_format_count}")
        print(f"  Final wrong answer   : {bad_final_wrong_count}")
        print(f"  Failed node          : {bad_failed_count}")
        print(f"  Depth-0 incorrect    : {bad_depth0_count}")
        print(f"  High future_length   : {bad_high_future_length_count}")
        print(f"  BAD total            : {bad_count}")
        print(f"  --- Unlabeled (skipped) ---")
        print(f"  Unlabeled            : {unlabeled_count}")
        print(f"{'='*50}\n")

    # --- Write labeled nodes to output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        for node in good_nodes:
            record = {
                "input": magic_formatter(node.prompt, node.main_problem, node.parent_problem, node.fragment),
                "output": node.output,
                "label": GOOD,
            }
            f.write(json.dumps(record) + '\n')
        for node in bad_nodes:
            record = {
                "input": magic_formatter(node.prompt, node.main_problem, node.parent_problem, node.fragment),
                "output": node.output,
                "label": BAD,
            }
            f.write(json.dumps(record) + '\n')