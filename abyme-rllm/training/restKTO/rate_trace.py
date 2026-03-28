from pathlib import Path
from abyme.tree_trace import TreeTraceNode, length, dict_to_node, future_length, collect_all_nodes
from abyme.utils import AND, OR
from typing import Callable, Dict, Any, List
from abyme.utils import verify_output_format_strict, get_format_error
from abyme.magic import magic_formatter
from collections import defaultdict
import json

GOOD = True
BAD = False

def rate_all(input_path: Path, output_path: Path, score_function: Callable[[str, Dict[str, Any]], float], verbose: bool = True):
    """
    Rate all nodes from generated traces and output labeled training data.

    Reads trace results from input_path JSONL, extracts all nodes, computes future_length
    for each node, and labels based on future_length percentiles.

    Labeling Strategy:
    1. For each question, label all nodes of the trace with correct final answer and minimum length as GOOD (2 per unique problem)
    2. Label all nodes with incorrect format, parent of FAIL node as BAD
    3. Label failed/wrong answer trace level 0 as BAD
    4. Add AND/OR nodes with lower future_length from each correct trace as GOOD and BAD nodes balances
   
   
    Args:
        input_path: Path to JSONL file with trace results (from ParallelTreeOrchestrator)
        output_path: Path to output JSONL file with labeled nodes
        score_function: Function to score final answers (e.g., MATH500Benchmark.score)
        verbose: If True, print label counts per category after rating.
    """
    # Dict key is the problem prompt string
    correct_traces: Dict[str, List[TreeTraceNode]] = defaultdict(list)

    incorrect_traces: Dict[str, List[TreeTraceNode]] = defaultdict(list) # but not failed

    failed_traces: Dict[str, List[TreeTraceNode]] = defaultdict(list)

    all_traces: Dict[str, List[TreeTraceNode]] = defaultdict(list)

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

            prompt = record.get("prompt", "")

            if record.get("status") == "FAILED":
                failed_traces[prompt].append(trace)
            elif is_correct:
                correct_traces[prompt].append(trace)
            else:
                incorrect_traces[prompt].append(trace)
            all_traces[prompt].append(trace)

    # 1 Label GOOD nodes from correct traces with minimum length (2 per unique problem)
    for prompt in correct_traces:
        if not correct_traces[prompt]:
            continue
        sorted_traces = sorted(correct_traces[prompt], key=lambda t: length(t))
        top_traces = sorted_traces[:2] if len(sorted_traces) >= 2 else sorted_traces[:1]
        for trace in top_traces:
            good_nodes.extend(
                node for node in collect_all_nodes(trace)
                if node.output and node.status not in ("FAILED", "CANCELLED")
                and verify_output_format_strict(node.output)
            )
    
    good_count = len(good_nodes)
    
    bad_format_count = 0
    bad_failed_count = 0
    bad_format_reasons: Dict[str, int] = defaultdict(int)

    # 2 Label BAD nodes with incorrect format, parent of FAIL node
    for prompt in all_traces:
        for trace in all_traces[prompt]:
            nodes = collect_all_nodes(trace)
            for node in nodes:
                if node.output:
                    err = get_format_error(node.output)
                    if err is not None:
                        bad_nodes.append(node)
                        bad_format_count += 1
                        bad_format_reasons[err] += 1
                # if node.status == "FAILED":
                #     # Prefer the spatial parent; fall back to last temporal past node
                #     culprit = node.parent if node.parent else (node.past[-1] if node.past else None)
                #     if culprit and culprit.output:
                #         bad_nodes.append(culprit)
                #         bad_failed_count += 1

    # 3 Label BAD nodes at depth 0 of failed traces
    bad_failed_depth0_count = 0
    for prompt in failed_traces:
        for trace in failed_traces[prompt]:
            nodes_to_add = [n for n in [trace] + list(trace.past) if n.output]
            bad_nodes.extend(nodes_to_add)
            bad_failed_depth0_count += len(nodes_to_add)

    # 3 Label BAD nodes at depth 0 of incorrect traces
    bad_incorrect_depth0_count = 0
    for prompt in incorrect_traces:
        for trace in incorrect_traces[prompt]:
            nodes_to_add = [n for n in [trace] + list(trace.past) if n.output]
            bad_nodes.extend(nodes_to_add)
            bad_incorrect_depth0_count += len(nodes_to_add)

    # 4 Add AND/OR nodes (lowest future_length) from correct traces as GOOD until balanced
    # Sample uniformly per prompt: each prompt contributes at most its fair share.
    and_or_good_count = 0
    deficit = len(bad_nodes) - len(good_nodes)
    if deficit > 0:
        prompts_with_correct = [p for p in correct_traces if correct_traces[p]]
        per_prompt = deficit // len(prompts_with_correct) if prompts_with_correct else 0
        remainder = deficit - per_prompt * len(prompts_with_correct)
        for i, prompt in enumerate(prompts_with_correct):
            quota = per_prompt + (1 if i < remainder else 0)
            candidates = [
                node
                for trace in correct_traces[prompt]
                for node in collect_all_nodes(trace)
                if node.type in (AND, OR) and node.output and node.status not in ("FAILED", "CANCELLED")
                and verify_output_format_strict(node.output)
            ]
            candidates.sort(key=lambda n: future_length(n))
            to_add = candidates[:quota]
            good_nodes.extend(to_add)
            and_or_good_count += len(to_add)

    if verbose:
        from notifier import mailman
        sep = '=' * 50
        total_nodes = sum(len(collect_all_nodes(t)) for traces in all_traces.values() for t in traces)
        format_breakdown = "\n".join(
            f"    {reason:<35}: {count}"
            for reason, count in sorted(bad_format_reasons.items(), key=lambda x: -x[1])
        )
        msg = "\n".join([
            sep,
            f"Rate All: {input_path.name}",
            sep,
            f"  Total nodes          : {total_nodes}",
            f"  --- GOOD ---",
            f"  Minimum length trace : {good_count}",
            f"  AND/OR low future_L  : {and_or_good_count}",
            f"  GOOD total           : {len(good_nodes)}",
            f"  --- BAD ---",
            f"  Bad format           : {bad_format_count}",
            format_breakdown,
            f"  Parent of FAIL       : {bad_failed_count}",
            f"  Failed depth-0       : {bad_failed_depth0_count}",
            f"  Incorrect depth-0    : {bad_incorrect_depth0_count}",
            f"  BAD total            : {len(bad_nodes)}",
            sep,
        ])
        mailman.send(msg)

        
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