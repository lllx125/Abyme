from pathlib import Path
from abyme.tree_trace import TreeTraceNode, fold, length
from typing import Tuple, Callable, Dict, Any, List, Optional
from abyme.utils import verify_output_format_strict
from abyme.magic import magic_formatter
import json

GOOD = True
BAD = False

def _dict_to_node(node_dict: Dict[str, Any], parent: Optional[TreeTraceNode] = None) -> TreeTraceNode:
    """
    Reconstruct a TreeTraceNode from a dictionary (inverse of to_dict).

    Args:
        node_dict: Dictionary representation of a node
        parent: Parent node reference (used during recursion)

    Returns:
        Reconstructed TreeTraceNode
    """
    # Reconstruct past nodes first (temporal history)
    past_nodes = [_dict_to_node(p_dict, parent=None) for p_dict in node_dict.get("past", [])]

    # Create the current node
    node = TreeTraceNode(
        prompt=node_dict["prompt"],
        fragment=node_dict["fragment"],
        parent=parent,
        past=past_nodes,
        index=node_dict.get("index", 0)
    )

    # Set all node properties
    node.output = node_dict.get("output", "")
    node.type = node_dict.get("type", "leaf")
    node.status = node_dict.get("status", "WAIT_GEN")
    node.difficulty = node_dict.get("difficulty", 1)
    node.depth = node_dict.get("depth", 0)
    node.latency = node_dict.get("latency", 0.0)
    node.error_message = node_dict.get("error_message", "")
    node.is_cancelled = node_dict.get("is_cancelled", False)

    # Reconstruct subproblems (spatial children)
    for sub_dict in node_dict.get("subproblems", []):
        sub_node = _dict_to_node(sub_dict, parent=node)
        node.subproblems.append(sub_node)

    return node

def compute_future_length(node: TreeTraceNode) -> float:
    """
    Compute the future length of a node.

    future_length = length from this node to completion - length of immediate past (0 if no past)

    This measures how much "new work" this node contributes to reaching the final answer.
    Lower future_length means the node is closer to completion and more efficient.

    Args:
        node: The TreeTraceNode to compute future_length for

    Returns:
        The future_length value (can be negative if this node is shorter than its past)
    """
    # Length from this node to the end
    node_to_end_length = length(node)

    # Length of the immediate past (most recent past state)
    if node.past:
        # The last element in past is the most recent past state
        past_length = length(node.past[-1])
    else:
        past_length = 0.0

    future_length = node_to_end_length - past_length

    return future_length

def rate_all(input_path: Path, output_path: Path, score_function: Callable[[str, Dict[str, Any]], float]):
    """
    Rate all nodes from generated traces and output labeled training data.

    Reads trace results from input_path JSONL, extracts all nodes, computes future_length
    for each node, and labels based on future_length percentiles.

    Labeling Strategy:
    1. Compute future_length for all nodes (see compute_future_length documentation)
    2. Filter nodes from traces with correct answers
    3. Sort filtered nodes by future_length
    4. Label lowest 25% (of total nodes) from correct traces as GOOD (most efficient correct nodes)
    5. Label highest 25% (of all nodes) as BAD (least efficient nodes)
    6. Remaining nodes are unlabeled and filtered out

    Args:
        input_path: Path to JSONL file with trace results (from ParallelTreeOrchestrator)
        output_path: Path to output JSONL file with labeled nodes
        score_function: Function to score final answers (e.g., MATH500Benchmark.score)
    """
    # Read all traces from input
    traces_data = []
    with input_path.open('r') as f:
        for line in f:
            if line.strip():
                traces_data.append(json.loads(line.strip()))

    # Extract all nodes with their future_length and correctness info
    all_nodes_with_metrics = []
    correct_nodes_with_metrics = []

    for trace_data in traces_data:
        # Skip if no trace tree
        trace_dict = trace_data.get('trace_tree')
        if not trace_dict:
            continue

        # Reconstruct the trace
        trace_node = _dict_to_node(trace_dict)

        # Check if this trace has a correct answer
        input_data = {
            'problem': trace_data.get('original_problem') or trace_data.get('prompt'),
            'answer': trace_data.get('ground_truth', '')
        }

        is_correct = False
        try:
            final_output = trace_node.get_final_output()
            answer_score = score_function(final_output, input_data)
            is_correct = (answer_score == 1.0)
        except Exception:
            is_correct = False

        # Extract all nodes from the trace
        def extract_nodes(node: TreeTraceNode, past_results: List, sub_results: List) -> List[Dict[str, Any]]:
            """Fold function to extract all nodes with their context and metrics."""
            nodes = []

            # Collect nodes from past
            for past_node_list in past_results:
                nodes.extend(past_node_list)

            # Only process nodes with output
            if node.output:
                # Get main problem (root prompt)
                root = node
                while root.parent is not None:
                    root = root.parent
                main_problem = root.prompt

                # Get parent problem
                parent_problem = node.parent.prompt if node.parent else "None"

                # Create formatted input using magic_formatter
                formatted_input = magic_formatter(
                    prompt=node.prompt,
                    main_problem=main_problem,
                    boss_problem=parent_problem,
                    fragment=node.fragment
                )

                # Compute future_length
                future_len = compute_future_length(node)

                nodes.append({
                    "input": formatted_input,
                    "output": node.output,
                    "future_length": future_len,
                    "is_correct_trace": is_correct
                })

            # Collect nodes from subproblems
            for sub_node_list in sub_results:
                nodes.extend(sub_node_list)

            return nodes

        trace_nodes = fold(trace_node, extract_nodes)
        all_nodes_with_metrics.extend(trace_nodes)

        # Also collect nodes from correct traces separately
        if is_correct:
            correct_nodes_with_metrics.extend(trace_nodes)

    # Sort all nodes by future_length for BAD labeling
    all_nodes_with_metrics.sort(key=lambda x: x["future_length"])

    # Sort correct nodes by future_length for GOOD labeling
    correct_nodes_with_metrics.sort(key=lambda x: x["future_length"])

    # Compute quartile counts
    total_nodes = len(all_nodes_with_metrics)
    quarter_count = total_nodes // 4

    # Label nodes
    labeled_nodes = []

    # Label GOOD: lowest 25% future_length from correct traces
    good_candidates = correct_nodes_with_metrics[:quarter_count]
    for node_data in good_candidates:
        labeled_nodes.append({
            "input": node_data["input"],
            "output": node_data["output"],
            "label": GOOD
        })

    # Label BAD: highest 25% future_length from all nodes
    bad_candidates = all_nodes_with_metrics[-quarter_count:]
    for node_data in bad_candidates:
        labeled_nodes.append({
            "input": node_data["input"],
            "output": node_data["output"],
            "label": BAD
        })

    # Write labeled nodes to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        for node_data in labeled_nodes:
            f.write(json.dumps(node_data) + '\n')

    print(f"Processed {total_nodes} nodes from {len(traces_data)} traces")
    print(f"  Correct trace nodes: {len(correct_nodes_with_metrics)}")
    print(f"  Labeled {len(labeled_nodes)} nodes")
    print(f"  GOOD nodes (lowest 25% future_length from correct traces): {sum(1 for n in labeled_nodes if n['label'] == GOOD)}")
    print(f"  BAD nodes (highest 25% future_length from all nodes): {sum(1 for n in labeled_nodes if n['label'] == BAD)}")
    if total_nodes > 0:
        print(f"  All nodes future_length range: [{all_nodes_with_metrics[0]['future_length']:.2f}, {all_nodes_with_metrics[-1]['future_length']:.2f}]")
    if len(correct_nodes_with_metrics) > 0:
        print(f"  Correct nodes future_length range: [{correct_nodes_with_metrics[0]['future_length']:.2f}, {correct_nodes_with_metrics[-1]['future_length']:.2f}]")

def rate(trace: TreeTraceNode, score_function: Callable[[str, Dict[str, Any]], float], input_data: Dict[str, Any]) -> Tuple[bool, bool, float]:
    """
    Rate a tree trace by checking format correctness, answer correctness, and trace length.

    Args:
        trace: The TreeTraceNode to rate
        score_function: Function to score the final answer (e.g., MATH500Benchmark.score)
        input_data: The input data dictionary containing problem and ground truth answer

    Returns:
        Tuple of (all_output_formatted_correctly, answer_is_correct, trace_length)
    """
    # Check if all outputs along the trace are formatted correctly using fold
    def check_format(node: TreeTraceNode, past_results: list, sub_results: list) -> bool:
        """Check if this node and all descendants have correctly formatted output."""
        # Check current node's output format
        if node.output:
            current_valid = verify_output_format_strict(node.output, print_reason=False)
        else:
            current_valid = True  # Empty output is considered valid

        # All must be valid: current node, all past states, and all subproblems
        return current_valid and all(past_results) and all(sub_results)

    all_formatted_correctly = fold(trace, check_format)

    # Get the final output and check if the answer is correct
    try:
        final_output = trace.get_final_output()
        answer_score = score_function(final_output, input_data)
        answer_correct = answer_score == 1.0
    except Exception:
        # If we can't get final output or scoring fails, answer is incorrect
        answer_correct = False

    # Get the trace length
    trace_length = length(trace)

    return all_formatted_correctly, answer_correct, trace_length
    