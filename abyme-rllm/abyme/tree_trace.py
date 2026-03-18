"""
Tree Trace Module
Handles the state, hierarchy, history, and metrics of the generation tree.
"""
from typing import List, Optional, Literal, Dict, Tuple, Callable, Any
from .utils import format_output, AND, OR, LEAF

NodeStatus = Literal["WAIT_GEN", "GENERATING", "WAIT_SUB", "COMPLETED", "FINAL", "FAILED", "CANCELLED"]


class TreeTraceNode:
    """
    Class representing a single node in the tree trace of a recursive generation process.
    Tracks generation state, output, errors, temporal history (past), and spatial children (subproblems).
    """

    def __init__(self, prompt: str, fragment: str, parent: Optional["TreeTraceNode"] = None, past: Optional[List["TreeTraceNode"]] = None, index: int = 0):
        self.prompt: str = prompt
        self.fragment: str = fragment
        self.parent: Optional["TreeTraceNode"] = parent
        self.depth: int = parent.depth + 1 if parent else 0
        self.index: int = index
        self.past: List["TreeTraceNode"] = past if past is not None else []
        self.subproblems: List["TreeTraceNode"] = []
        
        self.status: NodeStatus = "WAIT_GEN"
        self.type: str = LEAF  # AND, OR, or LEAF
        self.difficulty: int = 1
        self.is_cancelled: bool = False
        
        self.output: str = ""
        self.error_message: str = ""
        self.latency: float = 0.0

    def record_generation(self, output: str, latency: float):
        """Record the successful base model output and latency."""
        self.output = output
        self.latency = latency

    def add_subproblems(self, subproblem_prompts: List[str]):
        """Create child nodes for the current node based on extracted prompts."""
        for i, prompt in enumerate(subproblem_prompts):
            new_node = TreeTraceNode(prompt=prompt, fragment="", parent=self, index=i)
            self.subproblems.append(new_node)
        self.update_difficulty()

    def update_difficulty(self):
        """
        Recursively update the difficulty for Greedy DFS scheduling.
        LEAF = 1, AND = sum of children, OR = min of children.
        """
        if self.type == LEAF:
            self.difficulty = 1
        else:
            diffs = [sub.difficulty for sub in self.subproblems]
            if not diffs:
                self.difficulty = 1
            elif self.type == AND:
                self.difficulty = sum(diffs)
            elif self.type == OR:
                self.difficulty = min(diffs)
        
        # Propagate changes up to the root
        if self.parent:
            self.parent.update_difficulty()

    def cancel_tree(self):
        """Mark this node and all its non-final descendants as cancelled (used by OR node kill switch)."""
        self.is_cancelled = True
        self.status = "CANCELLED"
        for sub in self.subproblems:
            if sub.status not in ["FINAL", "COMPLETED"]:
                sub.cancel_tree()

    def continue_generation(self, new_fragment: str) -> "TreeTraceNode":
        """
        Create a temporal continuation of this node with an updated fragment.
        Appends the current state to the past history and replaces itself in the parent's tree.
        """
        # CRITICAL: Clear self.past before packing it to prevent duplicate subgraph 
        # traversal when using the fold() utility later.
        old_past = self.past
        self.past = [] 
        
        new_node = TreeTraceNode(
            prompt=self.prompt,
            fragment=new_fragment,
            parent=self.parent,
            past=old_past + [self],  # The new node holds the flat history
            index=self.index
        )
        
        # Replace the old node in the parent's subproblems list
        if self.parent and self.index < len(self.parent.subproblems):
            self.parent.subproblems[self.index] = new_node
            
        return new_node

    def get_final_output(self) -> str:
        """Get the cleaned and formatted final output."""
        if self.status == "FINAL":
            return format_output(self.output)
        raise ValueError(f"Output is not finalized yet. Current status: {self.status}")


# ==========================================
# EXTERNAL TREE UTILITY FUNCTIONS
# ==========================================

def fold(node: TreeTraceNode, func: Callable[[TreeTraceNode, List[Any], List[Any]], Any]) -> Any:
    """
    A universal tree traversal function (catamorphism) that processes both 
    historical states (`past`) and hierarchical states (`subproblems`).
    
    Args:
        node: The starting TreeTraceNode.
        func: A callable receiving (current_node, past_results_list, subproblem_results_list)
    """
    past_res = [fold(p, func) for p in node.past]
    sub_res = [fold(s, func) for s in node.subproblems]
    return func(node, past_res, sub_res)


def flatten_trace(node: TreeTraceNode) -> List[Tuple[str, str, str, str, str]]:
    """Returns a list of (prompt, fragment, parent_problem, main_problem, output) for all generations."""
    # Retrieve the main problem from the root
    curr = node
    while curr.parent is not None:
        curr = curr.parent
    main_problem = curr.prompt

    def agg(n: TreeTraceNode, p_res: List[List], s_res: List[List]) -> List:
        res = []
        for pr in p_res: res.extend(pr)
        
        parent_prompt = n.parent.prompt if n.parent else "None"
        res.append((n.prompt, n.fragment, parent_prompt, main_problem, n.output))
        
        for sr in s_res: res.extend(sr)
        return res
        
    return fold(node, agg)


def parallel_latency(node: TreeTraceNode) -> float:
    """Calculates the minimum execution latency assuming infinite concurrent workers."""
    def agg(n: TreeTraceNode, p_res: List[float], s_res: List[float]) -> float:
        # Time spent sequentially getting to this point (past) + generation time
        node_time = sum(p_res) + n.latency 
        
        # Time spent waiting on parallel children
        if not s_res:
            sub_time = 0.0
        elif n.type == AND:
            sub_time = max(s_res)  # Must wait for all AND children
        elif n.type == OR:
            sub_time = min(s_res)  # Only wait for the fastest OR child
        else:
            sub_time = 0.0
            
        return node_time + sub_time
        
    return fold(node, agg)


def sequencial_latency(node: TreeTraceNode) -> float:
    """Calculates the execution latency assuming only a single worker processing nodes sequentially."""
    def agg(n: TreeTraceNode, p_res: List[float], s_res: List[float]) -> float:
        return n.latency + sum(p_res) + sum(s_res)
    return fold(node, agg)


def total_calls(node: TreeTraceNode) -> int:
    """Returns the total number of LLM API calls made across the entire tree and history."""
    def agg(n: TreeTraceNode, p_res: List[int], s_res: List[int]) -> int:
        call = 1 if n.output else 0
        return call + sum(p_res) + sum(s_res)
    return fold(node, agg)


def max_depth(node: TreeTraceNode) -> int:
    """Returns the maximum depth reached in the generation tree."""
    def agg(n: TreeTraceNode, p_res: List[int], s_res: List[int]) -> int:
        return max([n.depth] + p_res + s_res)
    return fold(node, agg)


def max_subproblems(node: TreeTraceNode) -> int:
    """Returns the maximum number of simultaneous subproblems any node had."""
    def agg(n: TreeTraceNode, p_res: List[int], s_res: List[int]) -> int:
        return max([len(n.subproblems)] + p_res + s_res)
    return fold(node, agg)


def max_output_character(node: TreeTraceNode) -> int:
    """Returns the character length of the longest base-model output in the tree."""
    def agg(n: TreeTraceNode, p_res: List[int], s_res: List[int]) -> int:
        return max([len(n.output)] + p_res + s_res)
    return fold(node, agg)


def nodes_per_level(node: TreeTraceNode) -> Dict[int, int]:
    """Analyzes the total number of node states generated at each depth level."""
    def agg(n: TreeTraceNode, p_res: List[Dict[int, int]], s_res: List[Dict[int, int]]) -> Dict[int, int]:
        counts = {n.depth: 1}
        for res_dict in p_res + s_res:
            for depth, count in res_dict.items():
                counts[depth] = counts.get(depth, 0) + count
        return counts
    return fold(node, agg)

def draw_tree(
    node: TreeTraceNode, 
    right_spacing: float, 
    down_spacing: float, 
    node_radius: float
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, Tuple[float, float], Tuple[float, float]]]]:
    """
    Generates a 2D layout for the TreeTraceNode graph.
    - Past nodes: Drawn sequentially to the left.
    - Subproblems: First drawn directly below, subsequent ones to the right.
    
    Returns:
        nodes: A list of JSON-serializable dictionaries with node properties and (x, y).
        edges: A list of tuples (edge_type, (x1, y1), (x2, y2)).
    """

    def agg(
        n: TreeTraceNode, 
        p_res: List[Tuple], 
        s_res: List[Tuple]
    ) -> Tuple[List[Dict], List[Tuple], float, float, float, float]:
        """
        Inner aggregator function for `fold`.
        Returns: (nodes, edges, min_x, max_x, root_x, root_y)
        """
        nodes = []
        edges = []
        
        # 1. Base initialization for the current node (local origin at 0.0, 0.0)
        curr_dict = {
            "prompt": n.prompt,
            "fragment": n.fragment,
            "output": n.output,
            "type": n.type,
            "status": n.status,
            "difficulty": n.difficulty,
            "x": 0.0,
            "y": 0.0
        }
        nodes.append(curr_dict)
        
        # Bounding box of the current node alone
        min_x = -node_radius
        max_x = node_radius
        
        # 2. Process Subproblems (Spatial hierarchy downwards and rightwards)
        if s_res:
            # Place the first subproblem directly below the current node
            first_s_nodes, first_s_edges, s_min, s_max, s_rx, s_ry = s_res[0]
            
            # Shift amounts for the first subproblem
            dx = 0.0 - s_rx
            dy = down_spacing - s_ry
            
            # Apply shifts
            for nd in first_s_nodes:
                nd["x"] += dx
                nd["y"] += dy
            for i in range(len(first_s_edges)):
                etype, p1, p2 = first_s_edges[i]
                first_s_edges[i] = (etype, (p1[0]+dx, p1[1]+dy), (p2[0]+dx, p2[1]+dy))
                
            nodes.extend(first_s_nodes)
            edges.extend(first_s_edges)
            edges.append((n.type, (0.0, 0.0), (0.0, down_spacing))) # Edge from parent to first sub
            
            # Update parent bounding box
            min_x = min(min_x, s_min + dx)
            max_x = max(max_x, s_max + dx)
            prev_max_x = s_max + dx
            
            # Place remaining subproblems sequentially to the right
            for i in range(1, len(s_res)):
                s_nodes, s_edges, s_min_i, s_max_i, s_rx_i, s_ry_i = s_res[i]
                
                # Align the left edge of this subproblem to the right edge of the previous one
                dx_i = prev_max_x + right_spacing - s_min_i
                dy_i = down_spacing - s_ry_i
                
                # Apply shifts
                for nd in s_nodes:
                    nd["x"] += dx_i
                    nd["y"] += dy_i
                for j in range(len(s_edges)):
                    etype, p1, p2 = s_edges[j]
                    s_edges[j] = (etype, (p1[0]+dx_i, p1[1]+dy_i), (p2[0]+dx_i, p2[1]+dy_i))
                    
                nodes.extend(s_nodes)
                edges.extend(s_edges)
                
                # Edge from parent to this subproblem's root
                edges.append((n.type, (0.0, 0.0), (s_rx_i + dx_i, down_spacing)))
                
                # Update parent bounding box
                min_x = min(min_x, s_min_i + dx_i)
                max_x = max(max_x, s_max_i + dx_i)
                prev_max_x = s_max_i + dx_i

        # 3. Process Past nodes (Temporal history extending to the left)
        if p_res:
            # Start placing to the left of the current bounding box
            target_right = min_x - right_spacing
            
            # The coordinate we are pointing TO (initially the current node)
            next_rx, next_ry = 0.0, 0.0  
            
            # Iterate backwards (from most recent past to oldest past)
            for i in range(len(p_res)-1, -1, -1):
                p_nodes, p_edges, p_min, p_max, p_rx, p_ry = p_res[i]
                
                # Align the right edge of this past node to our target left edge
                dx_p = target_right - p_max
                dy_p = 0.0 - p_ry
                
                # Apply shifts
                for nd in p_nodes:
                    nd["x"] += dx_p
                    nd["y"] += dy_p
                for j in range(len(p_edges)):
                    etype, p1, p2 = p_edges[j]
                    p_edges[j] = (etype, (p1[0]+dx_p, p1[1]+dy_p), (p2[0]+dx_p, p2[1]+dy_p))
                    
                nodes.extend(p_nodes)
                edges.extend(p_edges)
                
                curr_rx_shifted = p_rx + dx_p
                curr_ry_shifted = p_ry + dy_p
                
                # Arrow pointing from this past node to the chronological next node
                edges.append(("PAST", (curr_rx_shifted, curr_ry_shifted), (next_rx, next_ry)))
                
                # Expand bounding box
                min_x = min(min_x, p_min + dx_p)
                max_x = max(max_x, p_max + dx_p)
                
                # Update targets for the previous chronological node
                target_right = p_min + dx_p - right_spacing
                next_rx, next_ry = curr_rx_shifted, curr_ry_shifted

        return nodes, edges, min_x, max_x, 0.0, 0.0

    # Execute the bottom-up fold
    final_nodes, final_edges, _, _, _, _ = fold(node, agg)
    
    return final_nodes, final_edges