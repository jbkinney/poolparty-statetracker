"""Generic tree building utilities for text-based visualization."""
from typing import Callable, Any, Literal
StyleType = Literal['clean', 'minimal', 'repr']

def build_tree_lines(
    root: Any,
    get_label: Callable[[Any], str],
    get_children: Callable[[Any], list],
) -> list[str]:
    """Build ASCII tree lines for any graph structure.
    
    Args:
        root: The root node of the tree.
        get_label: Function that returns the display label for a node.
        get_children: Function that returns the children (parents in DAG) of a node.
    
    Returns:
        List of strings, one per line of the tree visualization.
    """
    lines: list[str] = []
    
    def _build_subtree(node: Any, prefix: str, is_last: bool, is_root: bool) -> None:
        if is_root:
            connector = ""
            child_prefix = ""
        else:
            connector = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "
        
        lines.append(f"{prefix}{connector}{get_label(node)}")
        
        children = get_children(node)
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            _build_subtree(child, prefix + child_prefix, is_last_child, is_root=False)
    
    _build_subtree(root, "", is_last=True, is_root=True)
    return lines


def format_state_node(state, style: StyleType = 'clean') -> str:
    """Format a State node for display.
    
    Args:
        state: The State object to format.
        style: Display style - 'clean', 'minimal', or 'repr'.
    
    Returns:
        Formatted string representation of the state.
    """
    n = state.num_values
    order = state.iter_order
    
    if style == 'repr':
        return repr(state)
    
    name = state.name if state.name else f"id_{state.id}"
    
    if style == 'minimal':
        return f"{name} (n={n})"
    
    # style == 'clean'
    return f"{name} (state, io={order}, n={n})"


def format_operation_node(op, style: StyleType = 'clean') -> str:
    """Format an Operation node for display.
    
    Args:
        op: The Operation object to format.
        style: Display style - 'clean', 'minimal', or 'repr'.
    
    Returns:
        Formatted string representation of the operation.
    """
    op_name = type(op).__name__
    if op_name.endswith('Op'):
        op_name = op_name[:-2]
    
    if style == 'repr':
        return repr(op)
    elif style == 'minimal':
        return f"[{op_name}]"
    else:  # clean
        return f"[op={op_name}]"


def print_dag(state, style: StyleType = 'clean') -> None:
    """Print ASCII tree for a single State and its ancestors.
    
    The tree alternates between State and Operation nodes:
    - State -> its Operation -> Operation's parent States -> ...
    
    Args:
        state: The root State to visualize.
        style: Display style - 'clean', 'minimal', or 'repr'.
    """
    # We need to handle the alternating State/Operation structure.
    # We'll wrap nodes to track their type.
    
    class StateNode:
        def __init__(self, state):
            self.state = state
    
    class OpNode:
        def __init__(self, op, parent_states):
            self.op = op
            self.parent_states = parent_states
    
    def get_label(node):
        if isinstance(node, StateNode):
            return format_state_node(node.state, style)
        else:
            return format_operation_node(node.op, style)
    
    def get_children(node):
        if isinstance(node, StateNode):
            # State's child is its operation (if non-leaf)
            if node.state._op:
                return [OpNode(node.state._op, node.state._parents)]
            else:
                return []
        else:
            # Operation's children are its parent states
            return [StateNode(s) for s in node.parent_states]
    
    lines = build_tree_lines(StateNode(state), get_label, get_children)
    for line in lines:
        print(line)


def print_graph(states: list, style: StyleType = 'clean') -> None:
    """Print ASCII tree visualization of State dependency graphs.
    
    Finds root states (those not used as parents by others) and prints
    each tree.
    
    Args:
        states: List of all State objects to consider.
        style: Display style - 'clean', 'minimal', or 'repr'.
    """
    if not states:
        print("(no states registered)")
        return
    
    # Find states that are parents of other states
    parent_ids = set()
    for state in states:
        for parent in state._parents:
            parent_ids.add(parent._id)
    
    # Root states are those not used as parents
    roots = [s for s in states if s._id not in parent_ids]
    
    if not roots:
        print("(no states registered)")
        return
    
    for i, root in enumerate(roots):
        print_dag(root, style=style)
        if i < len(roots) - 1:
            print()
