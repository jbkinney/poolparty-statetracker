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


# def print_tree(
#     root: Any,
#     get_label: Callable[[Any], str],
#     get_children: Callable[[Any], list],
# ) -> None:
#     """Print ASCII tree for any graph structure.
    
#     Args:
#         root: The root node of the tree.
#         get_label: Function that returns the display label for a node.
#         get_children: Function that returns the children (parents in DAG) of a node.
#     """
#     lines = build_tree_lines(root, get_label, get_children)
#     for line in lines:
#         print(line)


def format_node(counter, style: StyleType = 'clean') -> str:
    """Format a Counter node for display.
    
    Args:
        counter: The Counter object to format.
        style: Display style - 'clean', 'minimal', or 'repr'.
    
    Returns:
        Formatted string representation of the counter.
    """
    if style == 'repr':
        return repr(counter)
    
    name = counter.name if counter.name else f"id_{counter.id}"
    n = counter.num_states
    
    if style == 'minimal':
        return name
    
    # style == 'clean'
    if counter._parents:
        op_name = type(counter._op).__name__
        if op_name.endswith('Op'):
            op_name = op_name[:-2].lower()
        return f"{name} [{op_name}, n={n}]"
    else:
        return f"{name} [leaf, n={n}]"


def print_tree(counter, style: StyleType = 'clean') -> None:
    """Print ASCII tree for a single Counter and its ancestors.
    
    Args:
        counter: The root Counter to visualize.
        style: Display style - 'clean', 'minimal', or 'repr'.
    """
    def get_label(c):
        return format_node(c, style)
    
    def get_children(c):
        return list(c._parents)
    
    #print_tree(counter, get_label, get_children)
    
    lines = build_tree_lines(counter, get_label, get_children)
    for line in lines:
        print(line)


def print_graph(counters: list, style: StyleType = 'clean') -> None:
    """Print ASCII tree visualization of Counter dependency graphs.
    
    Finds root counters (those not used as parents by others) and prints
    each tree.
    
    Args:
        counters: List of all Counter objects to consider.
        style: Display style - 'clean', 'minimal', or 'repr'.
    """
    if not counters:
        print("(no counters registered)")
        return
    
    # Find counters that are parents of other counters
    parent_ids = set()
    for counter in counters:
        for parent in counter._parents:
            parent_ids.add(parent._id)
    
    # Root counters are those not used as parents
    roots = [c for c in counters if c._id not in parent_ids]
    
    if not roots:
        print("(no counters registered)")
        return
    
    for i, root in enumerate(roots):
        print_tree(root, style=style)
        if i < len(roots) - 1:
            print()
