"""Generic tree building utilities for text-based visualization."""
from typing import Callable, Any


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


def print_tree(
    root: Any,
    get_label: Callable[[Any], str],
    get_children: Callable[[Any], list],
) -> None:
    """Print ASCII tree for any graph structure.
    
    Args:
        root: The root node of the tree.
        get_label: Function that returns the display label for a node.
        get_children: Function that returns the children (parents in DAG) of a node.
    """
    lines = build_tree_lines(root, get_label, get_children)
    for line in lines:
        print(line)

