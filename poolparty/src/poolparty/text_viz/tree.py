"""Generic tree building utilities for text-based visualization."""

from poolparty.types import Any, Callable
from statetracker.text_viz import build_tree_lines

def print_dag(
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
