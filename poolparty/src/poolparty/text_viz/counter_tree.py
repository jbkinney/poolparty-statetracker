"""Counter-specific tree visualization utilities."""
from poolparty.types import Literal
from .tree import print_dag

StyleType = Literal['clean', 'minimal', 'repr']


def format_counter_node(counter, style: StyleType = 'clean') -> str:
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
            op_name = op_name[:-2]
        return f"{name} [{op_name}, n={n}]"
    else:
        return f"{name} [Leaf, n={n}]"


def print_counter_tree(counter, style: StyleType = 'clean') -> None:
    """Print ASCII tree for a single Counter and its ancestors.
    
    Args:
        counter: The root Counter to visualize.
        style: Display style - 'clean', 'minimal', or 'repr'.
    """
    def get_label(c):
        return format_counter_node(c, style)
    
    def get_children(c):
        return list(c._parents)
    
    print_dag(counter, get_label, get_children)

