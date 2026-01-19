"""State-specific tree visualization utilities."""
from poolparty.types import Literal
from .tree import print_dag

StyleType = Literal['clean', 'minimal', 'repr']


def format_counter_node(state, style: StyleType = 'clean') -> str:
    """Format a State node for display.
    
    Args:
        state: The State object to format.
        style: Display style - 'clean', 'minimal', or 'repr'.
    
    Returns:
        Formatted string representation of the state.
    """
    if style == 'repr':
        return repr(state)
    
    name = state.name if state.name else f"id_{state.id}"
    n = state.num_values
    
    if style == 'minimal':
        return name
    
    # style == 'clean'
    if state._parents:
        op_name = type(state._op).__name__
        if op_name.endswith('Op'):
            op_name = op_name[:-2]
        return f"{name} [{op_name}, n={n}]"
    else:
        return f"{name} [State, n={n}]"


def print_counter_tree(state, style: StyleType = 'clean') -> None:
    """Print ASCII tree for a single State and its ancestors.
    
    Args:
        state: The root State to visualize.
        style: Display style - 'clean', 'minimal', or 'repr'.
    """
    def get_label(s):
        return format_counter_node(s, style)
    
    def get_children(s):
        return list(s._parents)
    
    print_dag(state, get_label, get_children)

