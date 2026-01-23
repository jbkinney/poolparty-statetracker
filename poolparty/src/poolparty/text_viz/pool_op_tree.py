"""Pool and Operation tree visualization utilities."""
from poolparty.types import Literal
from .tree import print_dag

StyleType = Literal['clean', 'minimal', 'repr']


def format_pool_node(pool, style: StyleType = 'clean') -> str:
    """Format a Pool node for display.
    
    Args:
        pool: The Pool object to format.
        style: Display style - 'clean', 'minimal', or 'repr'.
    
    Returns:
        Formatted string representation of the pool.
    """
    if style == 'repr':
        return repr(pool)
    elif style == 'minimal':
        return f"{pool.name} (pool)"
    else:  # clean
        num_states_str = "None" if pool.num_states is None else str(pool.num_states)
        return f"{pool.name} (pool, n={num_states_str})"


def format_operation_node(op, style: StyleType = 'clean') -> str:
    """Format an Operation node for display.
    
    Args:
        op: The Operation object to format.
        style: Display style - 'clean', 'minimal', or 'repr'.
    
    Returns:
        Formatted string representation of the operation.
    """
    if style == 'repr':
        return repr(op)
    elif style == 'minimal':
        return f"{op.name} [op]"
    else:  # clean
        num_values_str = "None" if op.num_values is None else str(op.num_values)
        return f"{op.name} [mode={op.mode}, n={num_values_str}]"


def print_pool_tree(pool, style: StyleType = 'clean', show_pools: bool = True) -> None:
    """Print ASCII tree for a single Pool and its upstream DAG."""
    
    class PoolNode:
        def __init__(self, pool):
            self.pool = pool
    
    class OpNode:
        def __init__(self, op):
            self.op = op
    
    def get_label(node):
        if isinstance(node, PoolNode):
            return format_pool_node(node.pool, style)
        else:
            return format_operation_node(node.op, style)
    
    def get_children(node):
        if show_pools:
            # Alternating Pool/Op structure
            if isinstance(node, PoolNode):
                return [OpNode(node.pool.operation)]
            else:
                return [PoolNode(p) for p in node.op.parent_pools]
        else:
            # Ops only - skip pools, link ops directly
            return [OpNode(p.operation) for p in node.op.parent_pools]
    
    root = PoolNode(pool) if show_pools else OpNode(pool.operation)
    print_dag(root, get_label, get_children)


def print_operation_tree(op, style: StyleType = 'clean') -> None:
    """Print ASCII tree for a single Operation and its upstream DAG.
    
    The tree alternates between Operation and Pool nodes:
    - Operation -> parent Pools -> their Operations -> ...
    
    Args:
        op: The root Operation to visualize.
        style: Display style - 'clean', 'minimal', or 'repr'.
    """
    class PoolNode:
        def __init__(self, pool):
            self.pool = pool
    
    class OpNode:
        def __init__(self, op):
            self.op = op
    
    def get_label(node):
        if isinstance(node, PoolNode):
            return format_pool_node(node.pool, style)
        else:
            return format_operation_node(node.op, style)
    
    def get_children(node):
        if isinstance(node, PoolNode):
            # Pool's child is its operation
            return [OpNode(node.pool.operation)]
        else:
            # Operation's children are its parent pools
            return [PoolNode(p) for p in node.op.parent_pools]
    
    print_dag(OpNode(op), get_label, get_children)

