"""Pool and Operation tree visualization utilities."""
from poolparty.types import Literal
from .tree import print_tree

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
        return f"{pool.name} (pool, n={pool.num_states})"


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
        return f"{op.name} [mode={op.mode}, n={op.num_states}]"


def print_pool_tree(pool, style: StyleType = 'clean') -> None:
    """Print ASCII tree for a single Pool and its upstream DAG.
    
    The tree alternates between Pool and Operation nodes:
    - Pool -> its Operation -> Operation's parent Pools -> ...
    
    Args:
        pool: The root Pool to visualize.
        style: Display style - 'clean', 'minimal', or 'repr'.
    """
    # We need to handle the alternating Pool/Operation structure.
    # We'll wrap nodes to track their type.
    
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
    
    print_tree(PoolNode(pool), get_label, get_children)


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
    
    print_tree(OpNode(op), get_label, get_children)

