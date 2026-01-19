"""High-level graph visualization functions."""
from poolparty.types import Literal
from .state_tree import print_counter_tree
from .pool_op_tree import print_pool_tree

StyleType = Literal['clean', 'minimal', 'repr']


def print_counter_graph(states: list, style: StyleType = 'clean') -> None:
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
        print_counter_tree(root, style=style)
        if i < len(roots) - 1:
            print()


def print_pool_graph(pools: list, ops: list, style: StyleType = 'clean') -> None:
    """Print ASCII tree visualization of Pool/Operation computation graphs.
    
    Finds root pools (those not consumed by any operation) and prints
    each tree.
    
    Args:
        pools: List of all Pool objects.
        ops: List of all Operation objects.
        style: Display style - 'clean', 'minimal', or 'repr'.
    """
    if not pools:
        print("(no pools registered)")
        return
    
    # Find pools that are parents of some operation
    parent_pool_ids: set[int] = set()
    for op in ops:
        for parent in op.parent_pools:
            parent_pool_ids.add(parent._id)
    
    # Root pools are those not used as parents
    roots = [p for p in pools if p._id not in parent_pool_ids]
    
    if not roots:
        print("(no pools registered)")
        return
    
    for i, root in enumerate(roots):
        print_pool_tree(root, style=style)
        if i < len(roots) - 1:
            print()

