"""High-level graph visualization functions."""
from poolparty.types import Literal
from .counter_tree import print_counter_tree
from .pool_op_tree import print_pool_tree

StyleType = Literal['clean', 'minimal', 'repr']


def print_counter_graph(counters: list, style: StyleType = 'clean') -> None:
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

