"""Stack operation - combine pools sequentially (disjoint union)."""
from numbers import Real
import statecounter as sc
from ..types import Pool_type, Optional, Sequence, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


class StackOp(Operation):
    """Stack multiple pools sequentially (disjoint union)."""
    factory_name = "stack"
    design_card_keys = ['active_parent']
    
    @beartype
    def __init__(
        self,
        parent_pools: list,
        name: Optional[str] = None,
        op_iteration_order: Real = 0,
    ) -> None:
        """Initialize StackOp.
        
        Args:
            parent_pools: List of parent pools to stack.
            name: Optional operation name.
            op_iteration_order: Iteration order for this operation's counter.
        """
        # Compute seq_length: same as parents if all equal, else None
        parent_lengths = [p.seq_length for p in parent_pools]
        if parent_lengths and all(L == parent_lengths[0] for L in parent_lengths):
            seq_length = parent_lengths[0]
        else:
            seq_length = None
        super().__init__(
            parent_pools=parent_pools,
            num_states=1,
            seq_length=seq_length,
            name=name,
            iter_order=op_iteration_order,
        )
    
    @beartype
    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool_type],
    ) -> sc.Counter:
        """Build pool counter using sc.stack (disjoint union)."""
        parent_counters = [p.counter for p in parent_pools]
        return sc.stack(parent_counters)
    
    @beartype
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with active parent index."""
        for i, parent in enumerate(self.parent_pools):
            if parent.counter.state is not None:
                return {'active_parent': i}
        return {'active_parent': None}
    
    @beartype
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Return the sequence from the active parent."""
        active = card['active_parent']
        if active is None:
            seq = parent_seqs[0] if parent_seqs else ''
        else:
            seq = parent_seqs[active]
        return {'seq_0': seq}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pools': self.parent_pools,
            'name': None,
            'op_iteration_order': self.iter_order,
        }


@beartype
def stack(
    pools: list,
    pool_iteration_order: Real = 0,
    op_iteration_order: Real = 0,
    op_name: Optional[str] = None,
    name: Optional[str] = None,
) -> Pool_type:
    """Stack multiple pools sequentially (disjoint union).
    
    Args:
        pools: List of pools to stack.
        pool_iteration_order: Sort key for the result pool (default 0).
        op_iteration_order: Sort key for this operation's counter (default 0).
        op_name: Optional operation name.
        name: Optional pool name.
    
    Returns:
        A pool that is the disjoint union of all input pools.
    """
    op = StackOp(pools, name=op_name, op_iteration_order=op_iteration_order)
    result_pool = Pool(operation=op, output_index=0)
    result_pool.iter_order = pool_iteration_order
    if name is not None:
        result_pool.name = name
    return result_pool
