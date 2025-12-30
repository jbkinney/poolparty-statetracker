"""Stack operation - combine pools sequentially (disjoint union)."""
from numbers import Real
import statecounter as sc
from ..types import Optional, Sequence, Integral, Real, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
class StackOp(Operation):
    """Stack multiple pools sequentially (disjoint union)."""
    factory_name = "stack"
    design_card_keys = ['active_parent']
    
    def __init__(
        self,
        parent_pools: Sequence[Pool],
        name: Optional[str] = None,
        iter_order: Real = 0,
    ) -> None:
        """Initialize StackOp."""
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
            iter_order=iter_order,
        )
    
    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool],
    ) -> sc.Counter:
        """Build pool counter using sc.stack (disjoint union)."""
        parent_counters = [p.counter for p in parent_pools]
        return sc.stack(parent_counters)
    
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
            'iter_order': self.iter_order,
        }


@beartype
def stack(
    pools: Sequence[Pool],
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Real = 0,
    op_iter_order: Real = 0,
) -> Pool:
    """
    Create a Pool by stacking multiple input Pools state-wise.

    Parameters
    ----------
    pools : Sequence[Pool]
        Sequence of Pool objects to stack into a single Pool.
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Stack operation.
    iter_order : Real, default=0
        Iteration order priority for the resulting Pool.
    op_iter_order : Real, default=0
        Iteration order priority for the underlying Stack operation.

    Returns
    -------
    Pool
        A Pool object representing the state-wise stacking of all provided input Pools. 
        Each state corresponds to a sequence from one of the input Pools.
    """
    op = StackOp(pools, name=op_name, iter_order=op_iter_order)
    result_pool = Pool(operation=op, name=name, iter_order=iter_order)
    return result_pool
