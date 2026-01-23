"""Stack operation - combine pools sequentially (disjoint union)."""
from numbers import Real
import statetracker as st
from ..types import Optional, Sequence, Integral, Real, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def stack(
    pools: Sequence[Pool],
    seq_name_prefix: Optional[str] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
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
    
    Raises
    ------
    ValueError
        If any input pool has no state (mode='random' with num_states=None).
    """
    # Check for stateless pools
    for i, pool in enumerate(pools):
        if pool.state is None:
            raise ValueError(
                f"Cannot stack stateless pool '{pool.name}' (index {i}). "
                f"Pools with mode='random' and num_states=None have no state to stack. "
                f"Use num_states=N to create a pool with explicit states."
            )
    op = StackOp(pools, seq_name_prefix=seq_name_prefix, name=op_name, iter_order=op_iter_order)
    result_pool = Pool(operation=op, name=name, iter_order=iter_order)
    return result_pool


@beartype
class StackOp(Operation):
    """Stack multiple pools sequentially (disjoint union)."""
    factory_name = "stack"
    design_card_keys = ['active_parent']
    
    def __init__(
        self,
        parent_pools: Sequence[Pool],
        seq_name_prefix: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
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
            num_values=len(parent_pools),  # Number of branches
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
            seq_name_prefix=seq_name_prefix,
        )
    
    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool],
    ) -> st.State:
        """Build pool state using st.stack (disjoint union)."""
        parent_states = [p.state for p in parent_pools]
        return st.stack(parent_states)
    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with active parent index."""
        for i, parent in enumerate(self.parent_pools):
            if parent.state.value is not None:
                # Set operation counter to branch index (safe for leaf counter)
                self.state.value = i
                return {'active_parent': i}
        self.state.value = None
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
    
    def compute_seq_names(
        self,
        parent_names: list[Optional[str]],
        card: dict,
    ) -> dict:
        """Return the name from the active parent."""
        # Block all names if _block_seq_names is set
        if self._block_seq_names:
            return {'name_0': None}
        # Apply clear_parent_names if set
        if self.clear_parent_names:
            parent_names = [None] * len(parent_names)
        
        # Get name from active parent (matching compute_seq_from_card logic)
        active = card['active_parent']
        if active is None:
            name = parent_names[0] if parent_names else None
        else:
            name = parent_names[active]
        
        # Append prefix if set
        if self.name_prefix is not None:
            state = self.state.value
            if state is not None:
                op_name = f'{self.name_prefix}{state}'
                name = f'{name}.{op_name}' if name else op_name
        
        return {'name_0': name}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pools': self.parent_pools,
            'seq_name_prefix': self.name_prefix,
            'name': None,
            'iter_order': self.iter_order,
        }
