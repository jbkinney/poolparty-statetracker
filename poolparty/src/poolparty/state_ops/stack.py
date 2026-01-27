"""Stack operation - combine pools sequentially (disjoint union)."""
from numbers import Real
import statetracker as st
from ..types import Optional, Sequence, Integral, Real, beartype, Seq
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def stack(
    pools: Sequence[Pool],
    prefix: Optional[str] = None,
    iter_order: Optional[Real] = None,
) -> Pool:
    """
    Create a Pool by stacking multiple input Pools state-wise.

    Parameters
    ----------
    pools : Sequence[Pool]
        Sequence of Pool objects to stack into a single Pool.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

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
    op = StackOp(pools, prefix=prefix, name=None, iter_order=iter_order)
    result_pool = Pool(operation=op)
    return result_pool


@beartype
class StackOp(Operation):
    """Stack multiple pools sequentially (disjoint union)."""
    factory_name = "stack"
    design_card_keys = ['active_parent']
    
    def __init__(
        self,
        parent_pools: Sequence[Pool],
        prefix: Optional[str] = None,
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
            num_states=len(parent_pools),  # Number of branches
            mode='sequential',  # Stack needs its own state to track active branch
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
        )
    
    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool],
    ) -> st.State:
        """Build pool state using st.stack (disjoint union)."""
        parent_states = [p.state for p in parent_pools]
        return st.stack(parent_states)
    
    def compute(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return Seq from active parent and design card."""
        # Find active parent
        for i, parent in enumerate(self.parent_pools):
            if parent.state.value is not None:
                # Set operation counter to branch index (safe for leaf counter)
                self.state.value = i
                active = i
                output_seq = parents[active]
                
                # Compute name from active parent
                name = self._compute_stack_name(parents, active)
                output_seq = output_seq.with_name(name)
                
                return output_seq, {'active_parent': active}
        
        # No active parent
        self.state.value = None
        active = None
        output_seq = parents[0] if parents else Seq.empty()
        
        # Compute name (from first parent)
        name = self._compute_stack_name(parents, 0 if parents else None)
        output_seq = output_seq.with_name(name)
        
        return output_seq, {'active_parent': active}
    
    def _compute_stack_name(self, parents: list[Seq], active: int | None) -> Optional[str]:
        """Compute name from active parent with optional prefix."""
        # Block name if _block_seq_names is set
        if self._block_seq_names:
            return None
        
        # Get name from active parent
        if self.clear_parent_names or active is None:
            name = None
        else:
            name = parents[active].name if active < len(parents) else None
        
        # Append prefix if set
        if self.prefix is not None:
            state = self.state.value
            if state is not None:
                op_name = f'{self.prefix}{state}'
                name = f'{name}.{op_name}' if name else op_name
        
        return name
