"""StateSlice operation - slice a pool's states (not sequences)."""
from numbers import Real
import statetracker as st
from ..types import Union, Optional, Sequence, Integral, Real, beartype, SeqStyle
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def state_slice(
    pool: Pool,
    key: Union[Integral, slice],
    prefix: Optional[str] = None,
    iter_order: Optional[Real] = None,
) -> Pool:
    """
    Create a Pool containing a slice of states from the input Pool.

    Parameters
    ----------
    pool : Pool
        The Pool whose states will be sliced.
    key : Union[Integral, slice]
        Integer index or slice specifying which states to include from the input Pool.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool containing states selected by applying the provided index or slice to the input Pool's state space.
    
    Raises
    ------
    ValueError
        If the input pool has no state (mode='random' with num_states=None).
    """
    # Check for stateless pool
    if pool.state is None:
        raise ValueError(
            f"Cannot slice stateless pool '{pool.name}'. "
            f"Pools with mode='random' and num_states=None have no state to slice. "
            f"Use num_states=N to create a pool with explicit states."
        )
    if isinstance(key, Integral):
        if key < 0:
            start = key
            stop = key + 1 if key != -1 else None
        else:
            start = key
            stop = key + 1
        step = 1
    else:
        start = key.start
        stop = key.stop
        step = key.step
    op = StateSliceOp(pool, start=start, stop=stop, step=step, prefix=prefix,
                      name=None, iter_order=iter_order)
    result_pool = Pool(operation=op)
    return result_pool


@beartype
class StateSliceOp(Operation):
    """Slice a pool's states to select a subset."""
    factory_name = "state_slice"
    design_card_keys = []
    
    def __init__(
        self,
        parent_pool: Pool,
        start: Optional[Integral],
        stop: Optional[Integral],
        step: Optional[Integral],
        prefix: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize StateSliceOp."""
        self.start = start
        self.stop = stop
        self.step = step
        super().__init__(
            parent_pools=[parent_pool],
            num_values=1,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
        )
    
    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool],
    ) -> st.State:
        """Build pool counter using st.slice."""
        return st.slice(
            parent_pools[0].state, 
            start=self.start, 
            stop=self.stop, 
            step=self.step,
        )
    
    def compute(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
        parent_styles: list[SeqStyle] | None = None,
    ) -> dict:
        """Return parent sequence (state mapping handled by counter)."""
        seq = parent_seqs[0]
        output_style = SeqStyle.from_parent(parent_styles, 0, len(seq))
        return {'seq': seq, 'style': output_style}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'start': self.start,
            'stop': self.stop,
            'step': self.step,
            'prefix': self.name_prefix,
            'name': None,
            'iter_order': self.iter_order,
        }
