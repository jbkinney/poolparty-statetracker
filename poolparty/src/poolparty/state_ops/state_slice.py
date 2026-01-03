"""StateSlice operation - slice a pool's states (not sequences)."""
from numbers import Real
import statecounter as sc
from ..types import Union, Optional, Sequence, Integral, Real, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def state_slice(
    pool: Pool,
    key: Union[Integral, slice],
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,    
) -> Pool:
    """
    Create a Pool containing a slice of states from the input Pool.

    Parameters
    ----------
    pool : Pool
        The Pool whose states will be sliced.
    key : Union[Integral, slice]
        Integer index or slice specifying which states to include from the input Pool.
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying state slice Operation.
    iter_order : Real, default=0
        Iteration order priority for the resulting Pool.
    op_iter_order : Real, default=0
        Iteration order priority for the underlying Operation.

    Returns
    -------
    Pool
        A Pool containing states selected by applying the provided index or slice to the input Pool's state space.
    """
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
    op = StateSliceOp(pool, start=start, stop=stop, step=step, name=op_name,
                      iter_order=op_iter_order)
    result_pool = Pool(operation=op, name=name, iter_order=iter_order)
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
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize StateSliceOp."""
        self.start = start
        self.stop = stop
        self.step = step
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,
            name=name,
            iter_order=iter_order,
        )
    
    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool],
    ) -> sc.Counter:
        """Build pool counter using sc.slice."""
        return sc.slice(
            parent_pools[0].counter, 
            start=self.start, 
            stop=self.stop, 
            step=self.step,
        )
    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return empty design card (no design decisions)."""
        return {}
    
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Return the parent sequence (state mapping handled by counter)."""
        return {'seq_0': parent_seqs[0]}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'start': self.start,
            'stop': self.stop,
            'step': self.step,
            'name': None,
            'iter_order': self.iter_order,
        }
