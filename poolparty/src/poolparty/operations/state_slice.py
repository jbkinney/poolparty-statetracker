"""StateSlice operation - slice a pool's states (not sequences)."""
from numbers import Real
import statecounter as sc
from ..types import Pool_type, Union, Optional, Sequence, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


class StateSliceOp(Operation):
    """Slice a pool's states to select a subset."""
    factory_name = "state_slice"
    design_card_keys = []
    
    @beartype
    def __init__(
        self,
        parent_pool: Pool_type,
        start: Optional[int],
        stop: Optional[int],
        step: Optional[int],
        name: Optional[str] = None,
        op_iteration_order: Real = 0,
    ) -> None:
        """Initialize StateSliceOp."""
        self.start = start
        self.stop = stop
        self.step = step
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,
            name=name,
            iter_order=op_iteration_order,
        )
    
    @beartype
    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool_type],
    ) -> sc.Counter:
        """Build pool counter using sc.slice."""
        return sc.slice(
            parent_pools[0].counter, 
            start=self.start, 
            stop=self.stop, 
            step=self.step,
        )
    
    @beartype
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return empty design card (no design decisions)."""
        return {}
    
    @beartype
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
            'op_iteration_order': self.iter_order,
        }


@beartype
def state_slice(
    pool: Pool_type,
    key: Union[int, slice],
    pool_iteration_order: Real = 0,
    op_iteration_order: Real = 0,
    op_name: Optional[str] = None,
    name: Optional[str] = None,
) -> Pool_type:
    """Slice a pool's states (not sequences)."""
    if isinstance(key, int):
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
                      op_iteration_order=op_iteration_order)
    result_pool = Pool(operation=op, output_index=0)
    result_pool.iter_order = pool_iteration_order
    if name is not None:
        result_pool.name = name
    return result_pool
