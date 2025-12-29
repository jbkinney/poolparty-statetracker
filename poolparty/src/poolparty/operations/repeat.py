"""Repeat operation - repeat a pool's states n times."""
from numbers import Real
from ..types import Pool_type, Optional, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


class RepeatOp(Operation):
    """Repeat a pool's states n times."""
    factory_name = "repeat"
    design_card_keys = ['repeat_index']
    
    @beartype
    def __init__(
        self,
        parent_pool: Pool_type,
        times: int,
        name: Optional[str] = None,
        op_iteration_order: Real = 0,
    ) -> None:
        """Initialize RepeatOp."""
        if times < 1:
            raise ValueError(f"times must be >= 1, got {times}")
        self.times = times
        super().__init__(
            parent_pools=[parent_pool],
            num_states=times,
            mode='sequential',
            seq_length=parent_pool.seq_length,
            name=name,
            iter_order=op_iteration_order,
        )
    
    @beartype
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with repeat index."""
        state = self.counter.state
        repeat_index = 0 if state is None else state
        return {'repeat_index': repeat_index}
    
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
            'times': self.times,
            'name': None,
            'op_iteration_order': self.iter_order,
        }


@beartype
def repeat(
    pool: Pool_type,
    times: int,
    pool_iteration_order: Real = 0,
    op_iteration_order: Real = 0,
    op_name: Optional[str] = None,
    name: Optional[str] = None,
) -> Pool_type:
    """Repeat a pool's states n times."""
    op = RepeatOp(pool, times=times, name=op_name, op_iteration_order=op_iteration_order)
    result_pool = Pool(operation=op, output_index=0)
    result_pool.iter_order = pool_iteration_order
    if name is not None:
        result_pool.name = name
    return result_pool
