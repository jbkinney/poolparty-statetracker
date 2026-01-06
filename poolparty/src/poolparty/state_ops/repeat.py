"""Repeat operation - repeat a pool's states n times."""
from numbers import Real
from ..types import Pool_type, Optional, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def repeat(
    pool: Pool_type,
    times: int,
    name_prefix: Optional[str] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool_type:
    """
    Repeat the states of a pool a specified number of times, producing a new pool with
    `times` as many states as the input pool has.

    Parameters
    ----------
    pool : Pool_type
        The Pool whose state(s) are to be repeated.
    times : int
        The number of times to repeat the pool's state(s).
    name : Optional[str], default=None
        Name to assign to the resulting Pool.
    op_name : Optional[str], default=None
        Name to assign to the internal RepeatOp operation.
    iter_order : Real, default=0
        Iteration order priority for the resulting Pool.
    op_iter_order : Real, default=0
        Iteration order priority for the internal RepeatOp operation (typically unused).

    Returns
    -------
    Pool_type
        A new Pool with `times` as many states as the input pool has.
    """
    op = RepeatOp(pool, times=times, name_prefix=name_prefix, name=op_name, iter_order=op_iter_order)
    result_pool = Pool(operation=op, name=name, iter_order=iter_order)
    return result_pool


@beartype
class RepeatOp(Operation):
    """Repeat a pool's states n times."""
    factory_name = "repeat"
    design_card_keys = ['repeat_index']
    
    def __init__(
        self,
        parent_pool: Pool_type,
        times: int,
        name_prefix: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
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
            iter_order=iter_order,
            name_prefix=name_prefix,
        )
    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with repeat index."""
        state = self.counter.state
        repeat_index = 0 if state is None else state
        return {'repeat_index': repeat_index}
    
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
            'name_prefix': self.name_prefix,
            'name': None,
            'iter_order': self.iter_order,
        }
