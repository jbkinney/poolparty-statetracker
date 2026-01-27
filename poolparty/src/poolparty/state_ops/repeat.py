"""Repeat operation - repeat a pool's states n times."""
from numbers import Real
from ..types import Pool_type, Optional, beartype, Seq
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def repeat(
    pool: Pool_type,
    times: int,
    prefix: Optional[str] = None,
    iter_order: Optional[Real] = None,
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
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool_type
        A new Pool with `times` as many states as the input pool has.
    """
    op = RepeatOp(pool, times=times, prefix=prefix, name=None, iter_order=iter_order)
    result_pool = Pool(operation=op)
    return result_pool


@beartype
class RepeatOp(Operation):
    """Repeat a pool's states n times."""
    factory_name = "repeat"
    design_card_keys = ['repeat_index']
    
    def __init__(
        self,
        pool: Pool_type,
        times: int,
        prefix: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize RepeatOp."""
        if times < 1:
            raise ValueError(f"times must be >= 1, got {times}")
        self.times = times
        super().__init__(
            parent_pools=[pool],
            num_states=times,
            mode='sequential',
            seq_length=pool.seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
        )
    
    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
        suppress_styles: bool = False,
    ) -> tuple[Seq, dict]:
        """Return parent Seq and design card."""
        state = self.state.value
        repeat_index = 0 if state is None else state
        
        # Pass through parent Seq
        return parents[0], {'repeat_index': repeat_index}
