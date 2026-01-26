"""Repeat operation - repeat a pool's states n times."""
from numbers import Real
from ..types import Pool_type, Optional, beartype, SeqStyle
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
            num_values=times,
            mode='sequential',
            seq_length=pool.seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
        )
    
    def compute(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
        parent_styles: list[SeqStyle] | None = None,
    ) -> dict:
        """Return design card and parent sequence together."""
        state = self.state.value
        repeat_index = 0 if state is None else state
        seq = parent_seqs[0]
        # Pass through parent styles
        output_style = SeqStyle.from_parent(parent_styles, 0, len(seq))
        return {'repeat_index': repeat_index, 'seq': seq, 'style': output_style}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'pool': self.parent_pools[0],
            'times': self.times,
            'prefix': self.name_prefix,
            'name': None,
            'iter_order': self.iter_order,
        }
