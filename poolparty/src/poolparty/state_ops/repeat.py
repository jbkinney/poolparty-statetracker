"""Repeat operation - repeat a pool's states n times."""

from numbers import Real

import numpy as np

from ..operation import Operation
from ..pool import Pool
from ..types import Optional, Pool_type, Seq, beartype


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


class RepeatOp(Operation):
    """Repeat a pool's states n times."""

    factory_name = "repeat"
    design_card_keys = ["repeat_index"]

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
            mode="sequential",
            seq_length=pool.seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return parent Seq and design card."""
        from ..party import cards_suppressed

        # Use state 0 when inactive (state is None)
        state = self.state.value
        repeat_index = 0 if state is None else state

        # Pass through parent Seq
        if cards_suppressed():
            return parents[0], {}
        return parents[0], {"repeat_index": repeat_index}
