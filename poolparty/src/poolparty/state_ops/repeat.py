"""Repeat operation - repeat a pool's states n times."""

from numbers import Integral, Real
from typing import TypeVar

import numpy as np

from ..operation import Operation
from ..pool import Pool
from ..types import CardsType, Optional, Pool_type, Seq, beartype

T = TypeVar("T", bound=Pool)


@beartype
def repeat(
    pool: T,
    times: Integral,
    prefix: Optional[str] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
) -> T:
    """Repeat a pool's states a specified number of times.

    Parameters
    ----------
    pool : Pool_type
        The Pool whose states are to be repeated.
    times : Integral
        The number of times to repeat the pool's states. Must be >= 1.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    cards : list[str] or dict, optional
        Design card keys to include. Available keys: ``'repeat_index'``.

    Returns
    -------
    Pool_type
        A new Pool with ``times`` as many states as the input pool.

    Raises
    ------
    ValueError
        If ``times`` is less than 1.
    """
    op = RepeatOp(pool, times=times, prefix=prefix, name=None, iter_order=iter_order, cards=cards)
    # Return same type as input
    pool_class = type(pool)
    result_pool = pool_class(operation=op)
    return result_pool


class RepeatOp(Operation):
    """Repeat a pool's states n times."""

    factory_name = "repeat"
    design_card_keys = ["repeat_index"]

    def __init__(
        self,
        pool: Pool_type,
        times: Integral,
        prefix: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
    ) -> None:
        """Initialize RepeatOp."""
        times = int(times)
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
            cards=cards,
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return parent Seq and design card."""
        # Use state 0 when inactive (state is None)
        state = self.state.value
        repeat_index = 0 if state is None else state

        # Pass through parent Seq
        return parents[0], {"repeat_index": repeat_index}
