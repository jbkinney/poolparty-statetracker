"""StateShuffle operation - randomly permute a pool's states."""

from numbers import Real
from typing import TypeVar

import numpy as np

import statetracker as st

from ..operation import Operation
from ..pool import Pool
from ..types import Integral, Optional, Pool_type, Real, Seq, Sequence, beartype

T = TypeVar("T", bound=Pool)


@beartype
def state_shuffle(
    pool: T,
    seed: Optional[Integral] = None,
    permutation: Optional[Sequence[Integral]] = None,
    prefix: Optional[str] = None,
    iter_order: Optional[Real] = None,
) -> T:
    """
    Create a Pool with randomly permuted states from the input Pool.

    Parameters
    ----------
    pool : Pool_type
        The Pool whose states will be shuffled.
    seed : Optional[Integral], default=None
        Random seed for deterministic shuffling. If None, a random seed is generated.
    permutation : Optional[Sequence[Integral]], default=None
        Custom permutation to use. If provided, seed must not be specified.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool_type
        A Pool containing the same states as the input but in a randomly permuted order.
    """
    op = StateShuffleOp(
        pool, seed=seed, permutation=permutation, prefix=prefix, name=None, iter_order=iter_order
    )
    # Return same type as input
    pool_class = type(pool)
    result_pool = pool_class(operation=op)
    return result_pool


class StateShuffleOp(Operation):
    """Randomly permute a pool's states."""

    factory_name = "state_shuffle"
    design_card_keys = []

    def __init__(
        self,
        parent_pool: Pool_type,
        seed: Optional[Integral] = None,
        permutation: Optional[Sequence[Integral]] = None,
        prefix: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize StateShuffleOp."""
        self.seed = seed
        self.permutation = permutation
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,
            seq_length=parent_pool.seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
        )

    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool],
    ) -> st.State:
        """Build pool counter using st.shuffle."""
        return st.shuffle(
            parent_pools[0].state,
            seed=self.seed,
            permutation=self.permutation,
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return parent Seq (state mapping handled by counter)."""
        return parents[0], {}
