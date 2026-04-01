"""Sample operation - sample states from a pool."""

from numbers import Real
from typing import TypeVar

import numpy as np

import statetracker as st

from ..operation import Operation
from ..pool import Pool
from ..types import Integral, Optional, Pool_type, Real, Seq, Sequence, beartype

T = TypeVar("T", bound=Pool)


@beartype
def sample(
    pool: T,
    num_seqs: Optional[Integral] = None,
    seq_states: Optional[Sequence[Integral]] = None,
    seed: Optional[Integral] = None,
    with_replacement: bool = True,
    prefix: Optional[str] = None,
    iter_order: Optional[Real] = None,
) -> T:
    """Sample states from a pool.

    Parameters
    ----------
    pool : Pool_type
        The Pool to sample states from.
    num_seqs : Optional[Integral], default=None
        Number of states to sample randomly. Mutually exclusive with ``seq_states``.
    seq_states : Optional[Sequence[Integral]], default=None
        Explicit list of state indices to select. Mutually exclusive with ``num_seqs``.
    seed : Optional[Integral], default=None
        Random seed for deterministic sampling. Only used with ``num_seqs``.
    with_replacement : bool, default=True
        Whether to sample with replacement. If False, ``num_seqs`` must be
        <= ``pool.num_states``.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool_type
        A Pool containing the sampled states from the input Pool.

    Raises
    ------
    ValueError
        If both ``num_seqs`` and ``seq_states`` are provided, or if neither is provided.
        If ``with_replacement`` is False and ``num_seqs`` exceeds the pool's state count.
    """
    op = SampleOp(
        pool,
        num_seqs=num_seqs,
        seq_states=seq_states,
        seed=seed,
        with_replacement=with_replacement,
        prefix=prefix,
        name=None,
        iter_order=iter_order,
    )
    # Return same type as input
    pool_class = type(pool)
    result_pool = pool_class(operation=op)
    return result_pool


class SampleOp(Operation):
    """Sample states from a pool."""

    factory_name = "sample"
    design_card_keys = []

    def __init__(
        self,
        parent_pool: Pool_type,
        num_seqs: Optional[Integral] = None,
        seq_states: Optional[Sequence[Integral]] = None,
        seed: Optional[Integral] = None,
        with_replacement: bool = True,
        prefix: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize SampleOp."""
        self._num_seqs = num_seqs
        self.seq_states = seq_states
        self.seed = seed
        self.with_replacement = with_replacement
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
        """Build pool counter using st.sample."""
        return st.sample(
            parent_pools[0].state,
            num_values=self._num_seqs,
            sampled_states=self.seq_states,
            seed=self.seed,
            with_replacement=self.with_replacement,
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return parent Seq (state mapping handled by counter)."""
        return parents[0], {}
