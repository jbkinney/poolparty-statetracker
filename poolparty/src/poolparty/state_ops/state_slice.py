"""StateSlice operation - slice a pool's states (not sequences)."""

from numbers import Real

import numpy as np

import statetracker as st

from ..operation import Operation
from ..pool import Pool
from ..types import Integral, Optional, Real, Seq, Sequence, Union, beartype


@beartype
def state_slice(
    pool: Pool,
    key: Union[Integral, slice],
    prefix: Optional[str] = None,
    iter_order: Optional[Real] = None,
) -> Pool:
    """
    Create a Pool containing a slice of states from the input Pool.

    Parameters
    ----------
    pool : Pool
        The Pool whose states will be sliced.
    key : Union[Integral, slice]
        Integer index or slice specifying which states to include from the input Pool.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool containing states selected by applying the provided index or slice to the input Pool's state space.

    Raises
    ------
    ValueError
        If the input pool has no state (mode='random' with num_states=None).
    """
    # Check for fixed/stateless pool
    if pool.state.is_fixed:
        raise ValueError(
            f"Cannot slice fixed/stateless pool '{pool.name}'. "
            f"Pools with mode='random' and num_states=None have no state to slice. "
            f"Use num_states=N to create a pool with explicit states."
        )
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
    op = StateSliceOp(
        pool, start=start, stop=stop, step=step, prefix=prefix, name=None, iter_order=iter_order
    )
    result_pool = Pool(operation=op)
    return result_pool


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
        prefix: Optional[str] = None,
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
            prefix=prefix,
        )

    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool],
    ) -> st.State:
        """Build pool counter using st.slice."""
        return st.slice(
            parent_pools[0].state,
            start=self.start,
            stop=self.stop,
            step=self.step,
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return parent Seq (state mapping handled by counter)."""
        return parents[0], {}
