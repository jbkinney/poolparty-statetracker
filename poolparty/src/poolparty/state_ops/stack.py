"""Stack operation - combine pools sequentially (disjoint union)."""

from numbers import Real
from typing import TypeVar

import numpy as np

import statetracker as st

from ..operation import Operation
from ..pool import Pool
from ..types import CardsType, Optional, Pool_type, Real, Seq, Sequence, beartype
from ..utils.dna_seq import DnaSeq

T = TypeVar("T", bound=Pool)


@beartype
def stack(
    pools: Sequence[T],
    prefix: Optional[str] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
) -> T:
    """Create a pool by stacking multiple input pools state-wise (disjoint union).

    Parameters
    ----------
    pools : Sequence[Pool_type]
        Sequence of Pool objects to stack into a single Pool.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    cards : list[str] or dict, optional
        Design card keys to include. Available keys: ``'active_parent'``.

    Returns
    -------
    Pool_type
        A Pool whose states are the disjoint union of all input pools' states.
        Each state produces the sequence from the corresponding input pool.
    """
    op = StackOp(pools, prefix=prefix, name=None, iter_order=iter_order, cards=cards)
    # Return same type as first input pool
    pool_class = type(pools[0]) if pools else Pool
    result_pool = pool_class(operation=op)
    return result_pool


class StackOp(Operation):
    """Stack multiple pools sequentially (disjoint union)."""

    factory_name = "stack"
    design_card_keys = ["active_parent"]

    def __init__(
        self,
        parent_pools: Sequence[Pool_type],
        prefix: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
    ) -> None:
        """Initialize StackOp."""
        # Compute seq_length: same as parents if all equal, else None
        parent_lengths = [p.seq_length for p in parent_pools]
        if parent_lengths and all(L == parent_lengths[0] for L in parent_lengths):
            seq_length = parent_lengths[0]
        else:
            seq_length = None
        super().__init__(
            parent_pools=parent_pools,
            num_states=len(parent_pools),  # Number of branches
            mode="sequential",  # Stack needs its own state to track active branch
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            cards=cards,
        )

    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool_type],
    ) -> st.State:
        """Build pool state using st.stack (disjoint union)."""
        parent_states = [p.state for p in parent_pools]
        return st.stack(parent_states)

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return Seq from active parent and design card."""
        # Find active parent
        for i, parent in enumerate(self.parent_pools):
            if parent.state.value is not None:
                # Set operation counter to branch index (safe for leaf counter)
                self.state.value = i
                active = i
                output_seq = parents[active]
                return output_seq, {"active_parent": active}

        # No active parent
        self.state.value = None
        active = None
        output_seq = parents[0] if parents else DnaSeq.empty()
        return output_seq, {"active_parent": active}
