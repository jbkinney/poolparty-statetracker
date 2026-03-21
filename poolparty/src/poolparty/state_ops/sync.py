"""Sync operation - synchronize pools to iterate in lockstep."""

import statetracker as st

from ..pool import Pool
from ..types import Sequence, beartype


@beartype
def sync(
    pools: Sequence[Pool],
) -> None:
    """
    Synchronize multiple Pools to iterate in lockstep (in-place).

    Parameters
    ----------
    pools : Sequence[Pool]
        Sequence of Pool objects to synchronize. All pools must have the same number of states.

    Raises
    ------
    ValueError
        If the input sequence is empty or if the pools have differing numbers of states.
    """
    if not pools:
        raise ValueError("Cannot sync empty sequence of pools")

    sizes = set(p.num_states for p in pools)
    if len(sizes) > 1:
        raise ValueError(f"Cannot sync pools with different num_states: {sizes=}")

    states = [p.state for p in pools]
    for s in states[1:]:
        st.sync(states[0], s)
    shared_state = states[0]
    for pool in pools:
        pool.state = shared_state
