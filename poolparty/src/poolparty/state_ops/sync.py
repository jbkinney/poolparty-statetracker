"""Sync operation - synchronize pools to iterate in lockstep."""
import statetracker as st
from ..types import Optional, Sequence, beartype
from ..pool import Pool


@beartype
def sync(
    pools: Sequence[Pool], 
    name: Optional[str] = None,
) -> Pool:
    """
    Create a Pool that synchronizes multiple input Pools to iterate in lockstep.

    Parameters
    ----------
    pools : Sequence[Pool]
        Sequence of Pool objects to synchronize. All pools must have the same number of states.
    name : Optional[str], default=None
        Name for the resulting synchronized Pool.

    Returns
    -------
    Pool
        A Pool whose state selection causes all input Pools to advance synchronously through their states.

    Raises
    ------
    ValueError
        If the input sequence is empty or if the pools have differing numbers of states.
    """
    if not pools:
        raise ValueError("Cannot sync empty sequence of pools")

    sizes = set(p.num_values for p in pools)
    if len(sizes) > 1:
        raise ValueError(f"Cannot sync pools with different num_values: {sizes=}")

    states = [p.state for p in pools]
    shared_state = st.sync(states, name=name)
    for pool in pools:
        pool.state = shared_state
