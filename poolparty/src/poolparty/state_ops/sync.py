"""Sync operation - synchronize pools to iterate in lockstep."""

import statetracker as st

from ..pool import Pool
from ..types import Sequence, beartype


def _is_ancestor(candidate, state) -> bool:
    """Check if *candidate* is an ancestor of *state* in the state DAG."""
    visited: set = set()
    stack = list(state._parents)
    while stack:
        current = stack.pop()
        if current is candidate:
            return True
        cid = id(current)
        if cid in visited:
            continue
        visited.add(cid)
        stack.extend(current._parents)
    return False


@beartype
def sync(
    pools: Sequence[Pool],
) -> None:
    """Synchronize multiple pools to iterate in lockstep (in-place).

    Parameters
    ----------
    pools : Sequence[Pool]
        Sequence of Pool objects to synchronize. All pools must have the same
        number of states.

    Returns
    -------
    None
        Pools are modified in-place; no new Pool is returned.

    Raises
    ------
    ValueError
        If the input sequence is empty, if the pools have differing numbers of
        states, or if any pool is an ancestor of another (circular constraint).
    """
    if not pools:
        raise ValueError("Cannot sync empty sequence of pools")

    sizes = set(p.num_states for p in pools)
    if len(sizes) > 1:
        raise ValueError(f"Cannot sync pools with different num_states: {sizes=}")

    for i, pi in enumerate(pools):
        for j, pj in enumerate(pools):
            if i != j and _is_ancestor(pi.state, pj.state):
                raise ValueError(
                    f"Cannot sync pools with ancestor-descendant relationship: "
                    f"pool[{i}] is an ancestor of pool[{j}]. "
                    f"Syncing them would create a circular state dependency."
                )

    wrappers = [st.synced_to(p.state) for p in pools]
    for w in wrappers[1:]:
        st.sync(wrappers[0], w)
    for pool, w in zip(pools, wrappers):
        pool.state = w
