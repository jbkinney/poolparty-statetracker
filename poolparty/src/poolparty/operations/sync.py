"""Sync operation - synchronize pools to iterate in lockstep."""
import statecounter as sc
from ..types import Optional, beartype


@beartype
def sync(pools: list, name: Optional[str] = None) -> None:
    """Synchronize multiple pools to iterate in lockstep.
    
    Modifies the pools in-place by replacing their counters with
    a single synchronized counter.
    
    Args:
        pools: List of Pool instances to synchronize.
        name: Optional name for the shared synchronized counter.
    
    Raises:
        ValueError: If pools have different num_states.
    """
    if not pools:
        return
    
    sizes = set(p.num_states for p in pools)
    if len(sizes) > 1:
        raise ValueError(f"Cannot sync pools with different num_states: {sizes}")
    
    counters = [p.counter for p in pools]
    shared_counter = sc.sync(counters, name=name)
    
    for pool in pools:
        pool.counter = shared_counter
