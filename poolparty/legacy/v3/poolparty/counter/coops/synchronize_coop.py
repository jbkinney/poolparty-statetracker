"""
SynchronizeCoOp - Keep N counters in lockstep.
"""
from ..counter_operation import CounterOperation


class SynchronizeCoOp(CounterOperation):
    """Keep N counters in lockstep."""
    
    def compute_num_states(self, parent_num_states):
        if len(set(parent_num_states)) != 1:
            raise ValueError(
                f"Cannot sync counters with different num_states: {parent_num_states}"
            )
        return parent_num_states[0]
    
    def decompose(self, state, parent_num_states):
        # All parents get the same state value
        return tuple(state for _ in parent_num_states)


def synchronize_counters(*counters, name=None):
    """Create sync counter from multiple counters.
    
    Args:
        *counters: Two or more Counter objects (must have same num_states).
        name: Optional name for the resulting counter.
    
    Returns:
        A new Counter that keeps all parents in lockstep.
    """
    from ..counter import Counter
    
    if len(counters) < 2:
        raise ValueError("synchronize_counters() requires at least 2 counters")
    for c in counters:
        if not isinstance(c, Counter):
            raise TypeError(f"Expected Counter, got {type(c)}")
    result = Counter(_parents=counters, _op=SynchronizeCoOp())
    if name is not None:
        result.name = name
    return result

