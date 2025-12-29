"""
SumCoOp - Disjoint union of N counters.
"""
from ..counter_operation import CounterOperation


class SumCoOp(CounterOperation):
    """D = A + B + ... : Disjoint union of N counters."""
    
    def compute_num_states(self, parent_num_states):
        return sum(parent_num_states)
    
    def decompose(self, state, parent_num_states):
        if state == -1:
            return tuple(-1 for _ in parent_num_states)
        # Find which counter is active
        cumsum = 0
        for i, n in enumerate(parent_num_states):
            if state < cumsum + n:
                # Counter i is active with state (state - cumsum)
                return tuple(
                    state - cumsum if j == i else -1
                    for j in range(len(parent_num_states))
                )
            cumsum += n
        raise ValueError(f"Invalid state {state}")


def sum_counters(*counters, name=None):
    """Create sum counter from multiple counters.
    
    Args:
        *counters: Two or more Counter objects.
        name: Optional name for the resulting counter.
    
    Returns:
        A new Counter representing the disjoint union.
    """
    from ..counter import Counter
    
    if len(counters) < 2:
        raise ValueError("sum_counters() requires at least 2 counters")
    for c in counters:
        if not isinstance(c, Counter):
            raise TypeError(f"Expected Counter, got {type(c)}")
    result = Counter(_parents=counters, _op=SumCoOp())
    if name is not None:
        result.name = name
    return result
