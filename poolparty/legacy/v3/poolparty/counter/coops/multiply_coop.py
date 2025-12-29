"""
MultiplyCoOp - Cartesian product of N counters.
"""
from ..counter_operation import CounterOperation


class MultiplyCoOp(CounterOperation):
    """C = A * B * ... : Cartesian product of N counters."""
    
    def compute_num_states(self, parent_num_states):
        result = 1
        for n in parent_num_states:
            result *= n
        return result
    
    def decompose(self, state, parent_num_states):
        if state == -1:
            return tuple(-1 for _ in parent_num_states)
        result = []
        for n in parent_num_states:
            result.append(state % n)
            state //= n
        return tuple(result)


def multiply_counters(*counters, name=None):
    """Create product counter from multiple counters.
    
    Args:
        *counters: Two or more Counter objects.
        name: Optional name for the resulting counter.
    
    Returns:
        A new Counter representing the Cartesian product.
    """
    from ..counter import Counter
    
    if len(counters) < 2:
        raise ValueError("multiply_counters() requires at least 2 counters")
    for c in counters:
        if not isinstance(c, Counter):
            raise TypeError(f"Expected Counter, got {type(c)}")
    result = Counter(_parents=counters, _op=MultiplyCoOp())
    if name is not None:
        result.name = name
    return result
