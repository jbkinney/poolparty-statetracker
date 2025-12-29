"""RepeatCoOp - Repeat a single counter N times."""
from ..counter_operation import CounterOperation


class RepeatCoOp(CounterOperation):
    """Repeat a single counter N times (equivalent to A + A + ... N times)."""
    
    def __init__(self, times):
        self.times = times
    
    def compute_num_states(self, parent_num_states):
        return parent_num_states[0] * self.times
    
    def decompose(self, state, parent_num_states):
        if state is None:
            return (None,)
        return (state % parent_num_states[0],)


def repeat_counter(counter, times, name=None):
    """Create a counter that repeats another counter N times."""
    from ..counter import Counter
    if not isinstance(counter, Counter):
        raise TypeError(f"Expected Counter, got {type(counter)}")
    if times < 1:
        raise ValueError("times must be at least 1")
    result = Counter(_parents=(counter,), _op=RepeatCoOp(times))
    if name is not None:
        result.name = name
    return result
