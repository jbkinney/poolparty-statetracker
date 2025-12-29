"""
SliceCoOp - Slice a counter to select a subset of states.
"""
from ..counter_operation import CounterOperation


class SliceCoOp(CounterOperation):
    """Slice a counter to select a subset of states."""
    
    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step
    
    def compute_num_states(self, parent_num_states):
        # Calculate length of range(start, stop, step)
        return len(range(self.start, self.stop, self.step))
    
    def decompose(self, state, parent_num_states):
        if state == -1:
            return (-1,)
        # Map sliced state back to parent state
        parent_state = self.start + (state * self.step)
        return (parent_state,)


def slice_counter(counter, start=None, stop=None, step=None, name=None):
    """Create a sliced counter from a subset of states.
    
    Args:
        counter: The Counter to slice.
        start: Start index (default: 0 for positive step, num_states-1 for negative).
        stop: Stop index (default: num_states for positive step, -1 for negative).
        step: Step size (default: 1). Can be negative for reverse iteration.
        name: Optional name for the resulting counter.
    
    Returns:
        A new Counter representing the sliced subset of states.
    
    Examples:
        slice_counter(A, 1, 5)       # A[1:5]
        slice_counter(A, step=2)     # A[::2]
        slice_counter(A, step=-1)    # A[::-1] (reversed)
    """
    from ..counter import Counter
    
    if not isinstance(counter, Counter):
        raise TypeError(f"Expected Counter, got {type(counter)}")
    
    # Use slice.indices() to normalize start/stop/step
    s = slice(start, stop, step)
    start_norm, stop_norm, step_norm = s.indices(counter.num_states)
    
    result = Counter(_parents=(counter,), _op=SliceCoOp(start_norm, stop_norm, step_norm))
    if name is not None:
        result.name = name
    return result

