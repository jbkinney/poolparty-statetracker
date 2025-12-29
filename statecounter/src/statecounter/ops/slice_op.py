"""SliceOp - Slice a counter to select a subset of states."""
from ..operation import Operation
import builtins


class SliceOp(Operation):
    """Slice a counter to select a subset of states."""
    
    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step
    
    def compute_num_states(self, parent_num_states):
        return len(range(self.start, self.stop, self.step))
    
    def decompose(self, state, parent_num_states):
        if state is None:
            return (None,)
        parent_state = self.start + (state * self.step)
        return (parent_state,)


def slice(counter, start=None, stop=None, step=None, name=None):
    """Create a sliced counter from a subset of states."""
    from ..counter import Counter
    if not isinstance(counter, Counter):
        raise TypeError(f"Expected Counter, got {type(counter)}")
    s = builtins.slice(start, stop, step)
    start_norm, stop_norm, step_norm = s.indices(counter.num_states)
    result = Counter(_parents=(counter,), _op=SliceOp(start_norm, stop_norm, step_norm))
    if name is not None:
        result.name = name
    return result
