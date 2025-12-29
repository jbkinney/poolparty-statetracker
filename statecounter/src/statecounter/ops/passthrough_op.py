"""PassthroughOp - Pass through a single parent counter unchanged."""
from ..operation import Operation


class PassthroughOp(Operation):
    """Pass through a single parent counter unchanged."""
    
    def compute_num_states(self, parent_num_states):
        return parent_num_states[0]
    
    def decompose(self, state, parent_num_states):
        return (state,)


def passthrough(counter, name=None):
    """Create a passthrough counter that tracks its parent."""
    from ..counter import Counter
    if not isinstance(counter, Counter):
        raise TypeError(f"Expected Counter, got {type(counter)}")
    result = Counter(_parents=(counter,), _op=PassthroughOp())
    if name is not None:
        result.name = name
    return result
