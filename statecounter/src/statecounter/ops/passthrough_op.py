"""PassthroughOp - Pass through a single parent counter unchanged."""
from ..imports import beartype, Optional, Counter_type
from ..operation import Operation


@beartype
class PassthroughOp(Operation):
    """Pass through a single parent counter unchanged."""
    
    def compute_num_states(self, parent_num_states):
        return parent_num_states[0]
    
    def decompose(self, state, parent_num_states):
        return (state,)


@beartype
def passthrough(counter: Counter_type, name: Optional[str] = None):
    """Create a passthrough counter that tracks its parent."""
    from ..counter import Counter
    result = Counter(_parents=(counter,), _op=PassthroughOp())
    if name is not None:
        result.name = name
    return result
