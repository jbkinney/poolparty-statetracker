"""SliceOp - Slice a state to select a subset of values."""
from ..imports import beartype, Optional, Integral, State_type
from ..operation import Operation
import builtins


@beartype
def slice(state: State_type, start: Optional[Integral] = None, stop: Optional[Integral] = None, step: Optional[Integral] = None, name: Optional[str] = None):
    """
    Create a State whose values correspond to a slice of the parent State.

    Parameters
    ----------
    state : State_type
        The State to be sliced.
    start : Optional[Integral], default=None
        Start index of the slice (inclusive).
    stop : Optional[Integral], default=None
        Stop index of the slice (exclusive).
    step : Optional[Integral], default=None
        Step size for the slice.
    name : Optional[str], default=None
        Name for the resulting sliced State.

    Returns
    -------
    State_type
        A State whose values correspond to the specified slice of the parent State's values.
    """
    from ..state import State
    s = builtins.slice(start, stop, step)
    start_norm, stop_norm, step_norm = s.indices(state.num_values)
    result = State(_parents=(state,), _op=SliceOp(start_norm, stop_norm, step_norm), name=name)
    return result


@beartype
class SliceOp(Operation):
    """Slice a state to select a subset of values."""
    
    def __init__(self, start: Integral, stop: Integral, step: Integral):
        self.start = start
        self.stop = stop
        self.step = step
    
    def compute_num_states(self, parent_num_values):
        return len(range(self.start, self.stop, self.step))
    
    def decompose(self, value, parent_num_values):
        if value is None:
            return (None,)
        parent_value = self.start + (value * self.step)
        return (parent_value,)
