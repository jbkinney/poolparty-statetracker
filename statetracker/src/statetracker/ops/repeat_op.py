"""RepeatOp - Repeat a single state N times."""
from ..imports import beartype, Optional, Integral, State_type
from ..operation import Operation


@beartype
def repeat(state: State_type, times: Integral, name: Optional[str] = None):
    """
    Create a State that repeats the values of another State a specified number of times.

    Parameters
    ----------
    state : State_type
        The State to repeat.
    times : Integral
        Number of times to repeat the entire sequence of the parent's values.
    name : Optional[str], default=None
        Name for the resulting repeated State.

    Returns
    -------
    State_type
        A State whose values iterate through the parent's values, repeated the specified number of times.
    """
    from ..state import State
    if times < 1:
        raise ValueError("times must be at least 1")    
    result = State(_parents=(state,), _op=RepeatOp(times), name=name)
    return result


@beartype
class RepeatOp(Operation):
    """Repeat a single state N times."""
    
    def __init__(self, times: Integral):
        self.times = times
    
    def compute_num_states(self, parent_num_values):
        return parent_num_values[0] * self.times
    
    def decompose(self, value, parent_num_values):
        if value is None:
            return (None,)
        return (value % parent_num_values[0],)
