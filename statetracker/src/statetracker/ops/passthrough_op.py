"""PassthroughOp - Pass through a single parent state unchanged."""
from ..imports import beartype, Optional, State_type
from ..operation import Operation


@beartype
def passthrough(state: State_type, name: Optional[str] = None):
    """
    Create a State that passes through its parent's values unchanged.

    Parameters
    ----------
    state : State_type
        Parent State whose values will be tracked.
    name : Optional[str], default=None
        Optional name for the resulting passthrough State.

    Returns
    -------
    State_type
        A State that mirrors the values of its parent.
    """
    from ..state import State
    result = State(_parents=(state,), _op=PassthroughOp(), name=name)
    return result


@beartype
class PassthroughOp(Operation):
    """Pass through a single parent state unchanged."""
    
    def compute_num_states(self, parent_num_values):
        return parent_num_values[0]
    
    def decompose(self, value, parent_num_values):
        return (value,)
