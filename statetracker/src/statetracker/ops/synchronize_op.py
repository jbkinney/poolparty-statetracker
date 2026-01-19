"""SynchronizeOp - Keep N states in lockstep."""
from ..imports import beartype, Sequence, Optional, State_type
from ..operation import Operation


@beartype
def sync(states: Sequence[State_type], name: Optional[str] = None):
    """
    Create a State that synchronizes the values of multiple parent States.

    Parameters
    ----------
    states : Sequence[State_type]
        Sequence of parent States to synchronize.
    name : Optional[str], default=None
        Optional name for the resulting synchronized State.

    Returns
    -------
    State_type
        A State whose values are the same as all parent States' values.
    """
    from ..state import State
    if len(states) == 0:
        result = State(1)
    else:
        result = State(_parents=states, _op=SyncOp(), name=name)
    return result


@beartype
class SyncOp(Operation):
    """Keep N states in lockstep."""
    
    def compute_num_states(self, parent_num_values):
        if len(parent_num_values) == 0:
            return 1
        if len(set(parent_num_values)) != 1:
            raise ValueError(
                f"Cannot sync states with different num_values: {parent_num_values}"
            )
        return parent_num_values[0]
    
    def decompose(self, value, parent_num_values):
        return tuple(value for _ in parent_num_values)
