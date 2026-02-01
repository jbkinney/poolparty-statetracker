"""StackOp - Disjoint union of N states."""

from ..imports import Optional, Sequence, State_type, beartype
from ..operation import Operation


@beartype
def stack(states: Sequence[State_type], name: Optional[str] = None):
    """
    Create a State representing the disjoint union of the provided States.

    Parameters
    ----------
    states : Sequence[State_type]
        Sequence of parent States to combine into a disjoint union State.
    name : Optional[str], default=None
        Optional name for the resulting disjoint union State.

    Returns
    -------
    State_type
        A State whose values correspond to the disjoint union of the input States' values.
    """
    from ..state import State

    if len(states) == 0:
        result = State(0)
    else:
        result = State(_parents=states, _op=StackOp(), name=name)
    return result


@beartype
class StackOp(Operation):
    def compute_num_states(self, parent_num_values):
        # Treat None (fixed states) as 1
        return sum(n if n is not None else 1 for n in parent_num_values)

    def decompose(self, value, parent_num_values):
        if value is None:
            return tuple(None for _ in parent_num_values)
        cumsum = 0
        for i, n in enumerate(parent_num_values):
            effective_n = n if n is not None else 1
            if value < cumsum + effective_n:
                return tuple(
                    value - cumsum if j == i else None for j in range(len(parent_num_values))
                )
            cumsum += effective_n
        raise ValueError(f"Invalid value {value}")
