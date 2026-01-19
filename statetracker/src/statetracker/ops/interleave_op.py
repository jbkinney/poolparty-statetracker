"""InterleaveOp - Interleave values from N states."""
from ..imports import beartype, Sequence, Optional, State_type
from ..operation import Operation


@beartype
def interleave(states: Sequence[State_type], name: Optional[str] = None):
    """
    Create a State that interleaves the values from multiple State objects.

    Parameters
    ----------
    states : Sequence[State_type]
        Sequence of State instances to interleave, each with equal num_values.
    name : Optional[str], default=None
        Optional name for the resulting interleaved State.

    Returns
    -------
    State_type
        A State whose values interleave across the provided states.
    """
    from ..state import State
    if len(states) < 2:
        raise ValueError("interleave() requires at least 2 states")
    return State(_parents=states, _op=InterleaveOp(), name=name)


@beartype
class InterleaveOp(Operation):
    """Interleave values from N states with equal num_values."""
    
    def compute_num_states(self, parent_num_values):
        if len(set(parent_num_values)) != 1:
            raise ValueError(
                f"Cannot interleave states with different num_values: {parent_num_values}"
            )
        return parent_num_values[0] * len(parent_num_values)
    
    def decompose(self, value, parent_num_values):
        if value is None:
            return tuple(None for _ in parent_num_values)
        k = len(parent_num_values)
        active_idx = value % k
        parent_value = value // k
        return tuple(parent_value if i == active_idx else None for i in range(k))
