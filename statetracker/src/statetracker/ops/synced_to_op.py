"""synced_to - Create a synced parent state."""
from ..imports import beartype, Optional, State_type


@beartype
def synced_to(child_state: State_type, name: Optional[str] = None) -> State_type:
    """Create a synced parent state that receives the child's value.
    
    The synced parent appears below child_state in the DAG but doesn't
    contribute to its value computation.
    """
    from ..state import State
    synced = State(num_values=child_state.num_values, name=name)
    synced._synced_child = child_state
    child_state._synced_parents.append(synced)
    synced._value = child_state._value
    return synced
