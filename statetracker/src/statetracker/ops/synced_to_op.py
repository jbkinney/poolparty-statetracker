"""synced_to - Create or pair synced parent states."""
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


@beartype
def sync(child: State_type, parent: State_type) -> None:
    """Pair two existing states so parent receives child's value.
    
    Raises ValueError if num_values don't match.
    """
    if child.num_values != parent.num_values:
        raise ValueError(
            f"Cannot sync states with different num_values: "
            f"child has {child.num_values}, parent has {parent.num_values}"
        )
    parent._synced_child = child
    child._synced_parents.append(parent)
    parent._value = child._value
