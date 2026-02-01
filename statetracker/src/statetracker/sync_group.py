"""SynchronizedGroup - A group of states that share the same value."""

from numbers import Integral

from .imports import Optional, State_type, beartype


@beartype
class SynchronizedGroup:
    """A group of synchronized states that share the same value.

    Owns tree traversal logic to ensure all sync peers are updated
    at every level of value propagation.

    States in a group may have different num_values. The group's num_values
    is the maximum across all states. When a value is set, each state only
    receives it if the value is within its valid range; otherwise it gets None.
    """

    def __init__(self, state: State_type):
        """Create a sync group containing a single state."""
        self._states: set = {state}
        self._num_values: Optional[int] = state.num_values  # Can be None for fixed states
        self._value: Optional[int] = None  # Logical value for the group

    @property
    def num_values(self) -> Optional[int]:
        """Number of values for states in this group (max across all states, None for fixed)."""
        return self._num_values

    @property
    def value(self) -> Optional[int]:
        """Logical value of this group (may be out of range for some states)."""
        return self._value

    def add(self, state: State_type) -> None:
        """Add a state to this group. Group num_values becomes the max (ignoring None)."""
        if self._num_values is None:
            self._num_values = state.num_values
        elif state.num_values is not None:
            self._num_values = max(self._num_values, state.num_values)
        self._states.add(state)
        state._synced_group = self

    def merge(self, other: "SynchronizedGroup") -> None:
        """Merge another group into this one. num_values becomes the max (ignoring None)."""
        if other is self:
            return
        if self._num_values is None:
            self._num_values = other._num_values
        elif other._num_values is not None:
            self._num_values = max(self._num_values, other._num_values)
        for state in other._states:
            state._synced_group = self
        self._states |= other._states

    def inactivate_trees(self) -> None:
        """Clear group value and all states in this sync group.

        Does NOT propagate to op-parents - setting a derived state to None
        should not affect its parent states.
        """
        self._value = None
        for state in self._states:
            state._value = None

    def set_inactivated_values_in_trees(self, val) -> None:
        """Set group value and propagate to states within their valid ranges."""
        # Track the logical value at the group level
        self._value = val

        for state in self._states:
            # Only set if value is in range for this state
            state_val = val if (val is not None and val < state.num_values) else None

            match (state._value, state_val):
                case (None, None):
                    pass
                case (None, Integral()):
                    state._value = state_val
                case (Integral(), Integral()) if state._value == state_val:
                    pass
                case (Integral(), None):
                    pass
                case _:
                    from .state import ConflictingValueAssignmentError

                    raise ConflictingValueAssignmentError(
                        f"State '{state.name}' received conflicting values: "
                        f"{state._value} vs {state_val}"
                    )

        # Propagate to parent states
        for state in self._states:
            if state._parents:
                if state._value is not None:
                    # Propagate decomposed values to parents
                    parent_num_values = tuple(p.num_values for p in state._parents)
                    parent_values = state._op.decompose(state._value, parent_num_values)
                    for parent, pv in zip(state._parents, parent_values):
                        parent._synced_group.set_inactivated_values_in_trees(pv)
                else:
                    # State is None - propagate None to parents (safe: match preserves existing values)
                    for parent in state._parents:
                        parent._synced_group.set_inactivated_values_in_trees(None)

    def __iter__(self):
        return iter(self._states)

    def __len__(self) -> int:
        return len(self._states)

    def __contains__(self, state: State_type) -> bool:
        return state in self._states
