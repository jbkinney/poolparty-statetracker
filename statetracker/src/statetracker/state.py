"""State class for composable iteration."""
from .imports import beartype, Sequence, Optional, Real, Integral, Operation_type, State_type, Union
from .manager import Manager

@beartype
class ConflictingValueAssignmentError(RuntimeError):
    """Raised when a state receives conflicting value assignments during propagation."""
    pass


@beartype
class State:
    """A state that can be iterated and composed with other states."""
    
    def __init__(self, 
                 num_values: Optional[Integral] = None, 
                 name: Optional[str] = None, 
                 value: Optional[Integral] = None,
                 iter_order: Optional[Real] = None, *, 
                 _parents: Optional[Sequence[State_type]] = None, 
                 _op: Optional[Operation_type] = None):
        """Create a state."""
        # Require an active Manager
        if Manager._active_manager is None:
            raise RuntimeError("State must be created within a Manager context")
        
        self._id = None
        self._name = name
        self._parents = tuple(_parents) if _parents else ()
        self._op = _op
        
        # Set iter_order
        if iter_order is None:
            if len(self._parents) > 0:
                iter_order = min(p.iter_order for p in self._parents)
            else:
                iter_order = 0
        self._iter_order = iter_order
        
        if _parents and _op:
            parent_num_values = tuple(p.num_values for p in _parents)
            self._num_values = _op.compute_num_states(parent_num_values)
        else:
            if num_values is None:
                raise ValueError("Leaf states require num_values")
            self._num_values = num_values
            
        # Register with Manager
        self._manager = Manager._active_manager
        self._manager.register(self)
        
        # Set value
        if value is not None:
            self.value = value
        elif not self._parents:
            # Leaf state: default to active at value 0
            self.value = 0
        else:
            # Derived state: default to inactive (no propagation to parents)
            self._value = None
    
    @property
    def iter_order(self):
        """Iteration order for this state."""
        return self._iter_order
    
    @iter_order.setter
    def iter_order(self, value: Real):
        """Set iteration order for this state."""
        self._iter_order = value
        
    @property
    def num_values(self):
        """Number of values this state can take (read-only)."""
        return self._num_values
    
    @property
    def id(self):
        """Unique ID assigned by Manager (None if not registered)."""
        return self._id
    
    @property
    def name(self):
        """Name of this counter."""
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = value
    
    def named(self, name):
        """Set name and return self for fluent chaining."""
        self._name = name
        return self
    
    @property
    def value(self):
        """Current value of this state."""
        return self._value
    
    @value.setter
    def value(self, val: Optional[Integral]):
        """Set value and propagate to parents with conflict detection."""
        if val is None:
            self._value = None
        else:
            self._inactivate_tree()
            self._set_inactivated_values_in_tree(val)
    
    def _set_inactivated_values_in_tree(self, val: Optional[Integral]):
        """Set value in pre-inactivated tree with conflict detection."""
        match (self._value, val):
            case (None, None):
                # Both inactive - nothing to do
                pass
            case (None, Integral()):
                # Setting active value on inactive state - normal case
                self._value = val
                if self._parents:
                    parent_num_values = tuple(p.num_values for p in self._parents)
                    parent_values = self._op.decompose(self._value, parent_num_values)
                    for parent, pv in zip(self._parents, parent_values):
                        parent._set_inactivated_values_in_tree(pv)
            case (Integral(), Integral()) if self._value == val:
                # Already set to same value - no conflict, do nothing
                pass
            case (Integral(), Integral()):
                # Different values - CONFLICT
                raise ConflictingValueAssignmentError(
                    f"State '{self.name}' received conflicting value assignments: "
                    f"already set to {self._value}, now attempting to set to {val}."
                )
            case (Integral(), None):
                # State already set by another path, this path doesn't need it
                # (happens with StackOp where inactive branches return None)
                pass
    
    def advance(self):
        """Advance to next value (wraps around)."""
        if self._value is None:
            raise RuntimeError("Cannot advance an inactive state (value=None)")
        self.value = (self._value + 1) % self._num_values
    
    def reset(self, value: Integral = 0):
        """Reset to specified value (default 0)."""
        self.value = value
    
    def _inactivate_tree(self):
        """Set this state and all parents to inactive value (None)."""
        self._value = None
        for parent in self._parents:
            parent._inactivate_tree()
    
    def is_active(self):
        """Return True if state is active (value is not None)."""
        return self._value is not None
    
    def __iter__(self):
        """Iterate through all values."""
        self.reset()
        for _ in range(self._num_values):
            yield self._value
            self.advance()
        self.reset()
    
    def __getitem__(self, key: Union[Integral, slice]):
        """Create sliced state: B = A[1:5] or A[::2] or A[::-1]."""
        from .ops import SliceOp
        # If key is an int, convert it to a slice for that single value
        if isinstance(key, int):
            key = slice(key, key + 1, 1)
        start, stop, step = key.indices(self._num_values)
        return State(_parents=(self,), _op=SliceOp(start, stop, step))
    
    def copy(self, name: Optional[str] = None):
        """Create a shallow copy with the same parents but a new State object."""
        if self._parents:
            new_state = State(_parents=self._parents, _op=self._op, name=name)
        else:
            new_state = State(self._num_values, name=name)
        new_state.value = self._value
        return new_state
    
    def deepcopy(self, name: Optional[str] = None):
        """Create a deep copy with all ancestors also copied."""
        if not self._parents:
            new_state = State(self._num_values, name=name)
        else:
            new_parents = tuple(p.deepcopy() for p in self._parents)
            new_state = State(_parents=new_parents, _op=self._op, name=name)
        new_state.value = self._value
        return new_state
    
    def __repr__(self):
        if self._parents:
            op_name = type(self._op).__name__
            return f"State(name={self._name!r}, id={self._id}, op={op_name}, num_values={self._num_values}, value={self._value}, iter_order={self._iter_order})"
        else:
            return f"State(name={self._name!r}, id={self._id}, num_values={self._num_values}, value={self._value}, iter_order={self._iter_order})"
    
    def print_dag(self, style: str = 'clean'):
        """Print the ASCII tree visualization rooted at this state.
        
        Args:
            style: Display style - 'clean' (default), 'minimal', or 'repr'.
        """
        from .text_viz import print_dag
        print_dag(self, style=style)

    def get_iteration_df(self, **kwargs):
        ancestors = self._manager.get_ancestors(self)
        df = self._manager.get_iteration_df(self, counters=ancestors, **kwargs)
        return df
    
    def get_ancestors(self):
        return self._manager.get_ancestors(self)
    
    def _has_auto_name(self) -> bool:
        """Return True if this state has an auto-generated name (State[N])."""
        import re
        return bool(re.match(r'^State\[\d+\]$', self._name or ''))
    
    def get_states(self, include_inactive: bool = True, named_only: bool = True) -> dict:
        """Return dict of {name: value} for this state and all ancestors.
        
        Args:
            include_inactive: If True (default), include states with None value.
                If False, only include states that are currently active.
            named_only: If True (default), exclude states with auto-generated 
                names (State[N]). If False, include all states.
        
        Returns:
            Dictionary mapping state names to their current values,
            in reverse topological order (derived states first, parents last).
        """
        from collections import deque
        
        # BFS for reverse topological order (self first, then parents, then grandparents...)
        ordered = []
        visited = set()
        queue = deque([self])
        
        while queue:
            state = queue.popleft()
            if state.id in visited:
                continue
            visited.add(state.id)
            ordered.append(state)
            queue.extend(state._parents)
        
        # Apply filters
        if named_only:
            ordered = [s for s in ordered if not s._has_auto_name()]
        if not include_inactive:
            ordered = [s for s in ordered if s.value is not None]
        
        return {s.name: s.value for s in ordered}
    
    def print_states(self, include_inactive: bool = True, named_only: bool = True) -> None:
        """Print current values of this state and its ancestors.
        
        Args:
            include_inactive: If True (default), include states with None value.
                If False, only include states that are currently active.
            named_only: If True (default), exclude states with auto-generated 
                names (State[N]). If False, include all states.
        """
        states = self.get_states(include_inactive=include_inactive, named_only=named_only)
        parts = [f"{name}={value}" for name, value in states.items()]
        print(", ".join(parts))