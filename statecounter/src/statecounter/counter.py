"""Counter class for composable iteration."""
from .imports import beartype, Sequence, Optional, Real, Integral, Operation_type, Counter_type, Union
from .manager import Manager

@beartype
class ConflictingStateAssignmentError(RuntimeError):
    """Raised when a counter receives conflicting state assignments during propagation."""
    pass


@beartype
class Counter:
    """A counter that can be iterated and composed with other counters."""
    
    def __init__(self, 
                 num_states: Optional[Integral] = None, 
                 name: Optional[str] = None, 
                 state: Optional[Integral] = None,
                 iter_order: Optional[Real] = None, *, 
                 _parents: Optional[Sequence[Counter_type]] = None, 
                 _op: Optional[Operation_type] = None):
        """Create a counter."""
        # Require an active Manager
        if Manager._active_manager is None:
            raise RuntimeError("Counter must be created within a Manager context")
        
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
            parent_num_states = tuple(p.num_states for p in _parents)
            self._num_states = _op.compute_num_states(parent_num_states)
        else:
            if num_states is None:
                raise ValueError("Leaf counters require num_states")
            self._num_states = num_states
            
        # Register with Manager
        self._manager = Manager._active_manager
        self._manager.register(self)
        
        # Set state
        if state is not None:
            self.state = state
        elif not self._parents:
            # Leaf counter: default to active at state 0
            self.state = 0
        else:
            # Derived counter: default to inactive (no propagation to parents)
            self._state = None
    
    @property
    def iter_order(self):
        """Iteration order for this counter."""
        return self._iter_order
    
    @iter_order.setter
    def iter_order(self, value: Real):
        """Set iteration order for this counter."""
        self._iter_order = value
        
    @property
    def num_states(self):
        """Number of states this counter can take (read-only)."""
        return self._num_states
    
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
    def state(self):
        """Current state of this counter."""
        return self._state
    
    @state.setter
    def state(self, value: Optional[Integral]):
        """Set state and propagate to parents with conflict detection."""
        if value is None:
            self._state = None
        else:
            self._inactivate_tree()
            self._set_inactivated_states_in_tree(value)
    
    def _set_inactivated_states_in_tree(self, value: Optional[Integral]):
        """Set state in pre-inactivated tree with conflict detection."""
        match (self._state, value):
            case (None, None):
                # Both inactive - nothing to do
                pass
            case (None, Integral()):
                # Setting active state on inactive counter - normal case
                self._state = value
                if self._parents:
                    parent_num_states = tuple(p.num_states for p in self._parents)
                    parent_values = self._op.decompose(self._state, parent_num_states)
                    for parent, pv in zip(self._parents, parent_values):
                        parent._set_inactivated_states_in_tree(pv)
            case (Integral(), Integral()) if self._state == value:
                # Already set to same value - no conflict, do nothing
                pass
            case (Integral(), Integral()):
                # Different values - CONFLICT
                raise ConflictingStateAssignmentError(
                    f"Counter '{self.name}' received conflicting state assignments: "
                    f"already set to {self._state}, now attempting to set to {value}."
                )
            case (Integral(), None):
                # Counter already set by another path, this path doesn't need it
                # (happens with StackOp where inactive branches return None)
                pass
    
    def advance(self):
        """Advance to next state (wraps around)."""
        if self._state is None:
            raise RuntimeError("Cannot advance an inactive counter (state=None)")
        self.state = (self._state + 1) % self._num_states
    
    def reset(self, state: Integral = 0):
        """Reset to specified state (default 0)."""
        self.state = state
    
    def _inactivate_tree(self):
        """Set this counter and all parents to inactive state (None)."""
        self._state = None
        for parent in self._parents:
            parent._inactivate_tree()
    
    def is_active(self):
        """Return True if counter is active (state is not None)."""
        return self._state is not None
    
    def __iter__(self):
        """Iterate through all states."""
        self.reset()
        for _ in range(self._num_states):
            yield self._state
            self.advance()
        self.reset()
    
    def __getitem__(self, key: Union[Integral, slice]):
        """Create sliced counter: B = A[1:5] or A[::2] or A[::-1]."""
        from .ops import SliceOp
        # If key is an int, convert it to a slice for that single state
        if isinstance(key, int):
            key = slice(key, key + 1, 1)
        start, stop, step = key.indices(self._num_states)
        return Counter(_parents=(self,), _op=SliceOp(start, stop, step))
    
    def copy(self, name: Optional[str] = None):
        """Create a shallow copy with the same parents but a new Counter object."""
        if self._parents:
            new_counter = Counter(_parents=self._parents, _op=self._op, name=name)
        else:
            new_counter = Counter(self._num_states, name=name)
        new_counter.state = self._state
        return new_counter
    
    def deepcopy(self, name: Optional[str] = None):
        """Create a deep copy with all ancestors also copied."""
        if not self._parents:
            new_counter = Counter(self._num_states, name=name)
        else:
            new_parents = tuple(p.deepcopy() for p in self._parents)
            new_counter = Counter(_parents=new_parents, _op=self._op, name=name)
        new_counter.state = self._state
        return new_counter
    
    def __repr__(self):
        if self._parents:
            op_name = type(self._op).__name__
            return f"Counter(name={self._name!r}, id={self._id}, op={op_name}, num_states={self._num_states}, state={self._state}, iter_order={self._iter_order})"
        else:
            return f"Counter(name={self._name!r}, id={self._id}, num_states={self._num_states}, state={self._state}, iter_order={self._iter_order})"
    
    def print_dag(self, style: str = 'clean'):
        """Print the ASCII tree visualization rooted at this counter.
        
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