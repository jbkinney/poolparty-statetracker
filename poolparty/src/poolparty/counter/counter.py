"""Counter class for composable iteration."""
from .counter_manager import CounterManager
from .coops import MultiplyCoOp, SumCoOp, SliceCoOp, RepeatCoOp


class ConflictingStateAssignmentError(RuntimeError):
    """Raised when a counter receives conflicting state assignments during propagation."""
    pass


class Counter:
    """A counter that can be iterated and composed with other counters."""
    
    def __init__(self, num_states=None, name=None, *, _parents=None, _op=None, iteration_order=None, deduplicate_parents=True):
        """Create a counter."""
        # Require an active CounterManager
        if CounterManager._active_manager is None:
            raise RuntimeError("Counter must be created within a CounterManager context")
        
        self._id = None
        self._name = name
        self._state = 0
        
        # Process _parents: optionally uniquify by object identity, validate registration, sort
        if _parents:
            # Optionally uniquify by object identity (preserves order of first occurrence)
            if deduplicate_parents:
                _parents = tuple(dict.fromkeys(_parents))
            # Validate all parents are registered
            for p in _parents:
                if p._id is None:
                    raise ValueError(f"Parent counter {p!r} must be registered before use")
            # Sort by (iteration_order, _id)
            _parents = tuple(sorted(_parents, key=lambda p: (p.iteration_order, p._id)))
        
        self._parents = _parents or ()
        self._op = _op
        
        # Set iteration_order
        if iteration_order is not None:
            self.iteration_order = iteration_order
        elif self._parents:
            self.iteration_order = min(p.iteration_order for p in self._parents)
        else:
            self.iteration_order = 0
        
        if _parents and _op:
            parent_num_states = tuple(p.num_states for p in _parents)
            self._num_states = _op.compute_num_states(parent_num_states)
        else:
            if num_states is None:
                raise ValueError("Leaf counters require num_states")
            self._num_states = num_states
        
        # Register with the active manager
        CounterManager._active_manager.register(self)
    
    @property
    def num_states(self):
        """Number of states this counter can take (read-only)."""
        return self._num_states
    
    @property
    def id(self):
        """Unique ID assigned by CounterManager (None if not registered)."""
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
    def state(self, value):
        """Set state and propagate to parents with conflict detection."""
        self._inactivate_tree()
        if value is not None:
            self._set_inactivated_states_in_tree(value)
    
    def _set_inactivated_states_in_tree(self, value):
        """Set state in pre-inactivated tree with conflict detection."""
        match (self._state, value):
            case (None, None):
                # Both inactive - nothing to do
                pass
            case (None, int()):
                # Setting active state on inactive counter - normal case
                self._state = value
                if self._parents:
                    parent_num_states = tuple(p.num_states for p in self._parents)
                    parent_values = self._op.decompose(self._state, parent_num_states)
                    for parent, pv in zip(self._parents, parent_values):
                        parent._set_inactivated_states_in_tree(pv)
            case (int(), int()) if self._state == value:
                # Already set to same value - no conflict, do nothing
                pass
            case (int(), int()):
                # Different values - CONFLICT
                raise ConflictingStateAssignmentError(
                    f"Counter '{self.name}' received conflicting state assignments: "
                    f"already set to {self._state}, now attempting to set to {value}."
                )
            case (int(), None):
                # Counter already set by another path, this path doesn't need it
                # (happens with SumCoOp where inactive branches return None)
                pass
    
    def advance(self):
        """Advance to next state (wraps around)."""
        if self._state is None:
            raise RuntimeError("Cannot advance an inactive counter (state=None)")
        self.state = (self._state + 1) % self._num_states
    
    def reset(self, state: int = 0):
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
    
    def __mul__(self, other):
        """Create product counter (C = A * B) or repeat counter (B = A * 3)."""
        if isinstance(other, Counter):
            return Counter(_parents=(self, other), _op=MultiplyCoOp())
        elif isinstance(other, int):
            if other < 1:
                raise ValueError("Repeat count must be at least 1")
            return Counter(_parents=(self,), _op=RepeatCoOp(other))
        return NotImplemented
    
    def __rmul__(self, other):
        """Create repeat counter: B = 3 * A."""
        if isinstance(other, int):
            if other < 1:
                raise ValueError("Repeat count must be at least 1")
            return Counter(_parents=(self,), _op=RepeatCoOp(other))
        return NotImplemented
    
    def __add__(self, other):
        """Create sum counter: D = A + B."""
        if not isinstance(other, Counter):
            return NotImplemented
        return Counter(_parents=(self, other), _op=SumCoOp(), deduplicate_parents=False)
    
    def __getitem__(self, key):
        """Create sliced counter: B = A[1:5] or A[::2] or A[::-1]."""
        if not isinstance(key, slice):
            raise TypeError("Counter indices must be slices, not integers")
        start, stop, step = key.indices(self._num_states)
        return Counter(_parents=(self,), _op=SliceCoOp(start, stop, step))
    
    def copy(self, name=None):
        """Create a shallow copy with the same parents but a new Counter object."""
        if self._parents:
            new_counter = Counter(_parents=self._parents, _op=self._op, iteration_order=self.iteration_order)
        else:
            new_counter = Counter(self._num_states, iteration_order=self.iteration_order)
        if name is not None:
            new_counter._name = name
        new_counter.state = self._state
        return new_counter
    
    def deepcopy(self, name=None):
        """Create a deep copy with all ancestors also copied."""
        if not self._parents:
            new_counter = Counter(self._num_states, iteration_order=self.iteration_order)
        else:
            new_parents = tuple(p.deepcopy() for p in self._parents)
            new_counter = Counter(_parents=new_parents, _op=self._op, iteration_order=self.iteration_order)
        if name is not None:
            new_counter._name = name
        new_counter.state = self._state
        return new_counter
    
    def __repr__(self):
        if self._parents:
            op_name = type(self._op).__name__
            return f"Counter(name={self._name!r}, id={self._id}, op={op_name}, num_states={self._num_states}, state={self._state})"
        else:
            return f"Counter(name={self._name!r}, id={self._id}, num_states={self._num_states}, state={self._state})"
    
    def print_tree(self, style: str = 'clean'):
        """Print the ASCII tree visualization rooted at this counter.
        
        Args:
            style: Display style - 'clean' (default), 'minimal', or 'repr'.
        """
        from ..text_viz import print_counter_tree
        print_counter_tree(self, style=style)
