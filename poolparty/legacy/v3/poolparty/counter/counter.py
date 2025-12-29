"""
Counter class for composable iteration.

Counters can be composed via product (*), sum (+), and sync operations.
State flows from composite counters DOWN to their parent counters only.
"""
from .counter_manager import CounterManager
from .coops import MultiplyCoOp, SumCoOp, SliceCoOp, RepeatCoOp


class Counter:
    """
    A counter that can be iterated and composed with other counters.
    
    Leaf counters are created with just num_states:
        A = Counter(3, name='A')
    
    Composite counters are created via operations:
        C = A * B  # Product (binary)
        D = A + B  # Sum (binary)
        E = multiply_counters(A, B, C)  # Product (N-ary)
        F = sum_counters(A, B, C)  # Sum (N-ary)
        S = synchronize_counters(A, B)  # Sync
    
    Usage:
        for _ in C:
            print(f"A={A.state}, B={B.state}")
    """
    
    def __init__(self, num_states=None, name=None, *, _parents=None, _op=None):
        """
        Create a counter.
        
        For leaf counters, provide num_states.
        For composite counters, _parents and _op are set internally.
        
        Args:
            num_states: Number of states for leaf counters.
            name: Optional name for the counter.
            _parents: Internal - parent counters for composite counters.
            _op: Internal - operation for composite counters.
        """
        self._id = None  # Set by CounterManager.register()
        self._name = name
        self._state = 0
        self._parents = _parents or ()
        self._op = _op
        
        if _parents and _op:
            # Composite counter: compute num_states from parents
            parent_num_states = tuple(p.num_states for p in _parents)
            self._num_states = _op.compute_num_states(parent_num_states)
        else:
            # Leaf counter: use provided num_states
            if num_states is None:
                raise ValueError("Leaf counters require num_states")
            self._num_states = num_states
        
        # Auto-register with active manager if one exists
        if CounterManager._active_manager is not None:
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
        """Set name and return self for fluent chaining.
        
        Example:
            C = (A + B).named('C')
        """
        self._name = name
        return self
    
    @property
    def state(self):
        """Current state of this counter."""
        return self._state
    
    @state.setter
    def state(self, value):
        """Set state and propagate to parents."""
        if value == -1:
            self._state = -1
        else:
            self._state = value % self._num_states
        self._propagate_to_parents()
    
    def _propagate_to_parents(self):
        """Push state down to parents (recursively)."""
        if not self._parents:
            return
        
        parent_num_states = tuple(p.num_states for p in self._parents)
        parent_values = self._op.decompose(self._state, parent_num_states)
        
        for parent, value in zip(self._parents, parent_values):
            parent._state = value
            parent._propagate_to_parents()
    
    def advance(self):
        """Advance to next state (wraps around)."""
        if self._state == -1:
            raise RuntimeError("Cannot advance an inactive counter (state=-1)")
        self.state = self._state + 1
    
    def reset(self, state: int = 0):
        """Reset to specified state (default 0)."""
        self.state = state
    
    def inactive(self):
        """Set counter to inactive state (-1)."""
        self.state = -1
    
    def is_active(self):
        """Return True if counter is active (state != -1)."""
        return self._state != -1
    
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
        return Counter(_parents=(self, other), _op=SumCoOp())
    
    def __getitem__(self, key):
        """Create sliced counter: B = A[1:5] or A[::2] or A[::-1]."""
        if not isinstance(key, slice):
            raise TypeError("Counter indices must be slices, not integers")
        start, stop, step = key.indices(self._num_states)
        return Counter(_parents=(self,), _op=SliceCoOp(start, stop, step))
    
    def copy(self, name=None):
        """Create a shallow copy with the same parents but a new Counter object.
        
        The copy shares parents with the original. Iterating the copy will
        affect the shared parent counters.
        
        Args:
            name: Optional name for the copy.
        
        Returns:
            A new Counter object.
        """
        if self._parents:
            new_counter = Counter(_parents=self._parents, _op=self._op)
        else:
            new_counter = Counter(self._num_states)
        
        if name is not None:
            new_counter._name = name
        new_counter.state = self._state
        return new_counter
    
    def deepcopy(self, name=None):
        """Create a deep copy with all ancestors also copied.
        
        The copy is completely independent from the original.
        
        Args:
            name: Optional name for the copy.
        
        Returns:
            A new Counter object with copied ancestors.
        """
        if not self._parents:
            new_counter = Counter(self._num_states)
        else:
            new_parents = tuple(p.deepcopy() for p in self._parents)
            new_counter = Counter(_parents=new_parents, _op=self._op)
        
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
    
    def print_tree(self):
        """Print the ASCII tree visualization rooted at this counter."""
        self._print_subtree("", is_last=True, is_root=True)
    
    def _format_node(self):
        """Format this counter node for display."""
        name = self.name if self.name else f"id_{self.id}"
        n = self.num_states
        
        if self._parents:
            # Composite counter - get operation type name
            op_name = type(self._op).__name__
            # Remove 'CoOp' suffix for cleaner display
            if op_name.endswith('CoOp'):
                op_name = op_name[:-4]
            return f"{name} [{op_name}, n={n}]"
        else:
            # Leaf counter
            return f"{name} [Leaf, n={n}]"
    
    def _print_subtree(self, prefix, is_last, is_root=False):
        """Recursively print this counter and its parents as a tree."""
        if is_root:
            connector = ""
            child_prefix = ""
        else:
            connector = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "
        
        print(f"{prefix}{connector}{self._format_node()}")
        
        for i, parent in enumerate(self._parents):
            is_last_parent = (i == len(self._parents) - 1)
            parent._print_subtree(prefix + child_prefix, is_last_parent)