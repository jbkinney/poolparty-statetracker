"""Party class - context manager for building and executing sequence libraries."""
import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
import statetracker as st
from .types import Pool_type, Operation_type, Optional, beartype, Union, Any
from .codon_table import CodonTable
from . import dna_utils
from .marker import Marker

_active_party: Optional["Party"] = None
_default_party: Optional["Party"] = None


@beartype
def get_active_party() -> Optional["Party"]:
    """Get the currently active Party context, or None if not in a context."""
    return _active_party


@beartype
def init(
    genetic_code: Union[str, dict] = 'standard',
) -> "Party":
    """Initialize (or reset) the default Party, clearing all registered pools/operations/markers."""
    global _active_party, _default_party
    # Exit current default party if active
    if _default_party is not None and _default_party._is_active:
        _default_party._counter_manager.__exit__(None, None, None)
        _default_party._is_active = False
    # Create new default party
    _default_party = Party(genetic_code=genetic_code)
    _default_party._counter_manager.__enter__()
    _default_party._is_active = True
    _active_party = _default_party
    # Set default parameter values
    _default_party.set_default('remove_marker', False)
    return _default_party


def _init_default_party() -> None:
    """Initialize the default party on module import (called from __init__.py)."""
    global _default_party
    if _default_party is None:
        init()


@beartype
def clear_pools() -> None:
    """Clear all pools, operations, and markers from the active Party without resetting highlights."""
    party = get_active_party()
    if party is None:
        raise RuntimeError("No active Party context.")
    party.clear_pools()

@beartype
class Party:
    """Context manager for building and executing sequence libraries."""
    
    def __init__(
        self,
        genetic_code: Union[str, dict] = 'standard',
    ) -> None:
        self._operations: list = []
        self._outputs: dict[str, Pool_type] = {}
        self._is_active: bool = False
        self._previous_party: Optional["Party"] = None
        self._counter_manager: st.Manager = st.Manager()
        self._next_pool_id: int = 0
        self._next_op_id: int = 0
        self._next_marker_id: int = 0
        # Track pools and operations by ID (list) and name (dict)
        self._pools_by_id: list[Pool_type] = []
        self._ops_by_id: list[Operation_type] = []
        self._pools_by_name: dict[str, Pool_type] = {}
        self._ops_by_name: dict[str, Operation_type] = {}
        # Track markers by ID (list) and name (dict)
        self._markers_by_id: list[Marker] = []
        self._markers_by_name: dict[str, Marker] = {}
        # Build codon table for ORF operations
        self._codon_table: CodonTable = CodonTable(genetic_code)
        # Default parameter values for operations
        self._defaults: dict[str, Any] = {}
    
    def _get_next_pool_id(self) -> int:
        """Get the next unique pool ID."""
        id_ = self._next_pool_id
        self._next_pool_id += 1
        return id_

    def _get_next_op_id(self) -> int:
        """Get the next unique operation ID."""
        id_ = self._next_op_id
        self._next_op_id += 1
        return id_
    
    @property
    def state_manager(self) -> st.Manager:
        """Access the statetracker Manager for debugging state iteration."""
        return self._counter_manager
    
    @property
    def counter_manager(self) -> st.Manager:
        """Deprecated: Use state_manager instead. Access the statetracker Manager for debugging state iteration."""
        return self._counter_manager
    
    @property
    def codon_table(self) -> CodonTable:
        """Access the CodonTable for ORF operations."""
        return self._codon_table
    
    def set_genetic_code(self, genetic_code: Union[str, dict]) -> None:
        """Set or change the genetic code used for ORF operations."""
        self._codon_table = CodonTable(genetic_code)
    
    def set_default(self, key: str, value: Any) -> None:
        """Set a default parameter value for operations in this party."""
        self._defaults[key] = value
    
    def get_default(self, key: str, fallback: Any = None) -> Any:
        """Get a default parameter value, or fallback if not set."""
        return self._defaults.get(key, fallback)
    
    def load_defaults(self, filepath: str) -> None:
        """Load default parameter values from a TOML file."""
        with open(filepath, 'rb') as f:
            defaults = tomllib.load(f)
        self._defaults.update(defaults)
    
    def get_effective_seq_length(self, seq: str) -> int:
        """Get effective sequence length (DNA characters only, excluding markers)."""
        return dna_utils.get_seq_length(seq)
    
    def get_length_without_markers(self, seq: str) -> int:
        """Get sequence length excluding only marker tags (includes all chars)."""
        return dna_utils.get_length_without_markers(seq)
    
    def get_molecular_positions(self, seq: str) -> list[int]:
        """Get raw string positions of valid DNA characters, excluding marker interiors."""
        return dna_utils.get_molecular_positions(seq)
    
    def __enter__(self) -> "Party":
        """Enter the Party context, saving any previous active party."""
        global _active_party
        # Save previous party to restore on exit
        self._previous_party = _active_party
        _active_party = self
        self._is_active = True
        self._counter_manager.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the Party context, restoring the previous party."""
        global _active_party
        self._counter_manager.__exit__(exc_type, exc_val, exc_tb)
        self._is_active = False
        # Restore previous party (could be default or another explicit party)
        _active_party = self._previous_party
        self._previous_party = None
    
    def _validate_pool_name(self, name: str, pool: Optional[Pool_type] = None) -> str:
        """Validate that a pool name is unique."""
        existing = self._pools_by_name.get(name)
        if existing is not None and existing is not pool:
            raise ValueError(f"Pool name '{name}' already exists")
        return name
    
    def _validate_op_name(self, name: str, op: Optional[Operation_type] = None) -> str:
        """Validate that an operation name is unique."""
        existing = self._ops_by_name.get(name)
        if existing is not None and existing is not op:
            raise ValueError(f"Operation name '{name}' already exists")
        return name
    
    def _register_pool(self, pool: Pool_type) -> None:
        """Register a pool with this party."""
        self._pools_by_id.append(pool)
        self._pools_by_name[pool.name] = pool
    
    def _update_pool_name(self, pool: Pool_type, old_name: str, new_name: str) -> None:
        """Update a pool's name in the tracking dict."""
        if old_name in self._pools_by_name:
            del self._pools_by_name[old_name]
        self._pools_by_name[new_name] = pool
    
    def _register_operation(self, operation: Operation_type) -> None:
        """Register an operation with this party."""
        if operation not in self._operations:
            self._operations.append(operation)
        self._ops_by_id.append(operation)
        self._ops_by_name[operation.name] = operation
    
    def _update_op_name(self, op: Operation_type, old_name: str, new_name: str) -> None:
        """Update an operation's name in the tracking dict."""
        if old_name in self._ops_by_name:
            del self._ops_by_name[old_name]
        self._ops_by_name[new_name] = op
    
    def get_pool_by_id(self, id_: int) -> Pool_type:
        """Get a pool by its ID."""
        return self._pools_by_id[id_]
    
    def get_pool_by_name(self, name: str) -> Pool_type:
        """Get a pool by its name."""
        return self._pools_by_name[name]
    
    def get_op_by_id(self, id_: int) -> Operation_type:
        """Get an operation by its ID."""
        return self._ops_by_id[id_]
    
    def get_op_by_name(self, name: str) -> Operation_type:
        """Get an operation by its name."""
        return self._ops_by_name[name]
    
    def _get_next_marker_id(self) -> int:
        """Get the next unique marker ID."""
        id_ = self._next_marker_id
        self._next_marker_id += 1
        return id_
    
    def register_marker(self, name: str, seq_length: Optional[int]) -> Marker:
        """
        Register a marker with this party.
        
        If a marker with the same name already exists:
        - If it has the same seq_length, return the existing marker
        - If it has a different seq_length, raise ValueError
        
        Parameters
        ----------
        name : str
            The marker name.
        seq_length : Optional[int]
            The expected content length (None for variable, 0 for zero-length).
        
        Returns
        -------
        Marker
            The registered marker (existing or newly created).
        
        Raises
        ------
        ValueError
            If a marker with the same name but different seq_length exists.
        """
        existing = self._markers_by_name.get(name)
        if existing is not None:
            if existing.seq_length == seq_length:
                return existing
            else:
                # Format lengths for error message
                existing_len = 'variable' if existing.seq_length is None else str(existing.seq_length)
                new_len = 'variable' if seq_length is None else str(seq_length)
                raise ValueError(
                    f"Marker '{name}' already registered with seq_length={existing_len}, "
                    f"cannot re-register with seq_length={new_len}. "
                    f"Marker lengths must be consistent within a Party."
                )
        
        # Create and register new marker
        marker = Marker(name=name, seq_length=seq_length, _id=self._get_next_marker_id())
        self._markers_by_id.append(marker)
        self._markers_by_name[name] = marker
        return marker
    
    def get_marker_by_id(self, id_: int) -> Marker:
        """Get a marker by its ID."""
        if id_ < 0 or id_ >= len(self._markers_by_id):
            raise ValueError(f"No marker with ID {id_}")
        return self._markers_by_id[id_]
    
    def get_marker_by_name(self, name: str) -> Marker:
        """Get a marker by its name."""
        marker = self._markers_by_name.get(name)
        if marker is None:
            available = list(self._markers_by_name.keys())
            if available:
                raise ValueError(f"Marker '{name}' not found. Available: {available}")
            else:
                raise ValueError(f"Marker '{name}' not found. No markers registered.")
        return marker
    
    def get_marker(self, name: str) -> Marker:
        """Get a registered marker by name. Alias for get_marker_by_name."""
        return self.get_marker_by_name(name)
    
    def has_marker(self, name: str) -> bool:
        """Check if a marker with the given name is registered."""
        return name in self._markers_by_name
    
    def clear_pools(self) -> None:
        """Clear all pools, operations, and markers without resetting highlights.
        
        Unlike init(), this preserves:
        - Genetic code settings (_codon_table)
        - Default parameter values (_defaults)
        """
        # Clear pool tracking
        self._pools_by_id.clear()
        self._pools_by_name.clear()
        # Clear operation tracking
        self._operations.clear()
        self._ops_by_id.clear()
        self._ops_by_name.clear()
        # Clear marker tracking
        self._markers_by_id.clear()
        self._markers_by_name.clear()
        # Reset ID counters
        self._next_pool_id = 0
        self._next_op_id = 0
        self._next_marker_id = 0
        # Clear outputs
        self._outputs.clear()
        # Reset counter manager to clear counter state
        if self._is_active:
            self._counter_manager.__exit__(None, None, None)
            self._counter_manager = st.Manager()
            self._counter_manager.__enter__()
        else:
            self._counter_manager = st.Manager()
    
    def output(self, pool: Pool_type, name: Optional[str] = None) -> None:
        """Mark a pool as an output of this library."""
        if name is None:
            name = pool.name or f"output_{len(self._outputs)}"
        self._outputs[name] = pool
    
    def __repr__(self) -> str:
        return f"Party(outputs={list(self._outputs.keys())})"
    
    def print_graph(self, style: str = 'clean') -> None:
        """Print an ASCII tree visualization of the Pool-Operation computation graph.
        
        Shows pools (places) with parentheses and operations (transitions) with brackets,
        similar to a Petri net diagram. Root pools (not consumed by other operations)
        are printed first, with their upstream DAGs.
        
        Args:
            style: Display style - 'clean' (default), 'minimal', or 'repr'.
                - 'clean': Shows names with key attributes
                    Pool: (name) pool: n=num_states
                    Op: [name] op: factory_name, mode, n=num_states
                - 'minimal': Shows just names
                    Pool: (name)
                    Op: [name]
                - 'repr': Shows full repr() of each object
        """
        from .text_viz import print_pool_graph
        print_pool_graph(self._pools_by_id, self._ops_by_id, style=style)