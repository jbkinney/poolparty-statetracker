"""Party class - context manager for building and executing sequence libraries."""
from typing import Union
import statecounter as sc
from .types import Pool_type, Operation_type, Optional, beartype
from .codon_table import CodonTable
from .alphabet import Alphabet, get_alphabet

_active_party: Optional["Party"] = None


@beartype
def get_active_party() -> Optional["Party"]:
    """Get the currently active Party context, or None if not in a context."""
    return _active_party

@beartype
class Party:
    """Context manager for building and executing sequence libraries."""
    
    def __init__(
        self,
        alphabet: Union[str, Alphabet] = 'dna',
        genetic_code: Union[str, dict] = 'standard',
    ) -> None:
        self._operations: list = []
        self._outputs: dict[str, Pool_type] = {}
        self._is_active: bool = False
        self._counter_manager: sc.Manager = sc.Manager()
        self._next_pool_id: int = 0
        self._next_op_id: int = 0
        # Track pools and operations by ID (list) and name (dict)
        self._pools_by_id: list[Pool_type] = []
        self._ops_by_id: list[Operation_type] = []
        self._pools_by_name: dict[str, Pool_type] = {}
        self._ops_by_name: dict[str, Operation_type] = {}
        # Build alphabet for sequence operations
        if isinstance(alphabet, str):
            self._alphabet: Alphabet = get_alphabet(alphabet)
        else:
            self._alphabet = alphabet
        # Build codon table for ORF operations
        self._codon_table: CodonTable = CodonTable(genetic_code)
    
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
    def counter_manager(self) -> sc.Manager:
        """Access the statecounter Manager for debugging counter iteration."""
        return self._counter_manager
    
    @property
    def codon_table(self) -> CodonTable:
        """Access the CodonTable for ORF operations."""
        return self._codon_table
    
    def set_genetic_code(self, genetic_code: Union[str, dict]) -> None:
        """Set or change the genetic code used for ORF operations."""
        self._codon_table = CodonTable(genetic_code)
    
    @property
    def alphabet(self) -> Alphabet:
        """Access the Alphabet for sequence operations."""
        return self._alphabet
    
    def set_alphabet(self, alphabet: Union[str, Alphabet]) -> None:
        """Set or change the alphabet used for sequence operations."""
        if isinstance(alphabet, str):
            self._alphabet = get_alphabet(alphabet)
        else:
            self._alphabet = alphabet
    
    def __enter__(self) -> "Party":
        """Enter the Party context."""
        global _active_party
        if _active_party is not None:
            raise RuntimeError("Nested Party contexts are not supported")
        _active_party = self
        self._is_active = True
        self._counter_manager.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the Party context."""
        global _active_party
        self._counter_manager.__exit__(exc_type, exc_val, exc_tb)
        _active_party = None
        self._is_active = False
    
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