"""Party class - the context manager for building and executing sequence libraries.

Usage:
    with Party() as party:
        seq = from_seqs(['ACGT'])
        mutants = mutation_scan(seq, k=1)
        party.output(mutants, name='mutants')
    
    df = party.generate(num_seqs=100, seed=42)
"""

import numpy as np
import pandas as pd

from .types import beartype
from .pool import Pool
from .operation import Operation

# Global context for tracking the active party
_active_party: "Party | None" = None


@beartype
def get_active_party() -> "Party | None":
    """Get the currently active Party context, or None if not in a context."""
    return _active_party


class Party:
    """Context manager for building and executing sequence libraries.
    
    Pools created inside a Party context are automatically registered.
    After defining the DAG, call generate() to produce sequences.
    
    The Party maintains state between generate() calls, allowing for
    continuation of sequential iteration.
    """
    
    @beartype
    def __init__(self) -> None:
        """Initialize a new Party."""
        self._operations: list[Operation] = []
        self._outputs: dict[str, Pool] = {}
        self._current_state: int = 0
        self._master_seed: int | None = None
        self._is_active: bool = False
    
    @beartype
    def __enter__(self) -> "Party":
        """Enter the Party context."""
        global _active_party
        if _active_party is not None:
            raise RuntimeError("Nested Party contexts are not supported")
        _active_party = self
        self._is_active = True
        return self
    
    @beartype
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the Party context."""
        global _active_party
        _active_party = None
        self._is_active = False
    
    @beartype
    def _register_operation(self, operation: Operation) -> None:
        """Register an operation with this party.
        
        Called automatically when operations are created inside the context.
        """
        if operation not in self._operations:
            self._operations.append(operation)
    
    @beartype
    def output(self, pool: Pool, name: str | None = None) -> None:
        """Mark a pool as an output of this library.
        
        Args:
            pool: The pool to output
            name: Name for this output column in the result DataFrame
        """
        if name is None:
            name = pool.name or f"output_{len(self._outputs)}"
        self._outputs[name] = pool
    
    @beartype
    def reset(self) -> None:
        """Reset the state counter to 0."""
        self._current_state = 0
    
    @property
    def total_sequential_states(self) -> int:
        """Total number of sequential states across all sequential operations."""
        ops = self._topo_sort_operations()
        total = 1
        for op in ops:
            if op.mode == 'sequential':
                total *= op.num_states
        return total
    
    @beartype
    def generate(
        self,
        num_seqs: int | None = None,
        num_complete_iterations: int | None = None,
        seed: int | None = None,
        init_state: int | None = None,
    ) -> pd.DataFrame:
        """Generate sequences from the computation graph.
        
        Args:
            num_seqs: Number of sequences to generate
            num_complete_iterations: Generate this many complete iterations
                through all sequential states
            seed: Random seed (only used on first call or when explicitly set)
            init_state: Initial state for sequential operations (default: continue from last)
        
        Returns:
            DataFrame with output sequences and design card columns
        """
        if not self._outputs:
            raise ValueError("No outputs defined. Use party.output(pool) to mark outputs.")
        
        # Determine num_seqs
        if num_seqs is not None and num_complete_iterations is not None:
            raise ValueError("Specify num_seqs OR num_complete_iterations, not both")
        if num_seqs is None and num_complete_iterations is None:
            raise ValueError("Must specify num_seqs or num_complete_iterations")
        if num_complete_iterations is not None:
            num_seqs = num_complete_iterations * self.total_sequential_states
        
        # Set initial state
        if init_state is not None:
            self._current_state = init_state
        
        # Set master seed if provided
        if seed is not None:
            self._master_seed = seed
        if self._master_seed is None:
            self._master_seed = 0  # Default seed
        
        # Get topologically sorted operations
        sorted_ops = self._topo_sort_operations()
        
        # Seed RNGs for random operations
        self._seed_random_operations(sorted_ops)
        
        # Generate sequences
        rows = []
        for i in range(num_seqs):
            global_state = self._current_state + i
            row = self._compute_one(sorted_ops, global_state)
            rows.append(row)
        
        # Advance state
        self._current_state += num_seqs
        
        # Build DataFrame with outputs first
        df = pd.DataFrame(rows)
        output_cols = list(self._outputs.keys())
        other_cols = [c for c in df.columns if c not in output_cols]
        return df[output_cols + other_cols]
    
    @beartype
    def _topo_sort_operations(self) -> list[Operation]:
        """Topologically sort operations reachable from outputs."""
        visited: set[int] = set()
        result: list[Operation] = []
        
        def visit(pool: Pool) -> None:
            op = pool.operation
            if op.id in visited:
                return
            # Visit parents first
            for parent in op.parent_pools:
                visit(parent)
            visited.add(op.id)
            result.append(op)
        
        for pool in self._outputs.values():
            visit(pool)
        
        return result
    
    @beartype
    def _seed_random_operations(self, sorted_ops: list[Operation]) -> None:
        """Seed RNGs for random operations based on master seed."""
        for op in sorted_ops:
            if op.mode == 'random':
                # Derive per-op seed from master seed and op ID
                op_seed = self._master_seed + op.id * 1000003  # Large prime offset
                op.rng = np.random.default_rng(op_seed)
    
    @beartype
    def _compute_one(self, sorted_ops: list[Operation], global_state: int) -> dict:
        """Compute one row of output for the given global state."""
        # Cache for computed results: op.id -> result dict
        cache: dict[int, dict] = {}
        row: dict = {}
        
        # Decompose global state into per-op states
        remaining_state = global_state
        op_states: dict[int, int] = {}
        for op in sorted_ops:
            if op.mode == 'sequential':
                op_states[op.id] = remaining_state % op.num_states
                remaining_state //= op.num_states
            else:
                op_states[op.id] = 0
        
        # Execute operations in topological order
        for op in sorted_ops:
            # Gather parent sequences
            parent_seqs = []
            for parent in op.parent_pools:
                parent_result = cache[parent.operation.id]
                # Get the appropriate output from parent
                seq_key = f"seq_{parent.output_index}"
                parent_seqs.append(parent_result[seq_key])
            
            # Get state for this operation
            state = op_states[op.id]
            
            # Compute
            result = op.compute(parent_seqs, state, op.rng)
            cache[op.id] = result
            
            # Add design card columns (prefixed with op name)
            # Skip seq_0, seq_1, etc. (output sequence keys) but include other design cards
            for key in op.design_card_keys:
                if key in result:
                    # Check if this is an output sequence key (seq_N pattern)
                    is_output_key = (key.startswith('seq_') and 
                                     len(key) > 4 and 
                                     key[4:].isdigit())
                    if not is_output_key:
                        row[f"{op.name}.{key}"] = result[key]
        
        # Add output sequences
        for output_name, pool in self._outputs.items():
            result = cache[pool.operation.id]
            seq_key = f"seq_{pool.output_index}"
            row[output_name] = result[seq_key]
        
        return row
    
    def __repr__(self) -> str:
        return f"Party(outputs={list(self._outputs.keys())}, state={self._current_state})"
