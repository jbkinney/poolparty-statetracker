from .types import Pool_type, Optional, Sequence, ModeType, AlphabetType, beartype
from .alphabet import validate_alphabet
import numpy as np
import pandas as pd

@beartype
class Operation:
    id_counter: int = 0
    max_sequential_states: int = 1_000_000
    design_card_keys: Sequence[str] = []  # Subclasses override with their design keys
    num_outputs: int = 1  # Number of output sequences per state (>1 for multi-output ops)
    
    #########################################################################
    # Constructor
    #########################################################################
    
    def __init__(
        self,
        parent_pools: Sequence[Pool_type],
        num_states: int,
        mode: ModeType,
        seq_length: Optional[int] = None,
        name: Optional[str] = None,
        design_card_keys: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize Operation.
        
        Subclasses MUST call super().__init__() at the END of their __init__
        after setting any operation-specific attributes.
        
        Args:
            parent_pools: List of parent Pool objects for DAG traversal
            num_states: Number of internal states (-1 for unknown/infinite)
            mode: 'random', 'sequential', or 'fixed'
            seq_length: Sequence length specification:
                - None: variable/unknown length
                - int: fixed length
            name: Optional name for this operation
            design_card_keys: Which design card keys to include (default: all possible)
        """
        # Set parent_pools (late import to avoid circular dependency)
        from .pool import Pool
        self.parent_pools = parent_pools
        
        # Set num_states
        if not ((num_states > 0 or num_states == -1)):
            raise ValueError("num_states must be positive or -1; got {num_states}")
        self.num_states = num_states
        
        # Set seq_length 
        self.seq_length = seq_length
        
        # Set name
        if name is None:
            name = self.__class__.__name__
        self.name = name
        
        # Validate and set design_card_keys
        if design_card_keys is None:
            design_card_keys = self.__class__.design_card_keys
        self.active_design_card_keys = self._validate_card_keys(keys=design_card_keys)
        
        # Auto-set id and _current_state
        self.id = Operation.id_counter
        Operation.id_counter += 1
        self.state = 0
        self.states = []
        
        # Set mode
        self.set_mode(mode)
        
        # Create initial RNG (skip for fixed mode - fixed ops have no RNG)
        if mode != 'fixed':
            self.initialize_rng(0)
        
        # Initialize results DataFrame (stores seq + design card data)
        self._results_df: Optional[pd.DataFrame] = None
    
    #########################################################################
    # Sequence computation
    #########################################################################
    
    def compute_results_row(
        self, 
        input_strings: Sequence[str], 
        sequential_state: Optional[int],
    ) -> dict:
        raise NotImplementedError("Subclasses must implement compute_results_row")  
    
    def compute_results(
        self, 
        input_strings_lists: Sequence[Sequence[str]], 
        sequential_states: Sequence[int],
    ) -> None:
        """Compute sequences and store in self._results_df.
        
        Subclasses must create a DataFrame with 'seq' column first,
        followed by columns for each key in design_card_keys.
        
        Default implementation calls compute_results_row() for each and builds DataFrame.
        Derived classes can override for batch-optimized implementations.
        """
        num_seqs = len(sequential_states)
        if len(input_strings_lists) == 0:
            per_seq_input_strings = [[]] * num_seqs
        else:
            per_seq_input_strings = list(zip(*input_strings_lists))
        
        rows = []
        for state, input_strings in zip(sequential_states, per_seq_input_strings):
            row = self.compute_results_row(input_strings, int(state))
            rows.append(row)
        self._results_df = pd.DataFrame(rows)
    
    def run(self, num_seqs) -> list[str]:
        """Recursively compute num_seqs sequences in one DAG traversal"""
        input_seq_lists = [parent.operation.run(num_seqs) for parent in self.parent_pools]
        states = [int(s) for s in self.states] if self.mode == 'sequential' else [0] * num_seqs
        self.compute_results(input_seq_lists, states)
        return self._results_df['seq'].tolist()
    
    #########################################################################
    # Public methods
    #########################################################################
    
    def is_sequential_compatible(self) -> bool:
        return (
            self.num_states is not None 
            and 0 < self.num_states <= self.max_sequential_states
        )
        
    def initialize_rng(self, seed: int) -> None:
        if self.mode == 'fixed':
            raise RuntimeError(
                f"Cannot set RNG seed on fixed-mode operation '{self.name}' (id={self.id})"
            )
        self.rng = np.random.default_rng(seed)
    
    def set_state(self, state: int) -> None:
        """Set the state for this operation. Raises error for fixed mode."""
        if self.mode == 'fixed':
            raise RuntimeError(
                f"Cannot set state on fixed-mode operation '{self.name}' (id={self.id})"
            )
        self.state = state
    
    def set_mode(self, mode: ModeType) -> None:
        if mode == 'sequential' and not self.is_sequential_compatible():
            if self.num_states == -1:
                reason = "has unknown number of states"
            elif self.num_states is None or self.num_states <= 0:
                reason = f"has {self.num_states} states (need > 0)"
            else:
                reason = f"has {self.num_states:,} states (max: {self.max_sequential_states:,})"
            raise ValueError(
                f"Operation '{self.name}' cannot be sequential ({reason})"
            )
        self.mode = mode
    
    #########################################################################
    # Results methods
    #########################################################################
    
    def get_results(self) -> pd.DataFrame:
        """Return results DataFrame filtered by active_design_card_keys."""
        if self._results_df is None:
            return pd.DataFrame()
        # Always include 'seq' column, plus any active design card keys
        cols = ['seq'] + [k for k in self.active_design_card_keys if k != 'seq']
        available_cols = [c for c in cols if c in self._results_df.columns]
        return self._results_df[available_cols].copy()
    
    def clear_results(self) -> None:
        """Clear the results DataFrame."""
        self._results_df = None
        
    #########################################################################
    # Helper methods
    #########################################################################
        
    def _validate_card_keys(self, keys: Sequence[str]) -> list[str]:
        keys = set(keys)
        expected_keys = set(self.design_card_keys)
        if not keys.issubset(expected_keys):
            invalid = keys - expected_keys
            raise ValueError(
                f"Invalid design_card_keys {invalid}. Valid: {expected_keys}"
            )
        return list(keys)
        
    def _set_alphabet(self, alphabet: AlphabetType) -> None:
        self.alphabet = validate_alphabet(alphabet)
        self.alpha = len(self.alphabet)
        if self.alpha <= 1:
            raise ValueError(f"Alphabet must have at least 2 characters, got {self.alpha}")
        
    #########################################################################
    # Dunder methods
    #########################################################################
       
    def __repr__(self) -> str:
        parts = [f"id={self.id}", f"mode='{self.mode}'"]
        if self.mode == 'sequential':
            parts.append(f"num_states={self.num_states}")
        if self.name is not None:
            parts.append(f"name='{self.name}'")
        return f"{self.__class__.__name__}({', '.join(parts)})"
    
    def __len__(self) -> int:
        return self.seq_length
        