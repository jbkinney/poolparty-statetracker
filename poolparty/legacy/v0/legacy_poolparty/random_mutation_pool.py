import random
from typing import Union, List, Dict, Any
from .pool import Pool
from .utils import validate_alphabet


class RandomMutationPool(Pool):
    """A class for generating randomly mutated versions of an input sequence.
    
    Each position in the sequence independently has a probability of being mutated
    to a different character from the alphabet. Has infinite states.
    """
    def __init__(self, 
                 seq: Union[Pool, str], 
                 alphabet: Union[str, List[str]] = 'dna', 
                 mutation_rate: Union[float, List[float]] = 0.1,
                 mark_changes: bool = False,
                 mode: str = 'random',
                 iteration_order: int | None = None,
                 name: str | None = None,
                 metadata: str = 'features'):
        """Initialize a RandomMutationPool.
        
        Args:
            seq: Input sequence to mutate (Pool or string)
            alphabet: Either a string naming a predefined alphabet (e.g., 'dna', 'rna'),
                or a list of single-character strings to use as the alphabet.
                Default: 'dna'
            mutation_rate: Probability of mutation at each position (0-1).
                Can be a single float for uniform rate, or a list of floats
                for position-specific rates. Default: 0.1
            mark_changes: If True, apply swapcase() to mutated positions (default: False)
            mode: Either 'random' or 'sequential' (default: 'random')
            iteration_order: Order for sequential iteration (default: auto-assigned based on creation order)
                
        Raises:
            ValueError: If alphabet is empty, mutation_rate is out of range,
                or mutation_rate array length doesn't match sequence length
        """
        # RandomMutationPool only supports random mode (infinite states)
        if mode == 'sequential':
            raise ValueError("RandomMutationPool only supports mode='random'")
        
        # Validate and store alphabet
        self.alphabet = validate_alphabet(alphabet)
        
        # Validate and store mutation_rate
        if isinstance(mutation_rate, (list, tuple)):
            # Position-specific rates
            for rate in mutation_rate:
                if not 0 <= rate <= 1:
                    raise ValueError(f"mutation_rate values must be between 0 and 1, got {rate}")
            self.mutation_rate = list(mutation_rate)
            self._is_uniform_rate = False
        else:
            # Uniform rate
            if not 0 <= mutation_rate <= 1:
                raise ValueError(f"mutation_rate must be between 0 and 1, got {mutation_rate}")
            self.mutation_rate = mutation_rate
            self._is_uniform_rate = True
        
        # Store mark_changes flag
        self.mark_changes = mark_changes
        
        super().__init__(parents=(seq,), op='mutate', mode=mode, iteration_order=iteration_order, name=name, metadata=metadata)
        
        # Design cards: cached mutation info from last _compute_seq call
        self._cached_mut_count: int | None = None
        self._cached_mut_pos: List[int] | None = None
        self._cached_mut_from: List[str] | None = None
        self._cached_mut_to: List[str] | None = None
        self._cached_state: int | None = None
    
    def _calculate_num_internal_states(self) -> float:
        """RandomMutationPool has infinite internal states (different random mutations)."""
        return float('inf')
    
    def _calculate_seq_length(self) -> int:
        """Mutations don't change length - inherit from parent."""
        parent = self.parents[0]
        if isinstance(parent, Pool):
            return parent.seq_length
        return len(parent)
    
    def _compute_seq(self) -> str:
        """Compute mutated sequence based on current state.
        
        Returns:
            Mutated version of the input sequence
        """
        # Get base sequence
        if isinstance(self.parents[0], str):
            base_seq = self.parents[0]
        else:
            base_seq = self.parents[0].seq
        
        # Validate mutation_rate length if array
        if not self._is_uniform_rate and len(self.mutation_rate) != len(base_seq):
            raise ValueError(
                f"mutation_rate array length ({len(self.mutation_rate)}) "
                f"must match sequence length ({len(base_seq)})"
            )
        
        # Create local Random instance to avoid polluting global random state
        local_rng = random.Random(self.get_state())
        
        # Cache for design cards
        mut_pos_list = []
        mut_from_list = []
        mut_to_list = []
        
        # Generate mutations
        result = []
        for i, char in enumerate(base_seq):
            # Get mutation rate for this position
            rate = self.mutation_rate if self._is_uniform_rate else self.mutation_rate[i]
            
            # Check if position should mutate
            if local_rng.random() < rate:
                # Mutate to a different character
                available_chars = [c for c in self.alphabet if c != char]
                if available_chars:
                    mutated_char = local_rng.choice(available_chars)
                    result.append(mutated_char)
                    # Cache mutation info
                    mut_pos_list.append(i)
                    mut_from_list.append(char)
                    mut_to_list.append(mutated_char)
                else:
                    # If no other characters available, keep original
                    result.append(char)
            else:
                # No mutation
                result.append(char)
        
        # Store cached values
        self._cached_mut_count = len(mut_pos_list)
        self._cached_mut_pos = mut_pos_list
        self._cached_mut_from = mut_from_list
        self._cached_mut_to = mut_to_list
        self._cached_state = self.get_state()
        
        # Apply case change to mutated positions if requested
        if self.mark_changes:
            result = [
                result[i].swapcase() if result[i] != base_seq[i] else result[i]
                for i in range(len(result))
            ]
        
        return ''.join(result)
    
    # =========================================================================
    # Design Cards Methods
    # =========================================================================
    
    def get_metadata(self, abs_start: int, abs_end: int) -> Dict[str, Any]:
        """Return metadata for this RandomMutationPool at the current state.
        
        Extends base Pool metadata with mutation information.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end only
            - 'features': core + mut_count, mut_pos, mut_pos_abs, mut_from, mut_to (default)
            - 'complete': features + value
        
        Args:
            abs_start: Absolute start position in the final sequence
            abs_end: Absolute end position in the final sequence
            
        Returns:
            Dictionary with metadata fields based on metadata level.
        """
        # Ensure cached values are current (recompute if state changed)
        if self._cached_state != self.get_state():
            _ = self.seq  # Trigger computation to populate cache
        
        # Get base metadata (handles core fields and 'complete' level value)
        metadata = super().get_metadata(abs_start, abs_end)
        
        # Add RandomMutationPool-specific fields for 'features' and 'complete' levels
        if self._metadata_level in ('features', 'complete'):
            metadata['mut_count'] = self._cached_mut_count
            metadata['mut_pos'] = self._cached_mut_pos.copy() if self._cached_mut_pos else []
            metadata['mut_pos_abs'] = (
                [abs_start + p for p in self._cached_mut_pos] 
                if abs_start is not None and self._cached_mut_pos else None
            )
            metadata['mut_from'] = self._cached_mut_from.copy() if self._cached_mut_from else []
            metadata['mut_to'] = self._cached_mut_to.copy() if self._cached_mut_to else []
        
        return metadata
    
    def __repr__(self) -> str:
        parent_seq = self.parents[0].seq if isinstance(self.parents[0], Pool) else self.parents[0]
        if self._is_uniform_rate:
            rate_str = f"{self.mutation_rate}"
        else:
            rate_str = f"[{len(self.mutation_rate)} rates]"
        return f"RandomMutationPool(seq={parent_seq}, alphabet={self.alphabet}, rate={rate_str})"

