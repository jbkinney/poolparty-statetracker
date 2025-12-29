import random
from typing import Union, List, Optional, Tuple, Dict, Any
from itertools import combinations
from math import comb
from .pool import Pool
from .utils import validate_alphabet


def _nth_combination(iterable: range, r: int, index: int) -> Tuple[int, ...]:
    """Compute the nth combination directly without generating predecessors.
    
    Uses the combinatorial number system to find the combination at position 'index'
    when choosing r elements from iterable (must be a range).
    
    Time complexity: O(r * log(n)) where n = len(iterable)
    
    Args:
        iterable: A range object (e.g., range(10) or range(5, 15))
        r: Number of elements to choose
        index: 0-based index of desired combination
    
    Returns:
        Tuple of r elements representing the index-th combination
    """
    pool = list(iterable)
    n = len(pool)
    
    if r > n or r < 0:
        raise ValueError(f"Invalid r={r} for n={n}")
    
    c = comb(n, r)
    if index < 0 or index >= c:
        raise IndexError(f"Index {index} out of range [0, {c})")
    
    result = []
    k = r
    
    for i in range(n):
        if k == 0:
            break
        # Number of combinations that start with pool[i]
        count = comb(n - i - 1, k - 1)
        if index < count:
            result.append(pool[i])
            k -= 1
        else:
            index -= count
    
    return tuple(result)


class KMutationPool(Pool):
    """A class for generating sequences with exactly k mutations.
    
    Supports both random and sequential iteration through all possible
    k-mutation combinations (finite states).
    
    In sequential mode (via iteration or set_state), systematically explores
    all possible k-position combinations and mutation choices.
    The rightmost mutation position and character vary most rapidly.
    
    Position selection modes (mutually exclusive):
    1. Default (positions=None, adjacent=False): All positions eligible
    2. Explicit positions: Only specified positions can be mutated
    3. Adjacent mode (adjacent=True): Only k adjacent positions mutated
    """
    def __init__(self, 
                 seq: Union[Pool, str], 
                 alphabet: Union[str, List[str]] = 'dna', 
                 k: int = 1,
                 positions: Optional[List[int]] = None,
                 adjacent: bool = False,
                 mark_changes: bool = False,
                 max_num_states: int = None,
                 mode: str = 'random',
                 iteration_order: int | None = None,
                 name: str | None = None,
                 metadata: str = 'features'):
        """Initialize a KMutationPool.
        
        Args:
            seq: Input sequence to mutate (Pool or string)
            alphabet: Either a string naming a predefined alphabet (e.g., 'dna', 'rna'),
                or a list of single-character strings to use as the alphabet.
                Default: 'dna'
            k: Number of positions to mutate (must be > 0 and <= sequence length).
                Default: 1
            positions: Optional list of position indices eligible for mutation.
                If provided, only these positions can be mutated.
                Cannot be used together with adjacent=True.
                Default: None (all positions eligible)
            adjacent: If True, only mutate k adjacent positions.
                In sequential mode: iterate starting position from 0 to len(seq)-k,
                with rightmost position varying most rapidly.
                Cannot be used together with positions.
            mark_changes: If True, apply swapcase() to mutated positions (default: False)
            max_num_states: Maximum number of states before treating as infinite
            mode: Either 'random' or 'sequential' (default: 'random')
            iteration_order: Order for sequential iteration (default: auto-assigned based on creation order)
                
        Raises:
            ValueError: If alphabet is empty, k is invalid, sequence is too short,
                positions is invalid, or both positions and adjacent are specified
        """
        # Validate and store alphabet
        self.alphabet = validate_alphabet(alphabet)
        
        # Store k
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self.k = k
        
        # Validate positions and adjacent are mutually exclusive
        if positions is not None and adjacent:
            raise ValueError(
                "Cannot use both 'positions' and 'adjacent=True'. "
                "These options are mutually exclusive."
            )
        
        self.adjacent = adjacent
        
        # Store mark_changes flag
        self.mark_changes = mark_changes
        
        # Get sequence length for validation
        base_seq = seq if isinstance(seq, str) else seq.seq
        seq_len = len(base_seq)
        
        # Validate and store positions
        if positions is not None:
            if not positions or len(positions) == 0:
                raise ValueError("positions must be a non-empty list")
            for pos in positions:
                if not isinstance(pos, int):
                    raise ValueError(f"All positions must be integers, got {type(pos)}")
                if pos < 0 or pos >= seq_len:
                    raise ValueError(
                        f"Position {pos} is out of bounds. "
                        f"Valid range: [0, {seq_len})"
                    )
            # Check for duplicates
            if len(positions) != len(set(positions)):
                raise ValueError("positions must not contain duplicates")
            self.positions = list(positions)
            self._use_positions = True
        else:
            self.positions = None
            self._use_positions = False
        
        # Store whether parent is a Pool (affects caching strategy)
        self._parent_is_pool = isinstance(seq, Pool)
        self._initial_seq_len = seq_len
        
        super().__init__(parents=(seq,), op='k_mutate', max_num_states=max_num_states, mode=mode, iteration_order=iteration_order, name=name, metadata=metadata)
        
        # Validate k against available positions
        if self._use_positions:
            if k > len(self.positions):
                raise ValueError(
                    f"k ({k}) must be <= number of available positions ({len(self.positions)})"
                )
        else:
            if k > seq_len:
                raise ValueError(
                    f"k ({k}) must be <= sequence length ({seq_len})"
                )
        
        # === CACHING: Pre-compute static values ===
        # These depend only on alphabet and k, so always static
        self._num_alternatives = len(self.alphabet) - 1
        self._num_mutation_patterns = self._num_alternatives ** self.k if self._num_alternatives > 0 else 0
        
        # Cache position combinations (expensive to regenerate)
        self._sequential_cache_ready = False
        self._position_combinations = None  # Will be populated by _cache_position_combinations
        self._num_position_choices = None
        self._cached_valid_states = None
        
        if self._use_positions:
            # For explicit positions, combinations are always static
            self._cache_position_combinations()
        elif not self._parent_is_pool:
            # For string parent, sequence length is fixed, so we can cache
            self._cache_position_combinations()
        
        # Design cards: cached mutation info from last _compute_seq call
        self._cached_mut_pos: List[int] | None = None
        self._cached_mut_from: List[str] | None = None
        self._cached_mut_to: List[str] | None = None
        self._cached_state: int | None = None
    
    # Maximum number of combinations to cache (avoid memory explosion)
    _MAX_CACHED_COMBINATIONS = 100_000
    
    def _cache_position_combinations(self):
        """Pre-compute position combinations for sequential mode (O(1) lookup vs O(n choose k)).
        
        Only caches if number of combinations is below threshold to avoid memory issues.
        For large state spaces, falls back to on-the-fly computation.
        """
        if self._use_positions:
            num_combinations = comb(len(self.positions), self.k)
            self._num_position_choices = num_combinations
            if num_combinations <= self._MAX_CACHED_COMBINATIONS:
                self._position_combinations = list(combinations(self.positions, self.k))
            else:
                self._position_combinations = None  # Too large to cache
        elif self.adjacent:
            # Adjacent mode: no need to cache combinations, just compute start positions
            self._num_position_choices = self._initial_seq_len - self.k + 1
            self._position_combinations = None  # Not needed for adjacent
        else:
            # Default mode: all positions
            num_combinations = comb(self._initial_seq_len, self.k)
            self._num_position_choices = num_combinations
            if num_combinations <= self._MAX_CACHED_COMBINATIONS:
                self._position_combinations = list(combinations(range(self._initial_seq_len), self.k))
            else:
                self._position_combinations = None  # Too large to cache
        
        self._cached_valid_states = self._num_position_choices * self._num_mutation_patterns
        self._sequential_cache_ready = True
    
    def _calculate_seq_length(self) -> int:
        """Mutations don't change length - inherit from parent."""
        parent = self.parents[0]
        if isinstance(parent, Pool):
            return parent.seq_length
        return len(parent)
    
    def _calculate_num_internal_states(self) -> int:
        """Calculate the finite number of states for KMutationPool.
        
        Number of states:
        - If positions specified: C(len(positions), k) * (alpha - 1)^k
        - If adjacent=True: (L - k + 1) * (alpha - 1)^k
        - Otherwise: C(L, k) * (alpha - 1)^k
          where L = len(seq), alpha = len(alphabet)
        
        Each position is mutated to one of (alpha - 1) alternatives
        (excluding the original character at that position).
        """
        # Use cached values if available
        if hasattr(self, '_sequential_cache_ready') and self._sequential_cache_ready:
            return self._cached_valid_states
        
        # Fallback: compute from scratch (used during __init__ before cache is ready)
        parent = self.parents[0]
        
        # Get sequence length
        if isinstance(parent, Pool):
            L = parent.seq_length
        else:
            L = len(parent)
        
        # Count alternatives for each position
        # Each position can mutate to (alphabet_size - 1) characters
        num_alternatives_per_position = len(self.alphabet) - 1
        
        if num_alternatives_per_position <= 0:
            # If alphabet is too small, no mutations possible
            return 0
        
        # Calculate number of mutation patterns
        num_mutation_patterns = num_alternatives_per_position ** self.k
        
        # Calculate number of position choices based on mode
        if self._use_positions:
            # Explicit positions: choose k from the provided positions
            num_position_choices = comb(len(self.positions), self.k)
        elif self.adjacent:
            # Number of starting positions for k adjacent mutations
            num_position_choices = L - self.k + 1
        else:
            # Number of ways to choose k positions from L
            num_position_choices = comb(L, self.k)
        
        return num_position_choices * num_mutation_patterns
    
    def _compute_seq(self) -> str:
        """Compute mutated sequence with exactly k mutations.
        
        In sequential mode, the state determines:
        1. Which positions to mutate (combination, starting position, or from positions list)
        2. Which specific mutation to use at each position
        
        The rightmost mutation position varies most rapidly.
        
        Uses cached values for ~10-100x speedup on repeated calls.
        
        Returns:
            Sequence with exactly k mutations
        """
        # Get base sequence
        if self._parent_is_pool:
            base_seq = self.parents[0].seq
        else:
            base_seq = self.parents[0]
        
        # Use cached num_alternatives
        if self._num_alternatives <= 0:
            return base_seq
        
        # Lazy caching for Pool parents (built on first access)
        if not self._sequential_cache_ready:
            self._cache_position_combinations()
        
        # Use cached values
        valid_states = self._cached_valid_states
        if valid_states <= 0:
            return base_seq
        
        # Wrap state within valid range
        wrapped_state = self.get_state() % valid_states
        
        # Decompose state into position choice and mutation pattern
        position_idx = wrapped_state // self._num_mutation_patterns
        mutation_pattern_idx = wrapped_state % self._num_mutation_patterns
        
        # Get positions to mutate
        if self.adjacent:
            # Adjacent: simple range computation
            positions = list(range(position_idx, position_idx + self.k))
        elif self._position_combinations is not None:
            # O(1) lookup from cache
            positions = list(self._position_combinations[position_idx])
        else:
            # Combinations not cached (too large) - compute nth directly
            # O(k * log(n)) using combinatorial number system
            if self._use_positions:
                positions = list(_nth_combination(range(len(self.positions)), self.k, position_idx))
                positions = [self.positions[i] for i in positions]
            else:
                positions = list(_nth_combination(range(self._initial_seq_len), self.k, position_idx))
        
        # Apply mutations
        result = list(base_seq)
        
        # Cache for design cards
        mut_pos_list = []
        mut_from_list = []
        mut_to_list = []
        
        # Decompose mutation_pattern_idx to get specific mutation for each position
        # Rightmost position varies most rapidly
        for i in range(self.k - 1, -1, -1):
            pos = positions[i]
            original_char = base_seq[pos]
            
            # Get available mutations (exclude original character)
            available_chars = [c for c in self.alphabet if c != original_char]
            
            if available_chars:
                # Select mutation based on pattern index
                mutation_choice = mutation_pattern_idx % self._num_alternatives
                mutation_pattern_idx //= self._num_alternatives
                new_char = available_chars[mutation_choice]
                result[pos] = new_char
                
                # Cache mutation info (prepend to maintain position order)
                mut_pos_list.insert(0, pos)
                mut_from_list.insert(0, original_char)
                mut_to_list.insert(0, new_char)
        
        # Store cached values
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
        """Return metadata for this KMutationPool at the current state.
        
        Extends base Pool metadata with mutation information.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end only
            - 'features': core + mut_pos, mut_pos_abs, mut_from, mut_to (default)
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
        
        # Add KMutationPool-specific fields for 'features' and 'complete' levels
        if self._metadata_level in ('features', 'complete'):
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
        extra_args = []
        if self._use_positions:
            extra_args.append(f"positions={self.positions}")
        if self.adjacent:
            extra_args.append("adjacent=True")
        extra_str = ", " + ", ".join(extra_args) if extra_args else ""
        return f"KMutationPool(seq={parent_seq}, alphabet={self.alphabet}, k={self.k}{extra_str})"

