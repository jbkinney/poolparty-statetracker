"""ORFMutationScan operation - apply k codon-level mutations to an ORF.

Supported mutation types:
1. any_codon: Mutate to any different codon (uniform, 63 alternatives)
2. nonsynonymous_first: Different AA/stop, first codon (uniform, 20 alternatives)
3. nonsynonymous_random: Different AA/stop, random codon (non-uniform)
4. missense_only_first: Different AA only, first codon (uniform, 19 alternatives, NO stop)
5. missense_only_random: Different AA only, random codon (non-uniform, NO stop)
6. synonymous: Synonymous mutations only (non-uniform)
7. nonsense: Introduce stop codons (uniform, 3 alternatives)
"""

from __future__ import annotations
import random
from typing import TYPE_CHECKING, Union, List, Tuple, Optional
from itertools import combinations
from math import comb

from ..orf_operation import ORFOp

if TYPE_CHECKING:
    from ..pool import Pool


def _nth_combination(iterable: range, r: int, index: int) -> tuple[int, ...]:
    """Compute the nth combination directly without generating predecessors.
    
    Uses the combinatorial number system to find the combination at position 'index'
    when choosing r elements from iterable (must be a range).
    
    Args:
        iterable: A range object (e.g., range(10))
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
        count = comb(n - i - 1, k - 1)
        if index < count:
            result.append(pool[i])
            k -= 1
        else:
            index -= count
    
    return tuple(result)


class ORFMutationScanOp(ORFOp):
    """Apply k codon-level mutations to an ORF sequence.
    
    This is a transformer operation - it has one parent pool.
    Generates k-mutation combinations at the codon level while preserving
    the reading frame.
    
    Supports both uniform mutation types (compatible with sequential mode)
    and non-uniform types (random mode only).
    
    Uniform mutation types (sequential-compatible):
    - any_codon: 63 alternatives per codon
    - nonsynonymous_first: 20 alternatives per codon
    - missense_only_first: 19 alternatives per codon
    - nonsense: 3 alternatives per non-stop codon
    
    Non-uniform mutation types (random mode only):
    - nonsynonymous_random: variable by codon
    - missense_only_random: variable by codon
    - synonymous: variable by codon (0-5 alternatives)
    """
    
    op_name = 'orf_mutation_scan'
    
    # Maximum combinations to cache
    _MAX_CACHED_COMBINATIONS = 100_000
    
    # Mutation types that have uniform number of alternatives per codon
    UNIFORM_MUTATION_TYPES = {'any_codon', 'nonsynonymous_first', 'missense_only_first', 'nonsense'}
    
    def __init__(
        self,
        parent: Union['Pool', str],
        mutation_type: str,
        k: int = 1,
        positions: Optional[list[int]] = None,
        orf_start: int = 0,
        orf_end: Optional[int] = None,
        codon_table: Union[dict, str, None] = 'standard',
        mark_changes: bool = False,
        mode: str = 'random',
        name: Optional[str] = None,
    ):
        """Initialize ORFMutationScanOp.
        
        Args:
            parent: Input sequence to mutate (Pool or string). Can include flanking regions.
            mutation_type: Type of mutation (see class docstring for options)
            k: Number of codons to mutate (must be > 0). Default: 1
            positions: Optional list of codon indices eligible for mutation.
                Default: all codons are eligible.
            orf_start: Start index of ORF within seq (0-based, inclusive). Default: 0
            orf_end: End index of ORF within seq (0-based, exclusive). Default: len(seq)
            codon_table: Genetic code specification:
                - 'standard': Use standard genetic code (default, cached for speed)
                - dict: Custom genetic code as {AA: [codon, ...]}
                - None: Use standard genetic code
            mark_changes: If True, apply swapcase() to mutated codons
            mode: Either 'random' or 'sequential' (default: 'random')
            name: Optional name for this operation
        
        Raises:
            ValueError: If k is invalid, mutation_type is invalid, or ORF validation fails
        """
        # Treat None exactly like 'standard' since mutations always need codon lookups
        if codon_table is None:
            codon_table = 'standard'
        
        # Handle Pool vs string input to create parent_pool
        if isinstance(parent, str):
            from ..operations.from_seqs_op import from_seqs_op
            parent_pool = from_seqs_op([parent])
            seq = parent
        else:
            parent_pool = parent
            seq = parent.seq
        
        # We need to do minimal ORF validation before calling super()
        # to compute num_states and other required parameters
        
        # Validate DNA
        from ..orf_operation import ORFOp
        ORFOp._validate_dna_sequence(seq)
        
        # Handle ORF boundaries
        actual_orf_end = orf_end if orf_end is not None else len(seq)
        ORFOp._validate_orf_boundaries(seq, orf_start, actual_orf_end)
        
        # Extract and validate ORF
        orf_seq = seq[orf_start:actual_orf_end]
        if len(orf_seq) % 3 != 0:
            raise ValueError(
                f"ORF region length must be divisible by 3, got {len(orf_seq)} "
                f"(orf_start={orf_start}, orf_end={actual_orf_end})"
            )
        
        num_codons = len(orf_seq) // 3
        
        # Validate and store eligible codon positions
        if positions is None:
            # Default: all codon positions are eligible for mutation
            positions_list = list(range(num_codons))
        else:
            # Validate provided positions
            if not positions or len(positions) == 0:
                raise ValueError("positions must be a non-empty list")
            for pos in positions:
                if not isinstance(pos, int):
                    raise ValueError(f"All positions must be integers, got {type(pos)}")
                if pos < 0 or pos >= num_codons:
                    raise ValueError(
                        f"Position {pos} is out of bounds. "
                        f"Valid range: [0, {num_codons})"
                    )
            # Check for duplicates
            if len(positions) != len(set(positions)):
                raise ValueError("positions must not contain duplicates")
            positions_list = list(positions)
        
        # Validate k
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if k > len(positions_list):
            raise ValueError(
                f"k ({k}) must be <= number of available positions ({len(positions_list)})"
            )
        
        # Initialize base class (handles validation, codon tables, parent pool setup)
        # This will set up codon tables and mutation_lookup that we need
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,  # Temporary, will be recomputed below
            mode=mode,
            seq_length=len(seq),
            name=name,
            parent=parent,
            orf_start=orf_start,
            orf_end=orf_end,
            codon_table=codon_table,
            mark_changes=mark_changes,
        )
        
        # Store operation-specific attributes
        self.k = k
        self.positions = positions_list
        
        # Validate mutation_type
        valid_types = list(self.mutation_lookup.keys())
        if mutation_type not in valid_types:
            raise ValueError(f"mutation_type must be one of {valid_types}, got '{mutation_type}'")
        
        self.mutation_type = mutation_type
        
        # For nonsense, validate no stop codons in input ORF
        if mutation_type == 'nonsense':
            for i, codon in enumerate(self.codons):
                if codon.upper() in self.stop_codons:
                    raise ValueError(
                        f"ORF contains stop codon '{codon}' at position {i} "
                        f"(nonsense type requires no stop codons in input)"
                    )
        
        # Pre-compute mutation counts per codon
        self.num_possible_mutations = [
            self._get_num_mutations(c, mutation_type) for c in self.codons
        ]
        
        # Check uniformity (all eligible positions have same number of mutations)
        mutation_counts = [self.num_possible_mutations[p] for p in self.positions]
        self.is_uniform = len(set(mutation_counts)) == 1
        self.uniform_num_possible_mutations = mutation_counts[0] if self.is_uniform else None
        
        # Compute final num_states and update it
        if not self.is_uniform:
            self.num_states = -1
        else:
            self.num_states = comb(len(self.positions), self.k) * (self.uniform_num_possible_mutations ** self.k)
        
        # Sequential-mode caches are built lazily
        self._sequential_cache_ready = False
        self._position_combinations: Optional[list[tuple[int, ...]]] = None
        self._num_mutation_patterns: Optional[int] = None
    
    def _get_num_mutations(self, codon: str, mutation_type: str) -> int:
        """Get number of possible mutations for a codon (O(1))."""
        return len(self.mutation_lookup[mutation_type][codon.upper()])
    
    def _mutate_codon(self, codon: str, mutation_type: str, state: int) -> Optional[str]:
        """Mutate a codon (internal method, assumes valid inputs)."""
        available_mutations = self.mutation_lookup[mutation_type][codon.upper()]
        if len(available_mutations) == 0:
            return None
        return available_mutations[state % len(available_mutations)]
    
    def _cache_sequential_components(self):
        """Pre-compute position combinations for sequential mode (~100x speedup)."""
        if self.uniform_num_possible_mutations is None:
            return
        self._num_mutation_patterns = self.uniform_num_possible_mutations ** self.k
        
        # Only cache if small enough
        num_combinations = comb(len(self.positions), self.k)
        if num_combinations <= self._MAX_CACHED_COMBINATIONS:
            self._position_combinations = list(combinations(self.positions, self.k))
        
        self._sequential_cache_ready = True
    
    def compute_seq(
        self, 
        input_strings: list[str], 
        state: int
    ) -> str:
        """Compute mutated sequence with exactly k codon mutations.
        
        Args:
            input_strings: List containing the parent sequence
            state: Internal state number
        
        Returns:
            Mutated sequence
        
        The state determines:
        1. Which k codon positions to mutate (combination)
        2. Which specific mutation to use at each position
        """
        base_seq = input_strings[0]
        
        # Extract codons from current sequence
        base_codons = self._get_codons_from_seq(base_seq)
        
        # Handle edge case: no possible mutations
        if self.num_states == 0:
            mutated_orf = ''.join(base_codons)
            return self._reassemble_with_flanks(mutated_orf, base_seq)
        
        mutated_codons = base_codons.copy()
        codon_positions = []
        
        if self.is_uniform and self.num_states != -1:
            # Sequential/deterministic mode
            if not self._sequential_cache_ready:
                self._cache_sequential_components()
            
            # Wrap state within valid range
            wrapped_state = state % self.num_states if self.num_states > 0 else 0
            
            # Mixed-radix decomposition
            num_mutation_patterns = self._num_mutation_patterns
            position_index = wrapped_state // num_mutation_patterns
            mutation_pattern_index = wrapped_state % num_mutation_patterns
            
            # Get positions from cache or compute directly
            if self._position_combinations is not None:
                positions = self._position_combinations[position_index]
            else:
                positions = _nth_combination(range(len(self.positions)), self.k, position_index)
                positions = tuple(self.positions[p] for p in positions)
            
            # Apply mutations (iterate in reverse for mixed-radix decomposition)
            for i in range(self.k - 1, -1, -1):
                pos = positions[i]
                original_codon = base_codons[pos].upper()
                mutation_index = mutation_pattern_index % self.uniform_num_possible_mutations
                mutation_pattern_index //= self.uniform_num_possible_mutations
                
                mutated_codon = self._mutate_codon(original_codon, self.mutation_type, mutation_index)
                if mutated_codon is not None:
                    mutated_codons[pos] = mutated_codon
                    codon_positions.insert(0, pos)
        else:
            # Random mode
            rng = random.Random(state)
            positions_to_mutate = rng.sample(self.positions, self.k)
            
            for pos in sorted(positions_to_mutate):
                original_codon = base_codons[pos].upper()
                num_alternatives = self.num_possible_mutations[pos]
                
                if num_alternatives > 0:
                    mutation_index = rng.randint(0, num_alternatives - 1)
                    mutated_codon = self._mutate_codon(
                        original_codon, self.mutation_type, mutation_index
                    )
                    if mutated_codon is not None:
                        mutated_codons[pos] = mutated_codon
                        codon_positions.append(pos)
        
        # Apply case change if requested
        if self.mark_changes:
            for pos in codon_positions:
                mutated_codons[pos] = mutated_codons[pos].swapcase()
        
        # Reassemble sequence
        mutated_orf = ''.join(mutated_codons)
        return self._reassemble_with_flanks(mutated_orf, base_seq)


def orf_mutation_scan_op(
    seq: Union['Pool', str],
    mutation_type: str = 'missense_only_first',
    k: int = 1,
    positions: Optional[list[int]] = None,
    orf_start: int = 0,
    orf_end: Optional[int] = None,
    codon_table: Union[dict, str, None] = 'standard',
    mark_changes: bool = False,
    mode: str = 'random',
    name: Optional[str] = None,
) -> 'Pool':
    """Apply k codon-level mutations to a sequence or pool.
    
    Args:
        seq: Input sequence to mutate (Pool or string). Can include flanking regions.
        mutation_type: Type of mutation. Options:
            - 'any_codon': Any different codon (63 alternatives)
            - 'nonsynonymous_first': Different AA/stop, first codon (20 alternatives)
            - 'nonsynonymous_random': Different AA/stop, random codon (non-uniform)
            - 'missense_only_first': Different AA only, first codon (19 alternatives, no stop)
            - 'missense_only_random': Different AA only, random codon (non-uniform, no stop)
            - 'synonymous': Same amino acid, different codon (non-uniform)
            - 'nonsense': Stop codons only (3 alternatives)
            Default: 'missense_only_first'
        k: Number of codons to mutate (must be > 0). Default: 1
        positions: Optional list of codon indices eligible for mutation.
            Default: all codons are eligible.
        orf_start: Start index of ORF within seq (0-based, inclusive). Default: 0
        orf_end: End index of ORF within seq (0-based, exclusive). Default: len(seq)
        codon_table: Genetic code specification:
            - 'standard': Use standard genetic code (default, cached for speed)
            - dict: Custom genetic code as {AA: [codon, ...]}
            - None: Use standard genetic code
        mark_changes: If True, apply swapcase() to mutated codons. Default: False
        mode: Either 'random' or 'sequential'. Default: 'random'
            Note: 'sequential' requires a uniform mutation type (any_codon,
            nonsynonymous_first, missense_only_first, or nonsense).
        name: Optional name for this pool
    
    Returns:
        A Pool that generates k-codon-mutation variants.
    
    Example:
        >>> pool = orf_mutation_scan('ATGAAATTT', k=1, mutation_type='missense_only_first')
        >>> pool.operation.num_states
        57  # 3 codons × 19 alternatives each
        >>> pool.seq  # Returns a 1-codon-mutant variant
        'ATGAAGTTT'
    
    Raises:
        ValueError: If mutation_type is non-uniform and mode='sequential'
    """
    # Import here to avoid circular imports
    from ..pool import Pool
    
    # Validate mode compatibility before creating operation
    # (since is_uniform depends on the specific sequence and mutation type)
    if mode == 'sequential' and mutation_type not in ORFMutationScanOp.UNIFORM_MUTATION_TYPES:
        raise ValueError(
            f"mode='sequential' requires a uniform mutation type "
            f"({ORFMutationScanOp.UNIFORM_MUTATION_TYPES}); "
            f"got '{mutation_type}'"
        )
    
    # Create the operation (mode/name passed to operation, not pool)
    operation = ORFMutationScanOp(
        parent=seq,
        mutation_type=mutation_type,
        k=k,
        positions=positions,
        orf_start=orf_start,
        orf_end=orf_end,
        codon_table=codon_table,
        mark_changes=mark_changes,
        mode=mode,
        name=name,
    )
    
    return Pool(operation=operation)
