import random
from typing import Union, List
from itertools import combinations
from math import comb
from .pool import Pool
# from .utils import mutate_codon, codon_to_aa_dict, stop_codons
from .orf_pool import ORFPool


class KMutationORFPool(ORFPool):
    """ORF pool with exactly k codon-level mutations (inherits from ORFPool).
        
    KEY FEATURES:
    - Inherits codon table management from ORFPool (hybrid caching)
    - Inherits ORF boundary handling from ORFPool (UTR support)
    - Inherits DNA validation from ORFPool
    
    Supported mutation types:
    1. any_codon: Mutate to any different codon (uniform)
    2. nonsynonymous_first: Different AA/stop, first codon (uniform)
    3. nonsynonymous_random: Different AA/stop, random codon (non-uniform)
    4. missense_only_first: Different AA only, first codon (uniform, NO stop)
    5. missense_only_random: Different AA only, random codon (non-uniform, NO stop)
    6. synonymous: Synonymous mutations only (non-uniform)
    7. nonsense: Introduce stop codons (uniform for standard code)
    """
    
    def __init__(self,
                 seq: Union[str, Pool],
                 mutation_type: str,
                 k: int = 1,
                 positions: List[int] = None,
                 orf_start: int = 0,
                 orf_end: int = None,
                 codon_table: Union[dict, str, None] = 'standard',
                 mark_changes: bool = False,
                 max_num_states: int = None,
                 mode: str = 'random',
                 iteration_order: int = None,
                 name: str = None,
                 metadata: str = 'features'):
        """Initialize k-mutation ORF pool.
        
        Args:
            seq: Input sequence to mutate. Can be:
                - str: Full DNA sequence (ACGT only). Can include flanking regions (UTRs).
                - Pool: Pool instance whose sequence will be mutated (creates dependency).
            k: Number of codons to mutate
            mutation_type: Type of mutation (see class docstring)
            positions: Optional list of codon indices eligible for mutation. 
                Default: all codons are eligible.
            orf_start: Start index of ORF within seq (0-based, inclusive). Default: 0
            orf_end: End index of ORF within seq (0-based, exclusive). Default: len(seq)
            codon_table: Genetic code specification:
                - 'standard': Use standard genetic code (default, cached for speed)
                - dict: Custom genetic code as {AA: [codon, ...]}
                - None: Use standard genetic code
            mark_changes: If True, apply swapcase() to mutated codons
            max_num_states: Maximum number of states before treating as infinite
            mode: Either 'random' or 'sequential'. Default: 'random'
            iteration_order: Order for sequential iteration. Default: auto-assigned
            name: Optional name for the pool
        
        Raises:
            ValueError: If k is invalid, mutation_type is invalid, or ORF validation fails
        """
         # Treat None exactly like 'standard' since this subclass always needs codon lookups
        if codon_table is None:
            codon_table = 'standard'

        # Handle both Pool and str inputs
        if isinstance(seq, Pool):
            # TRANSFORMER pattern: extract sequence from Pool
            actual_seq = seq.seq
            parents_to_pass = (seq,)  # Track Pool as parent (creates dependency)
        else:
            # SOURCE pattern: use string directly
            actual_seq = seq
            parents_to_pass = (seq,)  # Track string as parent (for reference)
        
        allow_lowercase_input = isinstance(seq, Pool)

        # Handles all ORF setup, validation, codon tables
        super().__init__(
            seq=actual_seq,
            orf_start=orf_start,
            orf_end=orf_end,
            codon_table=codon_table,
            mark_changes=mark_changes,
            parents=parents_to_pass,
            op='k_mutate_orf',
            max_num_states=max_num_states,
            mode=mode,
            iteration_order=iteration_order,
            name=name,
            allow_lowercase=allow_lowercase_input,
            metadata=metadata
        )
                
        # Validate and store eligible codon positions
        if positions is None:
            # Default: all codon positions are eligible for mutation
            self.positions = list(range(self.num_codons))
        else:
            # Validate provided positions
            if not positions or len(positions) == 0:
                raise ValueError("positions must be a non-empty list")
            for pos in positions:
                if not isinstance(pos, int):
                    raise ValueError(f"All positions must be integers, got {type(pos)}")
                if pos < 0 or pos >= self.num_codons:
                    raise ValueError(
                        f"Position {pos} is out of bounds. "
                        f"Valid range: [0, {self.num_codons})"
                    )
            # Check for duplicates
            if len(positions) != len(set(positions)):
                raise ValueError("positions must not contain duplicates")
            self.positions = list(positions)
        
        # Validate k
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if k > len(self.positions):
            raise ValueError(
                f"k ({k}) must be <= number of available positions ({len(self.positions)})"
            )
        
        self.k = k
        
        # Validate mutation_type
        valid_types = list(self.mutation_lookup.keys())
        if mutation_type not in valid_types:
            raise ValueError(f"mutation_type must be one of {valid_types}, got '{mutation_type}'")
        
        self.mutation_type = mutation_type
        
        # For nonsense, validate no stop codons in input
        if mutation_type == 'nonsense':
            for i, codon in enumerate(self.codons):
                if codon.upper() in self.stop_codons:
                    raise ValueError(
                        f"ORF contains stop codon '{codon}' at position {i} "
                        f"(nonsense type requires no stop codons in input)"
                    )
        
        # Pre-compute mutation counts per codon, this will be computed live for non-uniform mutation types with pool input
        self.num_possible_mutations = [self._get_num_mutations(c, mutation_type) for c in self.codons]
        
        # Check uniformity (all positions have same number of mutations)
        mutation_counts = [self.num_possible_mutations[p] for p in self.positions]
        self.is_uniform = len(set(mutation_counts)) == 1
        self.uniform_num_possible_mutations = mutation_counts[0] if self.is_uniform else None
        
        if mode == 'sequential' and not self.is_uniform:
            raise ValueError(
                f"mode='sequential' requires a uniform mutation type; got '{mutation_type}'"
            )

        # Sequential-mode caches are built lazily
        self._sequential_cache_ready = False
        if self.is_uniform and mode == 'sequential':
            self._cache_sequential_components()
        
        # Design card metadata caching
        self._cached_codon_pos = []
        self._cached_codon_from = []
        self._cached_codon_to = []
        self._cached_state: int | None = None
        
        # Refresh state counts now that all subclass fields are set
        # (Pool.__init__ called _calculate_num_internal_states early, so update it here)
        self.num_internal_states = self._calculate_num_internal_states()
        # Also update num_states since it depends on num_internal_states
        self.num_states = self._calculate_num_states()
    
    def _mutate_codon(self, codon: str, mutation_type: str, state: int) -> Union[str, None]:
        """Mutate a codon (internal method, assumes valid inputs)."""
        available_mutations = self.mutation_lookup[mutation_type][codon.upper()]
        if len(available_mutations) == 0:
            return None
        return available_mutations[state % len(available_mutations)]
    
    def _get_num_mutations(self, codon: str, mutation_type: str) -> int:
        """Get number of possible mutations (internal method, O(1))."""
        return len(self.mutation_lookup[mutation_type][codon.upper()])
    
    def _cache_sequential_components(self):
        """Pre-compute position combinations for sequential mode (~100x speedup)."""
        self._num_mutation_patterns = self.uniform_num_possible_mutations ** self.k
        self._position_combinations = list(combinations(self.positions, self.k))
        self._sequential_cache_ready = True

    def _calculate_num_internal_states(self) -> Union[int, float]:
        """Calculate state space size.
        
        Note: Pool.__init__ may call this before our __init__ finishes,
        so we check if initialization is complete and return a temporary value.
        We then manually update num_internal_states after completing setup.
        """
        # Check if initialization is complete
        if not hasattr(self, 'is_uniform'):
            return 1  # Temporary value, updated after __init__ completes
        
        if not self.is_uniform:
            return float('inf')
        return comb(len(self.positions), self.k) * (self.uniform_num_possible_mutations ** self.k)
    
    def _compute_seq(self) -> str:
        """Compute mutated sequence (optimized for sequential mode).
        
        Handles both SOURCE pattern (string parent) and TRANSFORMER pattern (Pool parent).
        When parent is a Pool, we extract its current sequence and re-split into codons.
        Also caches mutation details for design card metadata.
        """
        # Get base sequence from parent (could be Pool or str)
        if isinstance(self.parents[0], str):
            # SOURCE: parent is the original string, use stored codons
            base_codons = self.codons.copy()
        else:
            # TRANSFORMER: parent is a Pool, get its current sequence
            # Re-extract ORF region (parent's sequence might have changed)
            parent_seq = self.parents[0].seq
            # Use same orf_start/orf_end we used during __init__
            # (stored implicitly via upstream_flank length)
            orf_start_idx = len(self.upstream_flank)
            orf_end_idx = orf_start_idx + len(self.orf_seq)
            current_orf = parent_seq[orf_start_idx:orf_end_idx]
            # Split into codons
            base_codons = [current_orf[i:i+3] for i in range(0, len(current_orf), 3)]
        
        # Use standard state wrapping pattern (handles float states from inf pools)
        state = self.get_state() % self.num_internal_states if self.num_internal_states > 0 else 0
        mutated_codons = base_codons.copy()
        
        # Track mutations for design cards
        mutation_positions = []
        mutation_from = []
        mutation_to = []
        
        if self.is_uniform and self.mode == 'sequential':
            if not self._sequential_cache_ready:
                self._cache_sequential_components()
                
            # Sequential: use deterministic decomposition
            num_mutation_patterns = self._num_mutation_patterns
            
            # Mixed-radix decomposition
            position_index = state // num_mutation_patterns
            mutation_pattern_index = state % num_mutation_patterns
            
            # Get positions from CACHED list (O(1), no regeneration!)
            positions = self._position_combinations[position_index]
            
            # Apply mutations (iterate in reverse for mixed-radix decomposition)
            for i in range(self.k - 1, -1, -1):
                pos = positions[i]
                original_codon = base_codons[pos].upper()
                mutation_index = mutation_pattern_index % self.uniform_num_possible_mutations
                mutation_pattern_index //= self.uniform_num_possible_mutations
                mutated_codon = self._mutate_codon(original_codon, self.mutation_type, mutation_index)
                if mutated_codon is not None:
                    mutated_codons[pos] = mutated_codon
                    mutation_positions.append(pos)
                    mutation_from.append(original_codon)
                    mutation_to.append(mutated_codon)
        else:
            # Random mode
            rng = random.Random(state)
            parent_is_pool = isinstance(self.parents[0], Pool)
            positions_to_mutate = rng.sample(self.positions, self.k)
            for pos in positions_to_mutate:
                original_codon = base_codons[pos].upper()
                if parent_is_pool and not self.is_uniform:
                    # Non-uniform + transformer: recompute from current codon
                    num_alternatives = self._get_num_mutations(
                        original_codon, self.mutation_type
                    )
                    self.num_possible_mutations[pos] = num_alternatives
                else:
                    num_alternatives = self.num_possible_mutations[pos]

                if num_alternatives > 0:
                    mutation_index = rng.randint(0, num_alternatives - 1)
                    mutated_codon = self._mutate_codon(
                        original_codon, self.mutation_type, mutation_index
                    )
                    if mutated_codon is not None:
                        mutated_codons[pos] = mutated_codon
                        mutation_positions.append(pos)
                        mutation_from.append(original_codon)
                        mutation_to.append(mutated_codon)
        
        # Cache mutation details for design cards (before case change)
        self._cached_codon_pos = mutation_positions
        self._cached_codon_from = mutation_from
        self._cached_codon_to = mutation_to
        self._cached_state = self.get_state()
        
        # Apply case change if requested (using inherited self.mark_changes)
        if self.mark_changes:
            mutated_codons = [
                mutated_codons[i].swapcase() if mutated_codons[i] != base_codons[i] else mutated_codons[i]
                for i in range(len(mutated_codons))
            ]
        
        # Use inherited method to reassemble with flanks
        return self._reassemble_with_flanks(''.join(mutated_codons))
    
    def get_metadata(self, abs_start: int, abs_end: int) -> dict:
        """Get design card metadata for current state.
        
        Extends base Pool metadata with codon-level mutation information.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end only
            - 'features': core + codon_pos, codon_pos_abs, codon_from, codon_to, aa_from, aa_to (default)
            - 'complete': features + value
        
        Args:
            abs_start: Absolute start position of this pool in the final sequence.
            abs_end: Absolute end position of this pool in the final sequence.
        
        Returns:
            Dictionary with metadata fields based on metadata level.
        """
        # Ensure cached values are current (recompute if state changed)
        if self._cached_state != self.get_state():
            _ = self.seq  # Trigger computation to populate cache
        
        # Get base metadata (handles core fields and 'complete' level value)
        metadata = super().get_metadata(abs_start, abs_end)
        
        # Add KMutationORFPool-specific fields for 'features' and 'complete' levels
        if self._metadata_level in ('features', 'complete'):
            # Compute codon_pos_abs: abs_start + upstream_flank + codon_pos * 3
            upstream_len = len(self.upstream_flank)
            codon_pos_abs = [
                abs_start + upstream_len + pos * 3 
                for pos in self._cached_codon_pos
            ] if abs_start is not None else None
            
            # Look up amino acids from codon table
            aa_from = [self.codon_to_aa_dict.get(c.upper(), '?') for c in self._cached_codon_from]
            aa_to = [self.codon_to_aa_dict.get(c.upper(), '?') for c in self._cached_codon_to]
            
            metadata['codon_pos'] = self._cached_codon_pos.copy()
            metadata['codon_pos_abs'] = codon_pos_abs
            metadata['codon_from'] = self._cached_codon_from.copy()
            metadata['codon_to'] = self._cached_codon_to.copy()
            metadata['aa_from'] = aa_from
            metadata['aa_to'] = aa_to
        
        return metadata
    
    def __repr__(self) -> str:
        seq_preview = self.orf_seq[:12] + "..." if len(self.orf_seq) > 12 else self.orf_seq
        return f"KMutationORFPool(seq={seq_preview}, k={self.k}, type='{self.mutation_type}')"

# Legacy
# class KMutationORFPool(Pool):
#     """A class for generating ORF sequences with exactly k codon-level mutations.
    
#     Introduces exactly k mutations at the codon level (not nucleotide level) in a
#     DNA open reading frame. Supports multiple mutation types including missense,
#     synonymous, nonsense, and all-codon mutations.
    
#     Sequential mode is only compatible with mutation types that have uniform
#     num_possible_mutations across all codon positions (missense_first_codon,
#     all_by_codon, nonsense). Other types use random mode only.
#     """
    
#     def __init__(self,
#                  orf_seq: str,
#                  k: int,
#                  mutation_type: str,
#                  change_case_of_mutations: bool = False,
#                  max_num_states: int = None,
#                  mode: str = 'random',
#                  iteration_order: int | None = None,
#                  name: str | None = None):
#         """Initialize a KMutationORFPool.
        
#         Args:
#             orf_seq: DNA open reading frame sequence (must be ACGT only and length divisible by 3)
#             k: Number of codons to mutate (must be > 0 and <= number of codons)
#             mutation_type: Type of mutation to perform. Must be one of:
#                 - 'missense_first_codon': Mutate to different amino acid, use first codon (uniform, 20 alternatives)
#                 - 'missense_random_codon': Mutate to different amino acid, random codon (non-uniform)
#                 - 'all_by_codon': Mutate to any different codon (uniform, 63 alternatives)
#                 - 'synonymous': Mutate to synonymous codon (non-uniform)
#                 - 'nonsense': Mutate to stop codon (uniform, 3 alternatives, ORF must not contain stops)
#             change_case_of_mutations: If True, apply swapcase() to mutated codons (default: False)
#             max_num_states: Maximum number of states before treating as infinite
#             mode: Either 'random' or 'sequential' (default: 'random')
#             iteration_order: Order for sequential iteration (default: auto-assigned based on creation order)
        
#         Raises:
#             ValueError: If orf_seq is invalid, k is invalid, mutation_type is invalid,
#                 or ORF contains stop codons when using 'nonsense' type
#         """
#         # Validate orf_seq is DNA
#         if not isinstance(orf_seq, str):
#             raise ValueError("orf_seq must be a string")
        
#         if not all(c in 'ACGT' for c in orf_seq):
#             raise ValueError(f"orf_seq must contain only ACGT characters, got '{orf_seq}'")
        
#         # Validate length is divisible by 3
#         if len(orf_seq) % 3 != 0:
#             raise ValueError(
#                 f"orf_seq length must be divisible by 3, got length {len(orf_seq)}"
#             )
        
#         # Validate k
#         if k <= 0:
#             raise ValueError(f"k must be > 0, got {k}")
        
#         # Split into codons
#         self.codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
#         num_codons = len(self.codons)
        
#         if k > num_codons:
#             raise ValueError(
#                 f"k ({k}) must be <= number of codons ({num_codons})"
#             )
        
#         # Validate mutation_type
#         valid_mutation_types = [
#             'missense_first_codon', 'missense_random_codon',
#             'all_by_codon', 'synonymous', 'nonsense'
#         ]
#         if mutation_type not in valid_mutation_types:
#             raise ValueError(
#                 f"mutation_type must be one of {valid_mutation_types}, got '{mutation_type}'"
#             )
        
#         self.mutation_type = mutation_type
        
#         # Store change_case_of_mutations flag
#         self.change_case_of_mutations = change_case_of_mutations
        
#         # For nonsense type, validate that ORF contains no stop codons
#         if mutation_type == 'nonsense':
#             for i, codon in enumerate(self.codons):
#                 if codon in stop_codons:
#                     raise ValueError(
#                         f"ORF contains stop codon '{codon}' at position {i} "
#                         f"(mutation_type='nonsense' requires no stop codons in input)"
#                     )
        
#         self.k = k
        
#         # Compute num_possible_mutations for each codon position
#         self.num_possible_mutations = self._compute_num_possible_mutations()
        
#         # Check if num_possible_mutations is uniform (required for sequential mode)
#         self.is_uniform = len(set(self.num_possible_mutations)) == 1
        
#         if self.is_uniform:
#             self.uniform_num_possible_mutations = self.num_possible_mutations[0]
#         else:
#             self.uniform_num_possible_mutations = None
        
#         # Store the original orf_seq for reference
#         self.orf_seq = orf_seq
        
#         # Call parent constructor
#         super().__init__(
#             parents=(orf_seq,),
#             op='k_mutate_orf',
#             max_num_states=max_num_states,
#             mode=mode,
#             iteration_order=iteration_order,
#             name=name
#         )
    
#     def _compute_num_possible_mutations(self) -> List[int]:
#         """Compute the number of possible mutations for each codon position.
        
#         Returns:
#             List of integers, one per codon, indicating how many mutations are possible
#             at that position given the mutation_type.
#         """
#         num_mutations = []
        
#         for codon in self.codons:
#             if self.mutation_type == 'missense_first_codon':
#                 # 21 amino acids - 1 (current) = 20 alternatives
#                 current_aa = codon_to_aa_dict[codon]
#                 num_alternatives = 20 if current_aa != '*' else 20
#                 # Actually, it's all 21 AAs minus the current one
#                 num_alternatives = 20  # 21 total AAs - 1 current = 20
                
#             elif self.mutation_type == 'missense_random_codon':
#                 # Variable: depends on current amino acid
#                 current_aa = codon_to_aa_dict[codon]
#                 # Count total codons for all other amino acids
#                 from .utils import aa_to_codon_dict
#                 num_alternatives = 0
#                 for aa, codon_list in aa_to_codon_dict.items():
#                     if aa != current_aa:
#                         num_alternatives += len(codon_list)
                
#             elif self.mutation_type == 'all_by_codon':
#                 # 64 codons - 1 (current) = 63 alternatives
#                 num_alternatives = 63
                
#             elif self.mutation_type == 'synonymous':
#                 # Variable: depends on synonymous codons available
#                 from .utils import codon_to_synonymous_dict
#                 num_alternatives = len(codon_to_synonymous_dict[codon])
                
#             elif self.mutation_type == 'nonsense':
#                 # 3 stop codons (validated that current is not a stop)
#                 num_alternatives = 3
            
#             num_mutations.append(num_alternatives)
        
#         return num_mutations
    
#     def _calculate_seq_length(self) -> int:
#         """ORF mutations don't change length - return original length."""
#         return len(self.orf_seq)
    
#     def _calculate_num_internal_states(self) -> Union[int, float]:
#         """Calculate the number of states for KMutationORFPool.
        
#         Returns:
#             - If mutation type produces uniform num_possible_mutations:
#               comb(L, k) * uniform_num_possible_mutations ** k
#             - Otherwise: float('inf') (only random mode supported)
#         """
#         if not self.is_uniform:
#             # Non-uniform: infinite states (random mode only)
#             return float('inf')
        
#         # Uniform: calculate finite states
#         L = len(self.codons)
#         num_position_choices = comb(L, self.k)
#         num_mutation_patterns = self.uniform_num_possible_mutations ** self.k
        
#         return num_position_choices * num_mutation_patterns
    
#     def _compute_seq(self) -> str:
#         """Compute mutated ORF sequence with exactly k codon mutations.
        
#         Returns:
#             DNA sequence with exactly k codons mutated according to mutation_type
#         """
#         state = self.get_state()
#         num_codons = len(self.codons)
        
#         # Make a copy of codons to mutate
#         mutated_codons = self.codons.copy()
        
#         if self.is_uniform and self.mode == 'sequential':
#             # Sequential mode with uniform mutations
#             num_mutation_patterns = self.uniform_num_possible_mutations ** self.k
#             num_position_choices = comb(num_codons, self.k)
            
#             # Wrap state within valid range
#             valid_states = num_position_choices * num_mutation_patterns
#             wrapped_state = state % valid_states
            
#             # Decompose state
#             position_idx = wrapped_state // num_mutation_patterns
#             mutation_pattern_idx = wrapped_state % num_mutation_patterns
            
#             # Get the k positions to mutate
#             all_combinations = list(combinations(range(num_codons), self.k))
#             positions = list(all_combinations[position_idx])
            
#             # Apply mutations to selected positions
#             # Rightmost position varies most rapidly
#             for i in range(self.k - 1, -1, -1):
#                 pos = positions[i]
#                 original_codon = self.codons[pos]
                
#                 # Get the specific mutation index for this position
#                 mutation_idx = mutation_pattern_idx % self.uniform_num_possible_mutations
#                 mutation_pattern_idx //= self.uniform_num_possible_mutations
                
#                 # Get the mutated codon
#                 mutated_codon = mutate_codon(original_codon, self.mutation_type, state=mutation_idx)
#                 if mutated_codon is not None:
#                     mutated_codons[pos] = mutated_codon
        
#         else:
#             # Random mode (or non-uniform mutations)
#             # Use state as random seed
#             local_rng = random.Random(state)
            
#             # Randomly select k positions to mutate
#             positions_to_mutate = local_rng.sample(range(num_codons), self.k)
            
#             # Mutate each selected position
#             for pos in positions_to_mutate:
#                 original_codon = self.codons[pos]
#                 num_alternatives = self.num_possible_mutations[pos]
                
#                 if num_alternatives > 0:
#                     # Generate a random mutation index
#                     mutation_idx = local_rng.randint(0, num_alternatives - 1)
                    
#                     # Get the mutated codon
#                     mutated_codon = mutate_codon(original_codon, self.mutation_type, state=mutation_idx)
#                     if mutated_codon is not None:
#                         mutated_codons[pos] = mutated_codon
        
#         # Apply case change to mutated codons if requested
#         if self.change_case_of_mutations:
#             mutated_codons = [
#                 mutated_codons[i].swapcase() if mutated_codons[i] != self.codons[i] else mutated_codons[i]
#                 for i in range(len(mutated_codons))
#             ]
        
#         # Reconstruct DNA sequence
#         return ''.join(mutated_codons)
    
#     def __repr__(self) -> str:
#         seq_preview = self.orf_seq[:12] + "..." if len(self.orf_seq) > 12 else self.orf_seq
#         return f"KMutationORFPool(seq={seq_preview}, k={self.k}, mutation_type='{self.mutation_type}')"

