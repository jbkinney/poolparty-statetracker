import random
from typing import Union, List
from .pool import Pool
#from .utils import mutate_codon, codon_to_aa_dict, stop_codons
from .orf_pool import ORFPool

class RandomMutationORFPool(ORFPool):
    """Generate ORF sequences where each codon mutates independently at a given rate.

    Supports both SOURCE inputs (raw strings) and TRANSFORMER inputs (another Pool).
    Output always preserves upstream/downstream flanks managed by ORFPool.
    """

    def __init__(
        self,
        seq: Union[str, Pool],
        mutation_type: str,
        mutation_rate: Union[float, List[float]] = 0.1,
        mark_changes: bool = False,
        mode: str = 'random',
        iteration_order: int | None = None,
        name: str | None = None,
        orf_start: int = 0,
        orf_end: int | None = None,
        codon_table: Union[dict, str, None] = 'standard',
        max_num_states: int | None = None,
        metadata: str = 'features',
    ):
        if mode == 'sequential':
            raise ValueError("RandomMutationORFPool only supports mode='random'")

        if codon_table is None:
            codon_table = 'standard'

        if isinstance(seq, Pool):
            actual_seq = seq.seq
            parents_to_pass = (seq,)
            allow_lowercase = True
        else:
            actual_seq = seq
            parents_to_pass = (seq,)
            allow_lowercase = False

        super().__init__(
            seq=actual_seq,
            orf_start=orf_start,
            orf_end=orf_end,
            codon_table=codon_table,
            mark_changes=mark_changes,
            parents=parents_to_pass,
            op='random_mutate_orf',
            max_num_states=max_num_states,
            mode='random',
            iteration_order=iteration_order,
            name=name,
            allow_lowercase=allow_lowercase,
            metadata=metadata,
        )

        valid_mutation_types = [
            'any_codon',
            'nonsynonymous_first',
            'nonsynonymous_random',
            'missense_only_first',
            'missense_only_random',
            'synonymous',
            'nonsense',
        ]
        if mutation_type not in valid_mutation_types:
            raise ValueError(
                f"mutation_type must be one of {valid_mutation_types}, got '{mutation_type}'"
            )
        self.mutation_type = mutation_type

        if mutation_type == 'nonsense':
            for i, codon in enumerate(self.codons):
                if codon.upper() in self.stop_codons:
                    raise ValueError(
                        f"ORF contains stop codon '{codon}' at position {i} "
                        "(nonsense type requires no stop codons in input)"
                    )

        if isinstance(mutation_rate, (list, tuple)):
            if len(mutation_rate) != self.num_codons:
                raise ValueError(
                    f"mutation_rate array length ({len(mutation_rate)}) "
                    f"must match number of codons ({self.num_codons})"
                )
            for rate in mutation_rate:
                if not 0 <= rate <= 1:
                    raise ValueError(
                        f"mutation_rate values must be between 0 and 1, got {rate}"
                    )
            self.mutation_rate = list(mutation_rate)
            self._is_uniform_rate = False
        else:
            if not 0 <= mutation_rate <= 1:
                raise ValueError(
                    f"mutation_rate must be between 0 and 1, got {mutation_rate}"
                )
            self.mutation_rate = mutation_rate
            self._is_uniform_rate = True
        
        # Design card metadata caching
        self._cached_mut_count = 0
        self._cached_codon_pos = []
        self._cached_codon_from = []
        self._cached_codon_to = []
        self._cached_state: int | None = None

    def _calculate_num_internal_states(self) -> float:
        return float('inf')

    def _mutate_codon(self, codon: str, mutation_type: str, state: int) -> Union[str, None]:
        available = self.mutation_lookup[mutation_type][codon.upper()]
        if not available:
            return None
        return available[state % len(available)]

    def _compute_seq(self) -> str:
        """Compute randomly mutated sequence and cache mutation details for design cards."""
        if isinstance(self.parents[0], str):
            base_codons = self.codons.copy()
        else:
            parent_seq = self.parents[0].seq
            orf_start_idx = len(self.upstream_flank)
            orf_end_idx = orf_start_idx + len(self.orf_seq)
            current_orf = parent_seq[orf_start_idx:orf_end_idx]
            base_codons = [
                current_orf[i:i + 3] for i in range(0, len(current_orf), 3)
            ]

        if not self._is_uniform_rate and len(base_codons) != len(self.mutation_rate):
            raise ValueError(
                "mutation_rate length no longer matches codon count after upstream modifications"
            )

        rng = random.Random(self.get_state())
        mutated_codons = base_codons.copy()
        
        # Track mutations for design cards
        mutation_positions = []
        mutation_from = []
        mutation_to = []

        for idx, codon in enumerate(base_codons):
            rate = self.mutation_rate if self._is_uniform_rate else self.mutation_rate[idx]
            if rng.random() < rate:
                mutation_idx = rng.randint(0, 10**6)
                original_codon = codon.upper()
                mutated_codon = self._mutate_codon(original_codon, self.mutation_type, mutation_idx)
                if mutated_codon is not None:
                    mutated_codons[idx] = mutated_codon
                    mutation_positions.append(idx)
                    mutation_from.append(original_codon)
                    mutation_to.append(mutated_codon)
        
        # Cache mutation details for design cards (before case change)
        self._cached_mut_count = len(mutation_positions)
        self._cached_codon_pos = mutation_positions
        self._cached_codon_from = mutation_from
        self._cached_codon_to = mutation_to
        self._cached_state = self.get_state()

        if self.mark_changes:
            mutated_codons = [
                mutated_codons[i].swapcase() if mutated_codons[i] != base_codons[i]
                else mutated_codons[i]
                for i in range(len(mutated_codons))
            ]

        return self._reassemble_with_flanks(''.join(mutated_codons))

    def get_metadata(self, abs_start: int, abs_end: int) -> dict:
        """Get design card metadata for current state.
        
        Extends base Pool metadata with random codon-level mutation information.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end only
            - 'features': core + mut_count, codon_pos, codon_pos_abs, codon_from, codon_to, aa_from, aa_to (default)
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
        
        # Add RandomMutationORFPool-specific fields for 'features' and 'complete' levels
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
            
            metadata['mut_count'] = self._cached_mut_count
            metadata['codon_pos'] = self._cached_codon_pos.copy()
            metadata['codon_pos_abs'] = codon_pos_abs
            metadata['codon_from'] = self._cached_codon_from.copy()
            metadata['codon_to'] = self._cached_codon_to.copy()
            metadata['aa_from'] = aa_from
            metadata['aa_to'] = aa_to
        
        return metadata

    def __repr__(self) -> str:
        seq_preview = self.orf_seq[:12] + "..." if len(self.orf_seq) > 12 else self.orf_seq
        rate_str = f"{self.mutation_rate}" if self._is_uniform_rate else f"[{len(self.mutation_rate)} rates]"
        return (
            f"RandomMutationORFPool(seq={seq_preview}, "
            f"mutation_type='{self.mutation_type}', rate={rate_str})"
        )

# Legacy
# class RandomMutationORFPool(Pool):
#     """A class for generating randomly mutated ORF sequences at the codon level.
    
#     Each codon in the ORF independently has a probability of being mutated
#     according to the specified mutation_type. Has infinite states.
#     """
#     def __init__(self, 
#                  orf_seq: str, 
#                  mutation_type: str,
#                  mutation_rate: Union[float, List[float]] = 0.1,
#                  change_case_of_mutations: bool = False,
#                  mode: str = 'random',
#                  iteration_order: int | None = None,
#                  name: str | None = None):
#         """Initialize a RandomMutationORFPool.
        
#         Args:
#             orf_seq: DNA open reading frame sequence (must be ACGT only and length divisible by 3)
#             mutation_type: Type of mutation to perform. Must be one of:
#                 - 'missense_first_codon': Mutate to different amino acid, use first codon
#                 - 'missense_random_codon': Mutate to different amino acid, random codon
#                 - 'all_by_codon': Mutate to any different codon
#                 - 'synonymous': Mutate to synonymous codon
#                 - 'nonsense': Mutate to stop codon
#             mutation_rate: Probability of mutation at each codon position (0-1).
#                 Can be a single float for uniform rate, or a list of floats
#                 for position-specific rates. Default: 0.1
#             change_case_of_mutations: If True, apply swapcase() to mutated codons (default: False)
#             mode: Either 'random' or 'sequential' (default: 'random')
#             iteration_order: Order for sequential iteration (default: auto-assigned based on creation order)
                
#         Raises:
#             ValueError: If orf_seq is invalid, mutation_type is invalid,
#                 mutation_rate is out of range, or mutation_rate array length doesn't match codon count
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
        
#         # Split into codons
#         self.codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq), 3)]
#         num_codons = len(self.codons)
        
#         # For nonsense type, validate that ORF contains no stop codons
#         if mutation_type == 'nonsense':
#             for i, codon in enumerate(self.codons):
#                 if codon in stop_codons:
#                     raise ValueError(
#                         f"ORF contains stop codon '{codon}' at position {i} "
#                         f"(mutation_type='nonsense' requires no stop codons in input)"
#                     )
        
#         # Validate and store mutation_rate
#         if isinstance(mutation_rate, (list, tuple)):
#             # Position-specific rates
#             for rate in mutation_rate:
#                 if not 0 <= rate <= 1:
#                     raise ValueError(f"mutation_rate values must be between 0 and 1, got {rate}")
#             if len(mutation_rate) != num_codons:
#                 raise ValueError(
#                     f"mutation_rate array length ({len(mutation_rate)}) "
#                     f"must match number of codons ({num_codons})"
#                 )
#             self.mutation_rate = list(mutation_rate)
#             self._is_uniform_rate = False
#         else:
#             # Uniform rate
#             if not 0 <= mutation_rate <= 1:
#                 raise ValueError(f"mutation_rate must be between 0 and 1, got {mutation_rate}")
#             self.mutation_rate = mutation_rate
#             self._is_uniform_rate = True
        
#         # Store change_case_of_mutations flag
#         self.change_case_of_mutations = change_case_of_mutations
        
#         # Store the original orf_seq for reference
#         self.orf_seq = orf_seq
        
#         super().__init__(parents=(orf_seq,), op='mutate_orf', mode=mode, iteration_order=iteration_order, name=name)
    
#     def _calculate_num_internal_states(self) -> float:
#         """RandomMutationORFPool has infinite internal states (different random mutations)."""
#         return float('inf')
    
#     def _calculate_seq_length(self) -> int:
#         """ORF mutations don't change length - return original length."""
#         return len(self.orf_seq)
    
#     def _compute_seq(self) -> str:
#         """Compute mutated ORF sequence based on current state.
        
#         Returns:
#             Mutated version of the input ORF sequence
#         """
#         # Create local Random instance to avoid polluting global random state
#         local_rng = random.Random(self.get_state())
        
#         # Generate mutations at codon level
#         result_codons = []
#         for i, codon in enumerate(self.codons):
#             # Get mutation rate for this codon position
#             rate = self.mutation_rate if self._is_uniform_rate else self.mutation_rate[i]
            
#             # Check if codon should mutate
#             if local_rng.random() < rate:
#                 # Mutate the codon using the mutate_codon utility
#                 # Generate a random mutation index for this codon
#                 mutation_idx = local_rng.randint(0, 1000000)
#                 mutated_codon = mutate_codon(codon, self.mutation_type, state=mutation_idx)
                
#                 if mutated_codon is not None:
#                     result_codons.append(mutated_codon)
#                 else:
#                     # If mutation not possible, keep original
#                     result_codons.append(codon)
#             else:
#                 # No mutation
#                 result_codons.append(codon)
        
#         # Apply case change to mutated codons if requested
#         if self.change_case_of_mutations:
#             result_codons = [
#                 result_codons[i].swapcase() if result_codons[i] != self.codons[i] else result_codons[i]
#                 for i in range(len(result_codons))
#             ]
        
#         return ''.join(result_codons)
    
#     def __repr__(self) -> str:
#         seq_preview = self.orf_seq[:12] + "..." if len(self.orf_seq) > 12 else self.orf_seq
#         if self._is_uniform_rate:
#             rate_str = f"{self.mutation_rate}"
#         else:
#             rate_str = f"[{len(self.mutation_rate)} rates]"
#         return f"RandomMutationORFPool(seq={seq_preview}, mutation_type='{self.mutation_type}', rate={rate_str})"

