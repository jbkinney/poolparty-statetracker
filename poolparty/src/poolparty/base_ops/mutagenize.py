"""Mutagenize operation - apply mutations to a sequence."""
from itertools import combinations
from math import comb, prod
from ..types import Union, ModeType, Optional, Real, Integral, Sequence, RegionType, beartype, StyleList
from ..operation import Operation
from ..pool import Pool
from ..party import get_active_party
from ..utils import dna_utils
import numpy as np


@beartype
def mutagenize(
    pool: Union[Pool, str],
    region: RegionType = None,
    num_mutations: Optional[Integral] = None,
    mutation_rate: Optional[Real] = None,
    allowed_chars: Optional[str] = None,
    style: Optional[str] = None,
    prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_states: Optional[int] = None,
    iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = 'mutagenize',
) -> Pool:
    """
    Create a Pool that applies mutations to a sequence.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent pool or sequence string to mutate.
    region : Union[str, Sequence[Integral], None], default=None
        Region to mutagenize. Can be a marker name (str), explicit interval [start, stop],
        or None to mutagenize entire sequence. Positions are region-relative.
    num_mutations : Optional[Integral], default=None
        Fixed number of mutations to apply (mutually exclusive with mutation_rate).
    mutation_rate : Optional[Real], default=None
        Probability of mutation at each position (mutually exclusive with num_mutations).
    allowed_chars : Optional[str], default=None
        IUPAC string of same length as sequence, specifying allowed bases at each position.
        Each character is an IUPAC code (A, C, G, T, R, Y, S, W, K, M, B, D, H, V, N).
        Positions where only the wild-type is allowed are treated as non-mutable.
    style : Optional[str], default=None
        Style to apply to mutated positions.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : ModeType, default='random'
        Selection mode: 'random' or 'sequential'. Sequential only available with num_mutations.
    num_states : Optional[int], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order for the Operation.

    Returns
    -------
    Pool
        A Pool that generates mutated sequences.
    """
    
    from ..fixed_ops.from_seq import from_seq
    pool = from_seq(pool, _factory_name=f'{_factory_name}(from_seq)') if isinstance(pool, str) else pool
    op = MutagenizeOp(
        pool=pool,
        num_mutations=num_mutations,
        mutation_rate=mutation_rate,
        allowed_chars=allowed_chars,
        region=region,
        style=style,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        _factory_name=_factory_name,
    )
    pool = Pool(operation=op)
    return pool


@beartype
class MutagenizeOp(Operation):
    """Apply mutations to a parent sequence or a specified region within it.
    
    Supports two mutation modes:
    - num_mutations: Apply exactly this many mutations to each sequence
    - mutation_rate: Apply a random number of mutations based on a binomial distribution
    
    Exactly one of num_mutations or mutation_rate must be provided.
    Sequential mode is only available when num_mutations is specified.
    """
    factory_name = "mutagenize"
    design_card_keys = ['positions', 'wt_chars', 'mut_chars']
    
    def __init__(
        self,
        pool: Pool,
        num_mutations: Optional[Integral] = None,
        mutation_rate: Optional[Real] = None,
        allowed_chars: Optional[str] = None,
        region: RegionType = None,
        style: Optional[str] = None,
        prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        _factory_name: Optional[str] = 'mutagenize',
    ) -> None:
        # Set factory name if provided
        if _factory_name is not None:
            self.factory_name = _factory_name
            
        # Get alphabet from active Party context
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "mutagenize requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )
        # Validate mutually exclusive parameters
        if num_mutations is None and mutation_rate is None:
            raise ValueError("Either num_mutations or mutation_rate must be provided")
        if num_mutations is not None and mutation_rate is not None:
            raise ValueError("Only one of num_mutations or mutation_rate can be provided, not both")
        
        # Validate num_mutations
        if num_mutations is not None and num_mutations < 1:
            raise ValueError(f"num_mutations must be >= 1, got {num_mutations}")
        
        # Validate mutation_rate
        if mutation_rate is not None:
            if mutation_rate < 0 or mutation_rate > 1:
                raise ValueError(f"mutation_rate must be between 0 and 1, got {mutation_rate}")
            if mode == 'sequential':
                raise ValueError("mode='sequential' is not supported with mutation_rate (use num_mutations instead)")
        
        
        self.num_mutations = num_mutations
        self.mutation_rate = mutation_rate
        self.allowed_chars = allowed_chars
        self._style = style
        self.alpha_size = len(dna_utils.BASES)
        self._mode = mode
        
        # Validate and process allowed_chars if provided
        self._allowed_bases_per_pos = None  # Will be set if allowed_chars is provided
        self._mutation_counts_from_allowed = None  # Pre-computed mutation counts
        if allowed_chars is not None:
            invalid_chars = set()
            allowed_bases_per_pos = []
            mutation_counts = []
            for char in allowed_chars.upper():
                if char in dna_utils.IGNORE_CHARS:
                    continue  # Skip ignore chars (gaps, separators)
                if char not in dna_utils.IUPAC_TO_DNA:
                    invalid_chars.add(char)
                else:
                    bases = set(dna_utils.IUPAC_TO_DNA[char])
                    allowed_bases_per_pos.append(bases)
                    mutation_counts.append(len(bases) - 1)  # -1 for the wt
            if invalid_chars:
                raise ValueError(
                    f"allowed_chars contains invalid IUPAC character(s): {sorted(invalid_chars)}. "
                    f"Valid IUPAC characters are: {sorted(set(dna_utils.IUPAC_TO_DNA.keys()) - set('acgtryswkmbdhvn'))} "
                    f"(plus ignore characters: {sorted(dna_utils.IGNORE_CHARS)})"
                )
            self._allowed_bases_per_pos = allowed_bases_per_pos
            self._mutation_counts_from_allowed = mutation_counts
        
        # Build mutation map: (wt_char, index) -> mut_char
        self._mutation_map = {}
        for wt in dna_utils.VALID_CHARS:
            for i, mut in enumerate(dna_utils.get_mutations(wt)):
                self._mutation_map[(wt, i)] = mut
        
        self._seq_length = pool.seq_length
        self._sequential_cache = None
        self._num_mutable_positions = None  # Actual mutable positions, set on first use
        
        # Determine num_states based on mode
        if mode == 'sequential':
            # Sequential mode only available with num_mutations
            effective_length = self._seq_length
            # If seq_length is unknown but region is a marker, try to get marker's seq_length
            if effective_length is None and isinstance(region, str):
                try:
                    region_obj = party.get_region_by_name(region)
                    effective_length = region_obj.seq_length
                except ValueError:
                    pass  # Region not found, stay with None
            
            # If allowed_chars is provided, use its length and pre-computed mutation counts
            if self._mutation_counts_from_allowed is not None:
                effective_length = len(self._mutation_counts_from_allowed)
                # Filter to positions with at least 1 mutation option
                mutable_counts = [c for c in self._mutation_counts_from_allowed if c > 0]
                num_mutable = len(mutable_counts)
                if num_mutable < num_mutations:
                    raise ValueError(
                        f"{num_mutations=} exceeds mutable positions={num_mutable}. "
                        f"Cannot apply {num_mutations} mutations."
                    )
                num_states = self._build_caches(num_mutable, mutable_counts)
            elif effective_length is not None:
                if effective_length < num_mutations:
                    raise ValueError(
                        f"{num_mutations=} exceeds sequence length={effective_length}. "
                        f"Cannot apply {num_mutations} mutations to a sequence of length {effective_length}."
                    )
                num_states = self._build_caches(effective_length)
            else:
                num_states = 1
        elif mode == 'random':
            # num_states stays None for pure random mode
            pass
        else:
            num_states = 1
        
        super().__init__(
            parent_pools=[pool],
            num_values=num_states,
            mode=mode,
            seq_length=self._seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=region,
        )
    
    def _build_caches(self, num_positions: int, mutation_counts: Optional[list[int]] = None) -> int:
        """Build caches for sequential enumeration.
        
        Parameters
        ----------
        num_positions : int
            Number of mutable positions (valid alphabet characters only).
        mutation_counts : Optional[list[int]]
            Number of valid mutations per position. If None, uses uniform alpha_size-1.
        """
        if num_positions < self.num_mutations:
            raise ValueError(
                f"num_mutations={self.num_mutations} exceeds mutable positions={num_positions}. "
                f"Cannot apply {self.num_mutations} mutations."
            )
        
        if mutation_counts is None:
            # Uniform case: each position has alpha_size - 1 mutations
            alpha_minus_1 = self.alpha_size - 1
            num_combinations = comb(num_positions, self.num_mutations) * (alpha_minus_1 ** self.num_mutations)
            cache = []
            for positions in combinations(range(num_positions), self.num_mutations):
                num_mut_patterns = alpha_minus_1 ** self.num_mutations
                for mut_pattern in range(num_mut_patterns):
                    mut_indices = []
                    remaining = mut_pattern
                    for _ in range(self.num_mutations):
                        mut_indices.append(remaining % alpha_minus_1)
                        remaining //= alpha_minus_1
                    cache.append((positions, tuple(reversed(mut_indices))))
        else:
            # Non-uniform case: each position has different number of mutations
            cache = []
            for positions in combinations(range(num_positions), self.num_mutations):
                counts_for_positions = [mutation_counts[p] for p in positions]
                num_mut_patterns = prod(counts_for_positions)
                for mut_pattern in range(num_mut_patterns):
                    mut_indices = []
                    remaining = mut_pattern
                    for count in counts_for_positions:
                        mut_indices.append(remaining % count)
                        remaining //= count
                    cache.append((positions, tuple(mut_indices)))
            num_combinations = len(cache)
        
        self._sequential_cache = cache
        self._num_mutable_positions = num_positions
        self._mutation_counts = tuple(mutation_counts) if mutation_counts else None
        return num_combinations
    
    def _get_position_mutations(self, seq: str, valid_char_positions: list[int]) -> tuple[list[int], list[list[str]]]:
        """Get mutable positions and their valid mutation options.
        
        Returns a tuple of (mutable_logical_positions, mutation_options_per_position).
        Positions where wt is the only allowed char are excluded.
        
        When allowed_chars is set, also validates that the input sequence has
        allowed characters at each position.
        """
        mutable_positions = []
        mutation_options = []
        
        for logical_pos, raw_pos in enumerate(valid_char_positions):
            wt = seq[raw_pos]
            wt_upper = wt.upper()
            
            if self._allowed_bases_per_pos is not None:
                # Validate length
                if len(self._allowed_bases_per_pos) != len(valid_char_positions):
                    raise ValueError(
                        f"allowed_chars length ({len(self._allowed_bases_per_pos)}) must match "
                        f"sequence length ({len(valid_char_positions)})"
                    )
                # Get pre-computed allowed bases at this position
                allowed_bases_upper = self._allowed_bases_per_pos[logical_pos]
                
                # Validate that wt is in the allowed set
                if wt_upper not in allowed_bases_upper:
                    raise ValueError(
                        f"Sequence character '{wt}' at position {logical_pos} is not in "
                        f"allowed_chars '{self.allowed_chars[logical_pos]}' (allowed: {sorted(allowed_bases_upper)})"
                    )
                
                # Get mutation targets (allowed bases minus wt), preserving case
                if wt.islower():
                    valid_muts = [b.lower() for b in sorted(allowed_bases_upper) if b != wt_upper]
                else:
                    valid_muts = [b for b in sorted(allowed_bases_upper) if b != wt_upper]
            else:
                # No restriction: all non-wt bases are valid
                valid_muts = dna_utils.get_mutations(wt)
            
            if valid_muts:
                mutable_positions.append(logical_pos)
                mutation_options.append(valid_muts)
        
        return mutable_positions, mutation_options
    
    def _random_mutation(self, seq: str, rng: np.random.Generator) -> tuple:
        """Generate random mutation positions (logical) and characters."""
        valid_char_positions = self._get_molecular_positions(seq)
        
        # Get mutable positions and their options (respects allowed_chars)
        mutable_positions, mutation_options = self._get_position_mutations(seq, valid_char_positions)
        num_mutable = len(mutable_positions)
        
        if num_mutable == 0:
            return tuple(), tuple(), tuple()
        
        # Determine number of mutations
        if self.num_mutations is not None:
            num_mut = min(self.num_mutations, num_mutable)
        else:
            # Use binomial distribution based on mutation_rate
            num_mut = rng.binomial(num_mutable, self.mutation_rate)
            if num_mut == 0:
                return tuple(), tuple(), tuple()
        
        # Choose random positions from mutable positions
        chosen_indices = sorted(rng.choice(num_mutable, size=num_mut, replace=False))
        positions = tuple(mutable_positions[i] for i in chosen_indices)
        
        # Determine wild-type and mutant characters using raw positions
        wt_chars = []
        mut_chars = []
        for idx in chosen_indices:
            logical_pos = mutable_positions[idx]
            raw_pos = valid_char_positions[logical_pos]
            wt = seq[raw_pos]
            wt_chars.append(wt)
            # Select randomly from position-specific options
            opts = mutation_options[idx]
            mut = opts[rng.integers(0, len(opts))]
            mut_chars.append(mut)
        
        return positions, tuple(wt_chars), tuple(mut_chars)
    
    def compute(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
        parent_styles: list[StyleList] | None = None,
    ) -> dict:
        """Return design card, mutated sequence, and styles together.
        
        Note: Region handling is done by base class wrapper methods.
        parent_seqs[0] is the region content when region is specified.
        """
        seq = parent_seqs[0]
        valid_char_positions = self._get_molecular_positions(seq)
        
        # Get mutable positions and their options (also validates allowed_chars compatibility)
        mutable_positions, mutation_options = self._get_position_mutations(seq, valid_char_positions)
        num_mutable = len(mutable_positions)
        
        if self.num_mutations is not None and self.num_mutations > num_mutable:
            raise ValueError(f"Cannot apply {self.num_mutations} mutations: only {num_mutable} mutable positions")
        
        if self.mode == 'random':
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)")
            positions, wt_chars, mut_chars = self._random_mutation(seq, rng)
        else:
            # Sequential mode (only available with num_mutations)
            # When allowed_chars is set, cache is pre-built at init time with correct num_states
            # Otherwise, build/rebuild cache based on actual mutable positions
            if self._allowed_bases_per_pos is None:
                # No allowed_chars - may need to build cache dynamically
                if self._sequential_cache is None:
                    self._build_caches(num_mutable)
                    self._num_values = len(self._sequential_cache)
                    self.state._num_values = self._num_values
                elif self._num_mutable_positions != num_mutable:
                    self._build_caches(num_mutable)
                    self._num_values = len(self._sequential_cache)
                    self.state._num_values = self._num_values
            # With allowed_chars, cache was built at init - just use it
            
            # Use state 0 when inactive (state is None)
            state = self.state.value
            state = 0 if state is None else state
            rel_positions, mut_indices = self._sequential_cache[state % len(self._sequential_cache)]
            
            # Map relative positions back to logical positions
            positions = tuple(mutable_positions[p] for p in rel_positions)
            wt_chars = []
            mut_chars = []
            for rel_pos, mut_idx in zip(rel_positions, mut_indices):
                logical_pos = mutable_positions[rel_pos]
                raw_pos = valid_char_positions[logical_pos]
                wt = seq[raw_pos]
                wt_chars.append(wt)
                # Use position-specific mutation options
                mut = mutation_options[rel_pos][mut_idx]
                mut_chars.append(mut)
            wt_chars = tuple(wt_chars)
            mut_chars = tuple(mut_chars)
        
        # Apply mutations to sequence
        seq_list = list(seq)
        for logical_pos, mut in zip(positions, mut_chars):
            raw_pos = valid_char_positions[logical_pos]
            seq_list[raw_pos] = mut
        result_seq = ''.join(seq_list)
        
        # Build output styles: pass through parent styles (mutagenize preserves length)
        # and add mutation style if _style is set
        output_styles: StyleList = []
        if parent_styles and len(parent_styles) > 0:
            output_styles.extend(parent_styles[0])
        
        if self._style is not None and len(positions) > 0:
            # Convert logical positions to raw positions for styling
            raw_positions = np.array([valid_char_positions[p] for p in positions], dtype=np.int64)
            output_styles.append((self._style, raw_positions))
        
        return {
            'positions': positions,
            'wt_chars': wt_chars,
            'mut_chars': mut_chars,
            'seq': result_seq,
            'style': output_styles,
        }
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'pool': self.parent_pools[0],
            'num_mutations': self.num_mutations,
            'mutation_rate': self.mutation_rate,
            'allowed_chars': self.allowed_chars,
            'region': self._region,
            'style': self._style,
            'prefix': self.name_prefix,
            'mode': self.mode,
            'num_states': self.num_values if self.mode == 'random' and self.num_values is not None and self.num_values > 1 else None,
            'iter_order': self.iter_order,
        }
