"""Mutagenize operation - apply mutations to a sequence."""
from itertools import combinations
from math import comb
from ..types import Union, ModeType, Optional, Real, Integral, Sequence, RegionType, beartype
from ..operation import Operation
from ..pool import Pool
from ..party import get_active_party
import numpy as np


@beartype
def mutagenize(
    pool: Union[Pool, str],
    num_mutations: Optional[Integral] = None,
    mutation_rate: Optional[Real] = None,
    region: RegionType = None,
    mark_changes: Optional[bool] = None,
    swapcase: bool = False,
    seq_name_prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[int] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Create a Pool that applies mutations to a sequence.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent pool or sequence string to mutate.
    num_mutations : Optional[Integral], default=None
        Fixed number of mutations to apply (mutually exclusive with mutation_rate).
    mutation_rate : Optional[Real], default=None
        Probability of mutation at each position (mutually exclusive with num_mutations).
    region : Union[str, Sequence[Integral], None], default=None
        Region to mutagenize. Can be a marker name (str), explicit interval [start, stop],
        or None to mutagenize entire sequence. Positions are region-relative.
    mark_changes : Optional[bool], default=None
        If True, apply swapcase() to mutated positions. If None, uses party default.
    swapcase : bool, default=False
        If True, swap case of entire sequence after mutations are applied.
        Preserves XML marker tags unchanged.
    mode : ModeType, default='random'
        Selection mode: 'random', 'sequential', or 'hybrid'. Sequential only available with num_mutations.
    num_hybrid_states : Optional[int], default=None
        Required when mode='hybrid'.
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Optional[Real], default=None
        Iteration order for the Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order for the Operation.

    Returns
    -------
    Pool
        A Pool that generates mutated sequences.
    """
    
    from ..fixed_ops.from_seq import from_seq
    pool = from_seq(pool) if isinstance(pool, str) else pool
    op = MutagenizeOp(
        parent_pool=pool,
        num_mutations=num_mutations,
        mutation_rate=mutation_rate,
        region=region,
        mark_changes=mark_changes,
        swapcase=swapcase,
        seq_name_prefix=seq_name_prefix,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=op_name,
        iter_order=op_iter_order,
    )
    pool = Pool(operation=op, name=name, iter_order=iter_order)
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
        parent_pool: Pool,
        num_mutations: Optional[Integral] = None,
        mutation_rate: Optional[Real] = None,
        region: RegionType = None,
        mark_changes: Optional[bool] = None,
        swapcase: bool = False,
        seq_name_prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
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
        
        # Validate hybrid mode
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")
        
        # Store and validate region parameter using centralized validation
        self._region = region
        Operation._validate_region(region)
        
        self.num_mutations = num_mutations
        self.mutation_rate = mutation_rate
        # Resolve mark_changes from party defaults if not explicitly set
        if mark_changes is None:
            mark_changes = party.get_default('mark_changes', False)
        self.mark_changes = mark_changes
        self.swapcase = swapcase
        self.alphabet = party.alphabet
        self.alpha_size = self.alphabet.size
        self._mode = mode
        
        # Build mutation map: (wt_char, index) -> mut_char
        # Uses alphabet.mutation_map which maps char -> list of mutation targets
        # Include all_chars to support both uppercase and lowercase sequences
        self._mutation_map = {}
        for wt in self.alphabet.all_chars:
            for i, mut in enumerate(self.alphabet.mutation_map[wt]):
                self._mutation_map[(wt, i)] = mut
        
        self._seq_length = parent_pool.seq_length
        self._sequential_cache = None
        self._num_mutable_positions = None  # Actual mutable positions, set on first use
        
        # Determine num_states based on mode
        if mode == 'sequential':
            # Sequential mode only available with num_mutations
            # If seq_length is known, eagerly build cache assuming all chars are mutable
            # This will be verified/rebuilt at runtime if actual mutable positions differ
            if self._seq_length is not None:
                if self._seq_length < num_mutations:
                    raise ValueError(
                        f"{num_mutations=} exceeds sequence length={self._seq_length}. "
                        f"Cannot apply {num_mutations} mutations to a sequence of length {self._seq_length}."
                    )
                num_states = self._build_caches(self._seq_length)
            else:
                num_states = 1
        elif mode == 'hybrid':
            num_states = num_hybrid_states
        else:
            num_states = 1
        
        super().__init__(
            parent_pools=[parent_pool],
            num_states=num_states,
            mode=mode,
            seq_length=self._seq_length,
            name=name,
            iter_order=iter_order,
            seq_name_prefix=seq_name_prefix,
        )
    
    def _build_caches(self, num_positions: int) -> int:
        """Build caches for sequential enumeration.
        
        Parameters
        ----------
        num_positions : int
            Number of mutable positions (valid alphabet characters only).
        """
        if num_positions < self.num_mutations:
            raise ValueError(
                f"num_mutations={self.num_mutations} exceeds mutable positions={num_positions}. "
                f"Cannot apply {self.num_mutations} mutations."
            )
        alpha_minus_1 = self.alpha_size - 1
        num_combinations = comb(num_positions, self.num_mutations) * (alpha_minus_1 ** self.num_mutations)
        num_mut_patterns = alpha_minus_1 ** self.num_mutations
        cache = []
        for positions in combinations(range(num_positions), self.num_mutations):
            for mut_pattern in range(num_mut_patterns):
                mut_indices = []
                remaining = mut_pattern
                for _ in range(self.num_mutations):
                    mut_indices.append(remaining % alpha_minus_1)
                    remaining //= alpha_minus_1
                cache.append((positions, tuple(reversed(mut_indices))))
        self._sequential_cache = cache
        self._num_mutable_positions = num_positions
        return num_combinations
    
    def _random_mutation(self, seq: str, rng: np.random.Generator) -> tuple:
        """Generate random mutation positions (logical) and characters."""
        seq_len = self._get_effective_seq_length(seq)
        valid_char_positions = self._get_molecular_positions(seq)
        
        # Determine number of mutations
        if self.num_mutations is not None:
            num_mut = self.num_mutations
        else:
            # Use binomial distribution based on mutation_rate
            num_mut = rng.binomial(seq_len, self.mutation_rate)
            if num_mut == 0:
                return tuple(), tuple(), tuple()
        
        # Choose random logical positions (indices into valid_char_positions)
        positions = tuple(sorted(rng.choice(seq_len, size=num_mut, replace=False)))
        
        # Determine wild-type and mutant characters using raw positions
        wt_chars = []
        mut_chars = []
        for logical_pos in positions:
            raw_pos = valid_char_positions[logical_pos]
            wt = seq[raw_pos]
            wt_chars.append(wt)
            mut_idx = rng.integers(0, self.alpha_size - 1)
            mut = self._mutation_map[(wt, mut_idx)]
            mut_chars.append(mut)
        
        return positions, tuple(wt_chars), tuple(mut_chars)
    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with mutation positions (logical, region-relative) and characters."""
        seq = parent_seqs[0]
        
        # Extract region if specified - mutations apply only to region content
        _, region_seq, _ = self._extract_region_parts(seq, self._region)
        
        seq_len = self._get_effective_seq_length(region_seq)
        valid_char_positions = self._get_molecular_positions(region_seq)
        
        if self.num_mutations is not None and self.num_mutations > seq_len:
            raise ValueError(f"Cannot apply {self.num_mutations} mutations to sequence of length {seq_len}")
        
        if self.mode in ('random', 'hybrid'):
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)")
            positions, wt_chars, mut_chars = self._random_mutation(region_seq, rng)
        else:
            # Sequential mode (only available with num_mutations)
            # Build or rebuild cache based on actual mutable positions
            num_mutable = len(valid_char_positions)
            if self._sequential_cache is None:
                # First use - build cache
                self._build_caches(num_mutable)
                self.counter._num_states = len(self._sequential_cache)
            elif self._num_mutable_positions != num_mutable:
                # Actual mutable positions differ from cached - rebuild
                self._build_caches(num_mutable)
                self.counter._num_states = len(self._sequential_cache)
            # Use state 0 when inactive (state is None)
            state = self.counter.state
            state = 0 if state is None else state
            positions, mut_indices = self._sequential_cache[state % len(self._sequential_cache)]
            wt_chars = []
            mut_chars = []
            for logical_pos, mut_idx in zip(positions, mut_indices):
                raw_pos = valid_char_positions[logical_pos]
                wt = region_seq[raw_pos]
                wt_chars.append(wt)
                mut = self._mutation_map[(wt, mut_idx)]
                mut_chars.append(mut)
            wt_chars = tuple(wt_chars)
            mut_chars = tuple(mut_chars)
        
        # Apply case swap to mut_chars if mark_changes is True
        # This ensures design card reflects what appears in the sequence
        if self.mark_changes:
            mut_chars = tuple(c.swapcase() for c in mut_chars)
        
        return {
            'positions': positions,
            'wt_chars': wt_chars,
            'mut_chars': mut_chars,
        }
    
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Apply mutations to the parent sequence based on design card."""
        from ..marker_ops.parsing import transform_nonmarker_chars
        
        seq = parent_seqs[0]
        positions = card['positions']  # Logical positions (region-relative)
        mut_chars = card['mut_chars']
        
        # Extract region parts
        prefix, region_seq, suffix = self._extract_region_parts(seq, self._region)
        valid_char_positions = self._get_molecular_positions(region_seq)
        
        # Apply mutations to region
        region_list = list(region_seq)
        for logical_pos, mut in zip(positions, mut_chars):
            raw_pos = valid_char_positions[logical_pos]
            region_list[raw_pos] = mut
        mutated_region = ''.join(region_list)
        
        # Reassemble: prefix + mutated_region + suffix
        result_seq = prefix + mutated_region + suffix
        
        if self.swapcase:
            result_seq = transform_nonmarker_chars(result_seq, str.swapcase)
        return {'seq_0': result_seq}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'num_mutations': self.num_mutations,
            'mutation_rate': self.mutation_rate,
            'region': self._region,
            'mark_changes': self.mark_changes,
            'swapcase': self.swapcase,
            'seq_name_prefix': self.name_prefix,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }
