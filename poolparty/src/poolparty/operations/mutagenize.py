"""Mutagenize operation - apply mutations to a sequence."""
from itertools import combinations
from math import comb
from ..types import Union, ModeType, Optional, Real, Integral, beartype
from ..operation import Operation
from ..pool import Pool
from ..party import get_active_party
import numpy as np


@beartype
def mutagenize(
    pool: Union[Pool, str],
    num_mutations: Optional[Integral] = None,
    mutation_rate: Optional[Real] = None,
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
    
    from .from_seq import from_seq
    pool = from_seq(pool) if isinstance(pool, str) else pool
    op = MutagenizeOp(
        parent_pool=pool,
        num_mutations=num_mutations,
        mutation_rate=mutation_rate,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=op_name,
        iter_order=op_iter_order,
    )
    pool = Pool(operation=op, name=name, iter_order=iter_order)
    return pool


@beartype
class MutagenizeOp(Operation):
    """Apply mutations to a parent sequence.
    
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
        
        self.num_mutations = num_mutations
        self.mutation_rate = mutation_rate
        self.alphabet = party.alphabet
        self.alpha_size = self.alphabet.size
        self._mode = mode
        
        # Build mutation map: (wt_char, index) -> mut_char
        # Uses alphabet.mutation_map which maps char -> list of mutation targets
        self._mutation_map = {}
        for wt in self.alphabet.chars:
            for i, mut in enumerate(self.alphabet.mutation_map[wt]):
                self._mutation_map[(wt, i)] = mut
        
        self._seq_length = parent_pool.seq_length
        self._sequential_cache = None
        
        # Determine num_states based on mode
        if mode == 'sequential':
            # Sequential mode only available with num_mutations
            if self._seq_length is not None:
                if self._seq_length < num_mutations:
                    raise ValueError(
                        f"{num_mutations=} exceeds sequence length={self._seq_length}. "
                        f"Cannot apply {num_mutations} mutations to a sequence of length {self._seq_length}."
                    )
                num_states = self._build_caches()
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
        )
    
    def _build_caches(self) -> int:
        """Build caches for sequential enumeration."""
        if self._seq_length is None:
            return 1
        alpha_minus_1 = self.alpha_size - 1
        num_combinations = comb(self._seq_length, self.num_mutations) * (alpha_minus_1 ** self.num_mutations)
        num_mut_patterns = alpha_minus_1 ** self.num_mutations
        cache = []
        for positions in combinations(range(self._seq_length), self.num_mutations):
            for mut_pattern in range(num_mut_patterns):
                mut_indices = []
                remaining = mut_pattern
                for _ in range(self.num_mutations):
                    mut_indices.append(remaining % alpha_minus_1)
                    remaining //= alpha_minus_1
                cache.append((positions, tuple(reversed(mut_indices))))
        self._sequential_cache = cache
        return num_combinations
    
    def _random_mutation(self, seq: str, rng: np.random.Generator) -> tuple:
        """Generate random mutation positions and characters."""
        seq_len = len(seq)
        
        # Determine number of mutations
        if self.num_mutations is not None:
            num_mut = self.num_mutations
        else:
            # Use binomial distribution based on mutation_rate
            num_mut = rng.binomial(seq_len, self.mutation_rate)
            if num_mut == 0:
                return tuple(), tuple(), tuple()
        
        # Choose random positions
        positions = tuple(sorted(rng.choice(seq_len, size=num_mut, replace=False)))
        
        # Determine wild-type and mutant characters
        wt_chars = []
        mut_chars = []
        for pos in positions:
            wt = seq[pos]
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
        """Return design card with mutation positions and characters."""
        seq = parent_seqs[0]
        seq_len = len(seq)
        
        if self.num_mutations is not None and self.num_mutations > seq_len:
            raise ValueError(f"Cannot apply {self.num_mutations} mutations to sequence of length {seq_len}")
        
        if self.mode in ('random', 'hybrid'):
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)")
            positions, wt_chars, mut_chars = self._random_mutation(seq, rng)
        else:
            # Sequential mode (only available with num_mutations)
            if self._sequential_cache is None:
                self._seq_length = seq_len
                self._build_caches()
                self.counter._num_states = len(self._sequential_cache)
            # Use state 0 when inactive (state is None)
            state = self.counter.state
            state = 0 if state is None else state
            positions, mut_indices = self._sequential_cache[state % len(self._sequential_cache)]
            wt_chars = []
            mut_chars = []
            for pos, mut_idx in zip(positions, mut_indices):
                wt = seq[pos]
                wt_chars.append(wt)
                mut = self._mutation_map[(wt, mut_idx)]
                mut_chars.append(mut)
            wt_chars = tuple(wt_chars)
            mut_chars = tuple(mut_chars)
        
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
        seq = parent_seqs[0]
        positions = card['positions']
        mut_chars = card['mut_chars']
        seq_list = list(seq)
        for pos, mut in zip(positions, mut_chars):
            seq_list[pos] = mut
        return {'seq_0': ''.join(seq_list)}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'num_mutations': self.num_mutations,
            'mutation_rate': self.mutation_rate,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }
