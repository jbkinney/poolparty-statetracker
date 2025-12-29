"""MutationScan operation - apply k mutations to a sequence."""
from itertools import combinations
from math import comb
from numbers import Real
from ..types import Pool_type, Union, AlphabetType, ModeType, Optional, beartype
from ..operation import Operation
from ..pool import Pool
from ..alphabet import get_alphabet
import numpy as np


class MutationScanOp(Operation):
    """Apply k mutations to a parent sequence."""
    factory_name = "mutation_scan"
    design_card_keys = ['positions', 'wt_chars', 'mut_chars']
    
    @beartype
    def __init__(
        self,
        parent_pool: Pool_type,
        k: int = 1,
        alphabet: AlphabetType = 'dna',
        mode: ModeType = 'sequential',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        op_iteration_order: Real = 0,
    ) -> None:
        """Initialize MutationScanOp."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")
        self.k = k
        self.alphabet = get_alphabet(alphabet)
        self.alpha_size = len(self.alphabet)
        self._mode = mode
        self._mutation_map = {}
        for wt in self.alphabet:
            available = [c for c in self.alphabet if c != wt]
            for i, mut in enumerate(available):
                self._mutation_map[(wt, i)] = mut
        self._seq_length = parent_pool.seq_length
        self._sequential_cache = None
        if mode == 'sequential':
            if self._seq_length is not None:
                if self._seq_length < k:
                    raise ValueError(
                        f"k={k} exceeds sequence length={self._seq_length}. "
                        f"Cannot apply {k} mutations to a sequence of length {self._seq_length}."
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
            op_iteration_order=op_iteration_order,
        )
    
    @beartype
    def _build_caches(self) -> int:
        """Build caches for sequential enumeration."""
        if self._seq_length is None:
            return 1
        alpha_minus_1 = self.alpha_size - 1
        num_combinations = comb(self._seq_length, self.k) * (alpha_minus_1 ** self.k)
        num_mut_patterns = alpha_minus_1 ** self.k
        cache = []
        for positions in combinations(range(self._seq_length), self.k):
            for mut_pattern in range(num_mut_patterns):
                mut_indices = []
                remaining = mut_pattern
                for _ in range(self.k):
                    mut_indices.append(remaining % alpha_minus_1)
                    remaining //= alpha_minus_1
                cache.append((positions, tuple(reversed(mut_indices))))
        self._sequential_cache = cache
        return num_combinations
    
    @beartype
    def _random_mutation(self, seq: str, rng: np.random.Generator) -> tuple:
        """Generate random mutation positions and characters."""
        seq_len = len(seq)
        positions = tuple(sorted(rng.choice(seq_len, size=self.k, replace=False)))
        wt_chars = []
        mut_chars = []
        for pos in positions:
            wt = seq[pos]
            wt_chars.append(wt)
            available = [c for c in self.alphabet if c != wt]
            mut = available[rng.integers(0, len(available))]
            mut_chars.append(mut)
        return positions, tuple(wt_chars), tuple(mut_chars)
    
    @beartype
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with mutation positions and characters."""
        seq = parent_seqs[0]
        seq_len = len(seq)
        if self.k > seq_len:
            raise ValueError(f"Cannot apply {self.k} mutations to sequence of length {seq_len}")
        if self.mode in ('random', 'hybrid'):
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)")
            positions, wt_chars, mut_chars = self._random_mutation(seq, rng)
        else:
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
    
    @beartype
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
            'k': self.k,
            'alphabet': self.alphabet,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'op_iteration_order': self.iteration_order,
        }


@beartype
def mutation_scan(
    parent: Union[Pool_type, str],
    k: int = 1,
    alphabet: AlphabetType = 'dna',
    mode: ModeType = 'sequential',
    num_hybrid_states: Optional[int] = None,
    pool_iteration_order: Real = 0,
    op_iteration_order: Real = 0,
    op_name: Optional[str] = None,
    pool_name: Optional[str] = None,
) -> Pool_type:
    """Create a Pool that applies k mutations to a sequence."""
    from .from_seqs import from_seqs
    if isinstance(parent, str):
        parent = from_seqs([parent], mode='fixed')
    op = MutationScanOp(parent, k=k, alphabet=alphabet, mode=mode, 
                        num_hybrid_states=num_hybrid_states, name=op_name,
                        op_iteration_order=op_iteration_order)
    pool = Pool(operation=op, output_index=0)
    pool.iteration_order = pool_iteration_order
    if pool_name is not None:
        pool.name = pool_name
    return pool
