"""MutationScan operation - apply k mutations to a sequence."""

from itertools import combinations
from math import comb
import numpy as np

from ..types import Union, ModeType, AlphabetType, beartype
from ..operation import Operation
from ..pool import Pool
from ..alphabet import get_alphabet
from ..party import get_active_party


class MutationScanOp(Operation):
    """Apply k mutations to a parent sequence.
    
    In sequential mode, iterates through all C(L,k) * (|A|-1)^k combinations.
    In random mode, samples random positions and mutations.
    """
    
    design_card_keys = ['positions', 'wt_chars', 'mut_chars']
    
    @beartype
    def __init__(
        self,
        parent_pool: Pool,
        k: int = 1,
        alphabet: AlphabetType = 'dna',
        mode: ModeType = 'sequential',
        name: str = 'mutation_scan',
    ) -> None:
        """Initialize MutationScanOp.
        
        Args:
            parent_pool: Input sequence pool
            k: Number of mutations to apply
            alphabet: Alphabet for mutations
            mode: 'sequential' or 'random'
            name: Operation name
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        
        self.k = k
        self.alphabet = get_alphabet(alphabet)
        self.alpha_size = len(self.alphabet)
        
        # Build mutation map: (wt_char, mut_idx) -> mut_char
        self._mutation_map: dict[tuple[str, int], str] = {}
        for wt in self.alphabet:
            available = [c for c in self.alphabet if c != wt]
            for i, mut in enumerate(available):
                self._mutation_map[(wt, i)] = mut
        
        # We need parent's sequence length to compute num_states
        # For now, assume parent operation has a way to tell us
        # This is computed from the first sequence if available
        self._seq_length: int | None = None
        self._sequential_cache: list | None = None
        
        # Try to determine sequence length from parent
        # This works if parent is FromSeqsOp or similar
        if hasattr(parent_pool.operation, 'seqs'):
            self._seq_length = len(parent_pool.operation.seqs[0])
            num_states = self._build_caches(mode)
        else:
            # Unknown length - use placeholder
            # Will validate at compute time
            num_states = 1
        
        super().__init__(
            parent_pools=[parent_pool],
            num_states=num_states,
            mode=mode,
            name=name,
        )
        
        # Register with active party
        party = get_active_party()
        if party is not None:
            party._register_operation(self)
    
    @beartype
    def _build_caches(self, mode: ModeType) -> int:
        """Build caches for sequential enumeration.
        
        Returns:
            Validated num_states (may be -1 if too large for random mode)
        """
        if self._seq_length is None:
            return 1
        
        # Calculate number of combinations
        alpha_minus_1 = self.alpha_size - 1
        num_combinations = comb(self._seq_length, self.k) * (alpha_minus_1 ** self.k)
        
        # Validate against max_num_sequential_states
        num_states = Operation.validate_num_states(num_combinations, mode)
        
        # Only build cache if within limit (needed for sequential mode)
        if num_states == -1:
            self._sequential_cache = []
            return -1
        
        # Build list of (positions, mutation_indices) tuples
        num_mut_patterns = alpha_minus_1 ** self.k
        
        cache = []
        for positions in combinations(range(self._seq_length), self.k):
            for mut_pattern in range(num_mut_patterns):
                # Decode mutation pattern to indices
                mut_indices = []
                remaining = mut_pattern
                for _ in range(self.k):
                    mut_indices.append(remaining % alpha_minus_1)
                    remaining //= alpha_minus_1
                cache.append((positions, tuple(reversed(mut_indices))))
        
        self._sequential_cache = cache
        return num_states
    
    @beartype
    def compute(
        self,
        parent_seqs: list[str],
        state: int,
        rng: np.random.Generator | None,
    ) -> dict:
        """Apply mutations to the parent sequence."""
        seq = parent_seqs[0]
        seq_len = len(seq)
        
        if self.k > seq_len:
            raise ValueError(f"Cannot apply {self.k} mutations to sequence of length {seq_len}")
        
        # Get positions and mutation indices
        if self.mode == 'random':
            if rng is None:
                raise RuntimeError("Random mode requires RNG")
            positions = tuple(sorted(rng.choice(seq_len, size=self.k, replace=False)))
            mut_indices = tuple(int(rng.integers(0, self.alpha_size - 1)) for _ in range(self.k))
        else:  # sequential
            if self._sequential_cache is None:
                # Build cache on first access
                self._seq_length = seq_len
                self.num_states = self._build_caches(self.mode)
            positions, mut_indices = self._sequential_cache[state % len(self._sequential_cache)]
        
        # Apply mutations
        seq_list = list(seq)
        wt_chars = []
        mut_chars = []
        
        for pos, mut_idx in zip(positions, mut_indices):
            wt = seq_list[pos]
            wt_chars.append(wt)
            mut = self._mutation_map[(wt, mut_idx)]
            mut_chars.append(mut)
            seq_list[pos] = mut
        
        return {
            'seq_0': ''.join(seq_list),
            'positions': positions,
            'wt_chars': tuple(wt_chars),
            'mut_chars': tuple(mut_chars),
        }


@beartype
def mutation_scan(
    parent: Union[Pool, str],
    k: int = 1,
    alphabet: AlphabetType = 'dna',
    mode: ModeType = 'sequential',
    name: str = 'mutation_scan',
) -> Pool:
    """Create a Pool that applies k mutations to a sequence.
    
    Args:
        parent: Input sequence or pool
        k: Number of mutations
        alphabet: Alphabet for mutations
        mode: 'sequential' or 'random'
        name: Operation name
    
    Returns:
        Pool with mutated sequences
    
    Example:
        >>> mutants = mutation_scan('ACGTACGT', k=1)
        >>> left, right = breakpoint_scan(seq, num_breakpoints=1)
        >>> mutated_right = mutation_scan(right, k=1)
    """
    from .from_seqs import from_seqs
    
    if isinstance(parent, str):
        parent = from_seqs([parent], mode='sequential', name=f'{name}_input')
    
    op = MutationScanOp(parent, k=k, alphabet=alphabet, mode=mode, name=name)
    return Pool(operation=op, output_index=0)
