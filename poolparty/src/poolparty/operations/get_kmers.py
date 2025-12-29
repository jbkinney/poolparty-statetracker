"""GetKmers operation - generate k-mers from an alphabet."""
from numbers import Real
from ..types import Pool_type, AlphabetType, ModeType, Optional, beartype
from ..operation import Operation
from ..pool import Pool
from ..alphabet import get_alphabet
import numpy as np


class GetKmersOp(Operation):
    """Generate k-mers from an alphabet."""
    factory_name = "get_kmers"
    design_card_keys = ['kmer_index']
    
    @beartype
    def __init__(
        self,
        length: int,
        alphabet: AlphabetType = 'dna',
        mode: ModeType = 'sequential',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        op_iteration_order: Real = 0,
    ) -> None:
        """Initialize GetKmersOp."""
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")
        self.length = length
        self.alphabet = get_alphabet(alphabet)
        self.alpha_size = len(self.alphabet)
        total_kmers = self.alpha_size ** length
        if mode == 'sequential':
            num_states = self.validate_num_states(total_kmers, mode)
        elif mode == 'hybrid':
            num_states = num_hybrid_states
        else:
            num_states = 1
        super().__init__(
            parent_pools=[],
            num_states=num_states,
            mode=mode,
            seq_length=length,
            name=name,
            iter_order=op_iteration_order,
        )
    
    @beartype
    def _state_to_kmer(self, state: int) -> str:
        """Convert a state index to a k-mer string."""
        result = []
        remaining = state
        for _ in range(self.length):
            result.append(self.alphabet[remaining % self.alpha_size])
            remaining //= self.alpha_size
        return ''.join(reversed(result))
    
    @beartype
    def _random_kmer(self, rng: np.random.Generator) -> str:
        """Generate a random k-mer."""
        indices = rng.integers(0, self.alpha_size, size=self.length)
        return ''.join(self.alphabet[i] for i in indices)
    
    @beartype
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with kmer selection."""
        if self.mode in ('random', 'hybrid'):
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)")
            kmer = self._random_kmer(rng)
            return {'kmer_index': None, 'kmer': kmer}
        else:
            # Use state 0 when inactive (state is None)
            idx = self.counter.state
            idx = 0 if idx is None else idx
            return {'kmer_index': idx}
    
    @beartype
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Return the kmer based on design card."""
        if 'kmer' in card:
            # Random mode: kmer was pre-computed
            kmer = card['kmer']
        else:
            # Sequential mode: compute from index
            kmer = self._state_to_kmer(card['kmer_index'])
        return {'seq_0': kmer}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'length': self.length,
            'alphabet': self.alphabet,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'op_iteration_order': self.iter_order,
        }


@beartype
def get_kmers(
    length: int,
    alphabet: AlphabetType = 'dna',
    mode: ModeType = 'sequential',
    num_hybrid_states: Optional[int] = None,
    pool_iteration_order: Real = 0,
    op_iteration_order: Real = 0,
    op_name: Optional[str] = None,
    name: Optional[str] = None,
) -> Pool_type:
    """Create a Pool that generates k-mers."""
    op = GetKmersOp(length, alphabet=alphabet, mode=mode, 
                    num_hybrid_states=num_hybrid_states, name=op_name,
                    op_iteration_order=op_iteration_order)
    pool = Pool(operation=op, output_index=0)
    pool.iter_order = pool_iteration_order
    if name is not None:
        pool.name = name
    return pool
