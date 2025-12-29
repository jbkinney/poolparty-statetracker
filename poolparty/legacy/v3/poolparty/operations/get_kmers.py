"""GetKmers operation - generate k-mers from an alphabet."""

import numpy as np

from ..types import ModeType, AlphabetType, beartype
from ..operation import Operation
from ..pool import Pool
from ..alphabet import get_alphabet
from ..party import get_active_party


class GetKmersOp(Operation):
    """Generate k-mers from an alphabet.
    
    In sequential mode, iterates through all k-mers in lexicographic order.
    In random mode, samples k-mers uniformly.
    """
    
    design_card_keys = ['kmer_index']
    
    @beartype
    def __init__(
        self,
        length: int,
        alphabet: AlphabetType = 'dna',
        mode: ModeType = 'random',
        name: str = 'get_kmers',
    ) -> None:
        """Initialize GetKmersOp.
        
        Args:
            length: Length of k-mers to generate
            alphabet: Alphabet to use ('dna', 'rna', 'protein', or list of chars)
            mode: 'sequential' or 'random'
            name: Operation name
        """
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        
        self.length = length
        self.alphabet = get_alphabet(alphabet)
        self.alpha_size = len(self.alphabet)
        
        # Store total states for random sampling (even if too large to enumerate)
        self._total_states = self.alpha_size ** length
        num_states = Operation.validate_num_states(self._total_states, mode)
        
        super().__init__(
            parent_pools=[],
            num_states=num_states,
            mode=mode,
            name=name,
        )
        
        # Register with active party
        party = get_active_party()
        if party is not None:
            party._register_operation(self)
    
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
    def compute(
        self,
        parent_seqs: list[str],
        state: int,
        rng: np.random.Generator | None,
    ) -> dict:
        """Generate a k-mer."""
        if self.mode == 'random':
            if rng is None:
                raise RuntimeError("Random mode requires RNG")
            idx = int(rng.integers(0, self._total_states))
        else:  # sequential
            idx = state % self._total_states
        
        kmer = self._state_to_kmer(idx)
        
        return {
            'seq_0': kmer,
            'kmer_index': idx,
        }


@beartype
def get_kmers(
    length: int,
    alphabet: AlphabetType = 'dna',
    mode: ModeType = 'random',
    name: str = 'get_kmers',
) -> Pool:
    """Create a Pool that generates k-mers.
    
    Args:
        length: Length of k-mers
        alphabet: Alphabet to use ('dna', 'rna', 'protein', or list of chars)
        mode: 'sequential' or 'random'
        name: Operation name
    
    Returns:
        Pool that generates k-mers
    
    Example:
        >>> barcode = get_kmers(length=10, alphabet='dna')
    """
    op = GetKmersOp(length, alphabet=alphabet, mode=mode, name=name)
    return Pool(operation=op, output_index=0)
