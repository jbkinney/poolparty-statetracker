import random
import itertools
from typing import Union, List
from .pool import Pool
from .utils import validate_alphabet


class KmerPool(Pool):
    """A class for generating k-mer sequences.
    
    Supports both random k-mer generation and sequential iteration through
    all possible k-mers. The pool always has a finite number of states equal
    to len(alphabet)^length.
    
    When iterated directly or called with next(), generates random k-mers.
    When used with generate_seqs() and included in combinatorially_complete_pools,
    iterates through all k-mers sequentially.
    """
    def __init__(self, length: int, alphabet: Union[str, List[str]] = 'dna', max_num_states: int = None, mode: str = 'random', iteration_order: int | None = None, name: str | None = None, metadata: str = 'features'):
        """Initialize a KmerPool.
        
        Args:
            length: Length of k-mers to generate
            alphabet: Either a string naming a predefined alphabet (e.g., 'dna', 'rna'),
                or a list of single-character strings to use as the alphabet.
                Default: 'dna'
            max_num_states: Maximum number of states before treating as infinite
            mode: Either 'random' or 'sequential' (default: 'random')
            iteration_order: Order for sequential iteration (default: auto-assigned based on creation order)
            metadata: Metadata level ('core', 'features', 'complete'). Default: 'features'
        """
        self.length = length
        self.alphabet = validate_alphabet(alphabet)
        super().__init__(op='kmer', max_num_states=max_num_states, mode=mode, iteration_order=iteration_order, name=name, metadata=metadata)
    
    def _calculate_num_internal_states(self) -> int:
        """KmerPool always has finite internal states = len(alphabet)^length."""
        return len(self.alphabet) ** self.length
    
    def _calculate_seq_length(self) -> int:
        """KmerPool always produces sequences of length k."""
        return self.length
    
    def _compute_seq(self) -> str:
        """Compute sequence based on current state.
        
        Maps state directly to a specific k-mer using base conversion.
        For random generation, the calling code will set random states.
        """
        # Convert state to k-mer via mixed-radix (base-n) conversion
        state = self.get_state() % self.num_internal_states
        result = []
        for _ in range(self.length):
            result.append(self.alphabet[state % len(self.alphabet)])
            state //= len(self.alphabet)
        return ''.join(reversed(result))
    
    def __repr__(self) -> str:
        return f"KmerPool(L={self.length}, alphabet='{self.alphabet}')"

