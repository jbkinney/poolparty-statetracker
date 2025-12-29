from ..types import Optional, ModeType, AlphabetType, Sequence, beartype, Union
from ..operation import Operation
from ..pool import Pool
import numpy as np

@beartype
class GetKmersOp(Operation):
    """Generate k-mers from an alphabet, supporting multiple lengths."""
    design_card_keys = ['state', 'seq', 'length']
    
    #########################################################
    # Constructor 
    #########################################################
    
    def __init__(
        self,
        length: Union[int, Sequence[int]],
        alphabet: AlphabetType = 'dna',
        mode: ModeType = 'random',
        length_probs: Optional[Sequence[float]] = None,
        name: Optional[str] = None,
        design_card_keys: Optional[Sequence[str]] = None,
    ):
        # Normalize length to a list
        if isinstance(length, int):
            self.lengths = [length]
        else:
            self.lengths = list(length)
        
        # Validate lengths
        if len(self.lengths) == 0:
            raise ValueError("length must not be empty")
        for L in self.lengths:
            if L <= 0:
                raise ValueError(f"All lengths must be > 0, got {L}")
        
        # Set alphabet (sets self.alphabet and self.alpha)
        self._set_alphabet(alphabet)
        
        # Compute states per length and cumulative states for sequential lookup
        self._states_per_length = [self.alpha ** L for L in self.lengths]
        self._cumulative_states = []
        cumsum = 0
        for n in self._states_per_length:
            self._cumulative_states.append(cumsum)
            cumsum += n
        total_states = cumsum
        
        # Validate and normalize length_probs
        if length_probs is not None:
            if len(length_probs) != len(self.lengths):
                raise ValueError(
                    f"length_probs must have same length as lengths: "
                    f"got {len(length_probs)}, expected {len(self.lengths)}"
                )
            probs = np.array(length_probs, dtype=float)
            if np.any(probs < 0):
                raise ValueError("length_probs must be non-negative")
            if probs.sum() <= 0:
                raise ValueError("length_probs must sum to > 0")
            self.length_probs = probs / probs.sum()  # normalize
        else:
            # Uniform distribution
            self.length_probs = np.ones(len(self.lengths)) / len(self.lengths)
        
        # Determine seq_length: fixed if single length, None if variable
        seq_length = self.lengths[0] if len(self.lengths) == 1 else None
        
        super().__init__(
            parent_pools=[],
            num_states=total_states,
            mode=mode,
            seq_length=seq_length,
            name=name,
            design_card_keys=design_card_keys,
        )
    
    #########################################################
    # Results computation
    #########################################################
    
    def _state_to_kmer(self, state: int, length: int) -> str:
        """Convert a local state to a k-mer of given length via base conversion."""
        result = []
        remaining = state
        for _ in range(length):
            result.append(self.alphabet[remaining % self.alpha])
            remaining //= self.alpha
        return ''.join(reversed(result))
    
    def _find_length_for_state(self, global_state: int) -> tuple[int, int]:
        """Find which length bucket a global state belongs to.
        
        Returns:
            (length, local_state) where local_state is the state within that length's bucket
        """
        for i, (cumulative, n_states) in enumerate(zip(self._cumulative_states, self._states_per_length)):
            if global_state < cumulative + n_states:
                return self.lengths[i], global_state - cumulative
        # Should not reach here if global_state < num_states
        raise ValueError(f"State {global_state} out of range")
    
    def compute_results_row(self, input_strings: Sequence[str], sequential_state: int) -> dict:
        """Generate a k-mer and return dict with seq, state, and length."""
        if self.mode == 'random':
            # Sample length according to length_probs
            length_idx = int(self.rng.choice(len(self.lengths), p=self.length_probs))
            length = self.lengths[length_idx]
            # Generate random state for this length
            local_state = int(self.rng.integers(0, self._states_per_length[length_idx]))
            seq = self._state_to_kmer(local_state, length)
            # Compute global state for design card
            state = self._cumulative_states[length_idx] + local_state
        else:
            # Sequential mode: map global state to length and local state
            state = sequential_state % self.num_states
            length, local_state = self._find_length_for_state(state)
            seq = self._state_to_kmer(local_state, length)
        
        return {'seq': seq, 'state': state, 'length': length}

#########################################################
# Public factory function
#########################################################

@beartype
def get_kmers(
    length: Union[int, Sequence[int]],
    alphabet: AlphabetType = 'dna',
    mode: ModeType = 'random',
    length_probs: Optional[Sequence[float]] = None,
    name: str = 'get_kmers',
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Create a Pool that generates k-mers from an alphabet.
    
    Args:
        length: K-mer length(s). Can be a single int or a sequence of ints.
        alphabet: Alphabet to use ('dna', 'rna', 'protein', or custom sequence).
        mode: 'random' or 'sequential'.
        length_probs: Relative probabilities for each length (random mode only).
            If None, uniform distribution is used.
        name: Name for this operation.
        design_card_keys: Which design card keys to include.
    
    Returns:
        Pool: A pool that generates k-mers.
    """
    return Pool(operation=GetKmersOp(
        length=length,
        alphabet=alphabet, 
        mode=mode, 
        length_probs=length_probs,
        name=name,
        design_card_keys=design_card_keys,
    ))
