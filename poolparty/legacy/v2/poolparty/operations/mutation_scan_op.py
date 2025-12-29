from itertools import combinations
from math import comb

from ..types import Union, Optional, ModeType, AlphabetType, Sequence, beartype
from ..operation import Operation
from ..pool import Pool
from ..alphabet import validate_alphabet
from .from_seqs_op import from_seqs

@beartype
class MutationScanOp(Operation):
    """Apply a specific number of mutations to a parent sequence."""
    design_card_keys = ['positions', 'wt_chars', 'mut_chars', 'seq']
    
    #########################################################
    # Constructor 
    #########################################################
    
    def __init__(
        self,
        parent: Union[Pool, str],
        num_mutations: int,
        alphabet: AlphabetType = 'dna', 
        mode: ModeType = 'random',
        name: Optional[str] = None,
        design_card_keys: Optional[Sequence[str]] = None,
    ):
        if num_mutations <= 0:
            raise ValueError(f"num_mutations must be > 0, got {num_mutations}")
        
        self.num_mutations = num_mutations
        self._set_alphabet(alphabet) # sets self.alphabet and self.alpha
                
        # Cast parent to pool if needed
        if isinstance(parent, str):
            parent = from_seqs([parent], design_card_keys=[])
        
        # Calculate number of states
        seq_length = len(parent)
        if seq_length is not None:
            num_position_choices = comb(seq_length, num_mutations)
            num_mutation_patterns = (self.alpha - 1) ** num_mutations
            self.num_states = num_position_choices * num_mutation_patterns
        else:
            self.num_states = -1
            
        # Build mutation map
        self._build_mutation_map()

        # Build sequential cache
        if self.num_states <= Operation.max_sequential_states:
            self._build_sequential_cache(seq_length)

        super().__init__(parent_pools=[parent], 
                         num_states=self.num_states, 
                         mode=mode, 
                         seq_length=seq_length, 
                         name=name,
                         design_card_keys=design_card_keys)
    
    #########################################################
    # Results computation
    #########################################################
    
    def compute_results_row(self, input_strings: Sequence[str], sequential_state: int) -> dict:
        """Compute mutated sequence with exactly k mutations and return dict."""
        base_seq = input_strings[0]
        seq_len = len(base_seq)
        
        if not set(base_seq).issubset(set(self.alphabet)):
            raise ValueError(f"Sequence '{base_seq}' contains characters not in alphabet '{''.join(self.alphabet)}'")
        
        if seq_len < self.num_mutations:
            raise ValueError(f"Sequence length ({seq_len}) must be >= num_mutations ({self.num_mutations})")
        
        # Determine mutations
        if self.mode == 'random':
            positions = sorted([int(p) for p in self.rng.choice(seq_len, size=self.num_mutations, replace=False)])
            mut_ints = [int(self.rng.integers(0, self.alpha - 1)) for _ in range(self.num_mutations)]
        elif self.mode == 'sequential':
            positions, mut_ints = self._sequential_cache[sequential_state % len(self._sequential_cache)]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # Apply mutations
        wt_chars = [base_seq[p] for p in positions]
        mut_chars = [self._mutation_map[(wt_char, mut_int)] for wt_char, mut_int in zip(wt_chars, mut_ints)]
        seq_list = list(base_seq)
        for i, pos in enumerate(positions):
            seq_list[pos] = mut_chars[i]
        seq = ''.join(seq_list)
        
        return {
            'seq': seq,
            'positions': tuple(positions),
            'wt_chars': tuple(wt_chars),
            'mut_chars': tuple(mut_chars),
        }
    
    #########################################################
    # Helper methods
    #########################################################
    
    def _build_mutation_map(self) -> None:
        """Build lookup mapping (wt_char, mutation_int) -> mut_char."""
        mutation_map: dict[tuple[str, int], str] = {}
        for wt_char in self.alphabet:
            available = [c for c in self.alphabet if c != wt_char]
            for i, mut_char in enumerate(available):
                mutation_map[(wt_char, i)] = mut_char
        self._mutation_map = mutation_map
    
    def _build_sequential_cache(self, seq_len: int) -> None:
        """Build cache mapping state index to (positions, mutations) tuples.
        
        Iteration order: outer loop over position combinations, inner loop over 
        mutation patterns (so mutations iterate fastest).
        """
        cache: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        num_mutation_patterns = (self.alpha - 1) ** self.num_mutations
        pos_combinations = combinations(range(seq_len), self.num_mutations)
        for pos_tuple in pos_combinations:
            for mut_tuple_idx in range(num_mutation_patterns):
                mut_int_tuple = tuple(((mut_tuple_idx // (self.alpha - 1) ** i) % (self.alpha - 1)) for i in reversed(range(self.num_mutations)))
                cache.append((pos_tuple, mut_int_tuple))
        self._sequential_cache = cache

#########################################################
# Public factory function
#########################################################

@beartype
def mutation_scan(
    seq: Union[Pool, str],
    num_mutations: int = 1,
    alphabet: Union[str, Sequence[str]] = 'dna',
    mode: ModeType = 'random',
    name: str = 'mutation_scan',
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Create a Pool that introduces a specific number of mutations to a sequence.""" 
    parent = from_seqs([seq], design_card_keys=[]) if isinstance(seq, str) else seq
    
    if (parent.seq_length is not None) and (num_mutations > parent.seq_length):
        raise ValueError(f"num_mutations ({num_mutations}) must be <= sequence length ({parent.seq_length})")
    
    return Pool(operation=MutationScanOp(parent, 
                                         num_mutations=num_mutations, 
                                         alphabet=alphabet, 
                                         mode=mode, 
                                         name=name,
                                         design_card_keys=design_card_keys))
