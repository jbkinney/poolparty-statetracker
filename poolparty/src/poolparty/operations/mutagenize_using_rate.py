"""MutationScan operation - apply k mutations to a sequence."""
from itertools import combinations
from math import comb
from ..types import Union, AlphabetType, ModeType, Optional, Real, Integral, beartype
from ..operation import Operation
from ..pool import Pool
from ..alphabet import get_alphabet
import numpy as np


@beartype
class MutagenizeUsingRateOp(Operation):
    """Randomly mutate a sequence based on a mutation rate."""
    factory_name = "mutagenize_using_rate"
    design_card_keys = ['positions', 'wt_chars', 'mut_chars']
    
    def __init__(
        self,
        parent_pool: Pool,
        mutation_rate: Real = 0.1,
        alphabet: AlphabetType = 'dna',
        mode: ModeType = 'random',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Real = 0,
    ) -> None:
        """Initialize MutagenizeUsingRateOp."""
        if mutation_rate < 0 or mutation_rate > 1:
            raise ValueError(f"mutation_rate must be between 0 and 1, got {mutation_rate}")
        
        match mode:
            case 'random':
                num_states = 1
            case 'hybrid':
                if num_hybrid_states is None:
                    raise ValueError("num_hybrid_states is required when mode='hybrid'")
                num_states = num_hybrid_states
            case _:
                raise ValueError(f"mode must be 'random' or 'hybrid', got {mode}")
        
        self.mutation_rate = mutation_rate
        self.alphabet = get_alphabet(alphabet)
        self.alpha_size = len(self.alphabet)
        self._mode = mode
        self._mutation_map = {}
        for wt in self.alphabet:
            available = [c for c in self.alphabet if c != wt]
            for i, mut in enumerate(available):
                self._mutation_map[(wt, i)] = mut
        self._seq_length = parent_pool.seq_length
        
        super().__init__(
            parent_pools=[parent_pool],
            num_states=num_states,
            mode=mode,
            seq_length=self._seq_length,
            name=name,
            iter_order=iter_order,
        )
    
    def _random_mutation(self, seq: str, rng: np.random.Generator) -> tuple:        
        # Choose number of mutations as binomial draw
        seq_len = len(seq)
        num_mut = rng.binomial(seq_len, self.mutation_rate)
        if num_mut == 0:
            # If no mutations, return empty
            return tuple(), tuple(), tuple()
        # Randomly choose unique positions
        positions = tuple(sorted(rng.choice(seq_len, size=num_mut, replace=False)))
        wt_chars = [seq[pos] for pos in positions]
        mut_indices = [rng.integers(0, self.alpha_size-1) for _ in range(num_mut)]
        mut_chars = [self._mutation_map[(wt, mut_idx)] for wt, mut_idx in zip(wt_chars, mut_indices)]
        return positions, tuple(wt_chars), tuple(mut_chars)    
    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: np.random.Generator,
    ) -> dict:
        """Return design card with mutation positions and characters."""
        seq = parent_seqs[0]
        positions, wt_chars, mut_chars = self._random_mutation(seq, rng)
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
            'mutation_rate': self.mutation_rate,
            'alphabet': self.alphabet,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }


@beartype
def mutagenize_using_rate(
    pool: Union[Pool, str],
    mutation_rate: Real = 0.1,
    alphabet: AlphabetType = 'dna',
    mode: ModeType = 'random',
    num_hybrid_states: Optional[int] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Real = 0,
    op_iter_order: Real = 0,
) -> Pool:
    """Create a Pool that applies a random number of mutations to a sequence based on a mutation rate."""
    from .from_seq import from_seq
    pool = from_seq(pool) if isinstance(pool, str) else pool
    op = MutagenizeUsingRateOp(parent_pool=pool, mutation_rate=mutation_rate, alphabet=alphabet, mode=mode, 
                        num_hybrid_states=num_hybrid_states, name=op_name,
                        iter_order=op_iter_order)
    pool = Pool(operation=op, name=name, iter_order=iter_order)
    return pool
