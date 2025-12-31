"""GetKmers operation - generate k-mers from an alphabet."""
from numbers import Real
from ..types import Pool_type, ModeType, Optional, beartype
from ..operation import Operation
from ..pool import Pool
from ..party import get_active_party
import numpy as np


@beartype
def get_kmers(
    length: int,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[int] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool_type:
    """Create a Pool that generates k-mers from an alphabet.
    
    Must be called within a Party context. The alphabet is set via the Party
    constructor or Party.set_alphabet() method.

    Parameters
    ----------
    length : int
        Length of k-mers to generate.
    mode : ModeType, default='random'
        Sequence selection mode: 'sequential', 'random', or 'hybrid'.
    num_hybrid_states : Optional[int], default=None
        Number of pool states if mode is 'hybrid'. Ignored for other modes.
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Real, default=0
        Iteration order priority for the resulting Pool.
    op_iter_order : Real, default=0
        Iteration order priority for the internal Operation (typically unused).

    Returns
    -------
    Pool_type
        A Pool whose states yield k-mers of the specified length and alphabet.
    
    Raises
    ------
    RuntimeError
        If called outside of a Party context.
    """
    op = GetKmersOp(length, mode=mode, num_hybrid_states=num_hybrid_states,
                    name=op_name, iter_order=op_iter_order)
    pool = Pool(operation=op, name=name, iter_order=iter_order)
    return pool


@beartype
class GetKmersOp(Operation):
    """Generate k-mers from an alphabet."""
    factory_name = "get_kmers"
    design_card_keys = ['kmer_index']
    
    def __init__(
        self,
        length: int,
        mode: ModeType = 'random',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize GetKmersOp.
        
        Raises:
            RuntimeError: If called outside of a Party context.
        """
        # Get alphabet from active Party context
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "get_kmers requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )
        
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")
        self.length = length
        self.alphabet = party.alphabet
        self.alpha_size = self.alphabet.size
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
            iter_order=iter_order,
        )
    
    def _state_to_kmer(self, state: int) -> str:
        """Convert a state index to a k-mer string."""
        result = []
        remaining = state
        for _ in range(self.length):
            result.append(self.alphabet.chars[remaining % self.alpha_size])
            remaining //= self.alpha_size
        return ''.join(reversed(result))
    
    def _random_kmer(self, rng: np.random.Generator) -> str:
        """Generate a random k-mer."""
        indices = rng.integers(0, self.alpha_size, size=self.length)
        return ''.join(self.alphabet.chars[i] for i in indices)
    
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
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }
