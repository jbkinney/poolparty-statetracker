"""GetKmers operation - generate k-mers from an alphabet."""
from numbers import Real
from ..types import Pool_type, ModeType, Optional, Literal, Union, RegionType, beartype
from ..operation import Operation
from ..pool import Pool
from ..party import get_active_party
import numpy as np


@beartype
def get_kmers(
    length: int,
    bg_pool: Optional[Union[Pool, str]] = None,
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    mark_changes: Optional[bool] = None,
    case: Literal['lower', 'upper'] = 'upper',
    seq_name_prefix: Optional[str] = None,
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
    bg_pool : Optional[Union[Pool, str]], default=None
        Background pool or sequence. If provided with region, generated k-mer
        replaces the region content.
    region : RegionType, default=None
        Region to replace in bg_pool. Can be a marker name or [start, stop] interval.
        Required if bg_pool is provided.
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
    case : Literal['lower', 'upper'], default='upper'
        Case of output k-mers: 'upper' for uppercase, 'lower' for lowercase.
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
    ValueError
        If bg_pool is provided without region.
    """
    from ..fixed_ops.from_seq import from_seq
    bg_pool_obj = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool
    op = GetKmersOp(length, bg_pool=bg_pool_obj, region=region,
                    remove_marker=remove_marker, mark_changes=mark_changes,
                    case=case, seq_name_prefix=seq_name_prefix, mode=mode,
                    num_hybrid_states=num_hybrid_states,
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
        bg_pool: Optional[Pool] = None,
        region: RegionType = None,
        remove_marker: Optional[bool] = None,
        mark_changes: Optional[bool] = None,
        case: Literal['lower', 'upper'] = 'upper',
        seq_name_prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize GetKmersOp."""
        # Get alphabet from active Party context
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "get_kmers requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )
        
        # Validate bg_pool/region combination
        if bg_pool is not None and region is None:
            raise ValueError(
                "region is required when bg_pool is provided. "
                "Specify which region of bg_pool to replace with the generated k-mer."
            )
        
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")
        
        # Resolve mark_changes from party defaults if not explicitly set
        if mark_changes is None:
            mark_changes = party.get_default('mark_changes', False)
        self.mark_changes = mark_changes
        
        self.length = length
        self.case = case
        self.alphabet = party.alphabet
        self.alpha_size = self.alphabet.size
        total_kmers = self.alpha_size ** length
        if mode == 'sequential':
            num_states = self.validate_num_states(total_kmers, mode)
        elif mode == 'hybrid':
            num_states = num_hybrid_states
        else:
            num_states = 1
        
        parent_pools = [bg_pool] if bg_pool is not None else []
        super().__init__(
            parent_pools=parent_pools,
            num_states=num_states,
            mode=mode,
            seq_length=length,
            name=name,
            iter_order=iter_order,
            seq_name_prefix=seq_name_prefix,
            region=region,
            remove_marker=remove_marker,
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
        # Apply case transformation
        if self.case == 'lower':
            kmer = kmer.lower()
        # Apply mark_changes swapcase only when inserting into a region
        if self.mark_changes and self._region is not None:
            kmer = kmer.swapcase()
        return {'seq_0': kmer}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'length': self.length,
            'bg_pool': self.parent_pools[0] if self.parent_pools else None,
            'region': self._region,
            'remove_marker': self._remove_marker,
            'mark_changes': self.mark_changes,
            'case': self.case,
            'seq_name_prefix': self.name_prefix,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }
