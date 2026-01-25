"""FromSeqs operation - create a pool from a list of sequences."""
from numbers import Real
from ..types import Pool_type, Sequence, ModeType, Optional, Union, RegionType, beartype, StyleList
from ..operation import Operation
from ..pool import Pool
from .. import dna
import numpy as np


@beartype
def from_seqs(
    seqs: Sequence[str],
    bg_pool: Optional[Union[Pool, str]] = None,
    region: RegionType = None,
    style: Optional[str] = None,
    seq_names: Optional[Sequence[str]] = None,
    prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_states: Optional[int] = None,
    iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = None,
) -> Pool_type:
    """
    Create a Pool containing the specified sequences.

    Parameters
    ----------
    seqs : Sequence[str]
        Sequence of string sequences to include in the pool.
    bg_pool : Optional[Union[Pool, str]], default=None
        Background pool or sequence. If provided with region, selected sequence
        replaces the region content.
    region : RegionType, default=None
        Region to replace in bg_pool. Can be a marker name or [start, stop] interval.
        Required if bg_pool is provided.
    seq_names : Optional[Sequence[str]], default=None
        Explicit names for each sequence. If provided, these are used directly.
    prefix : Optional[str], default=None
        Prefix for auto-generated names (e.g., 'seq_' produces 'seq_0', 'seq_1', ...).
        Cannot be used together with seq_names.
    mode : ModeType, default='random'
        Sequence selection mode: 'sequential' or 'random'.
    num_states : Optional[int], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool_type
        A Pool object yielding the provided sequences using the specified selection mode.
    
    Raises
    ------
    ValueError
        If bg_pool is provided without region.
    """
    from ..fixed_ops.from_seq import from_seq
    bg_pool_obj = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool
    op = FromSeqsOp(seqs, bg_pool=bg_pool_obj, region=region,
                    style=style,
                    seq_names=seq_names, prefix=prefix,
                    mode=mode, num_states=num_states,
                    name=None, iter_order=iter_order,
                    _factory_name=_factory_name)
    pool = Pool(operation=op)
    return pool


@beartype
class FromSeqsOp(Operation):
    """Create a pool from a list of sequences."""
    factory_name = "from_seqs"
    design_card_keys = ['seq_name', 'seq_index']
    
    def __init__(
        self,
        seqs: Sequence[str],
        bg_pool: Optional[Pool] = None,
        region: RegionType = None,
        spacer_str: str = '',
        style: Optional[str] = None,
        seq_names: Optional[Sequence[str]] = None,
        prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        _factory_name: Optional[str] = None,
    ) -> None:
        """Initialize FromSeqsOp."""
        from ..party import get_active_party
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "from_seqs requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )
        
        # Set factory name if provided
        if _factory_name is not None:
            self.factory_name = _factory_name
 
        # Validate bg_pool/region combination
        if bg_pool is not None and region is None:
            raise ValueError(
                "region is required when bg_pool is provided. "
                "Specify which region of bg_pool to replace with the selected sequence."
            )
        
        self._style = style
        
        if len(seqs) == 0:
            raise ValueError("seqs must not be empty")
        if mode == 'fixed' and len(seqs) != 1:
            raise ValueError("mode='fixed' requires exactly 1 sequence")
        if seq_names is not None and prefix is not None:
            raise ValueError("Cannot specify both seq_names and prefix")
        self.seqs = list(seqs)
        # Track whether explicit seq_names were provided (for compute_seq_names)
        self._seq_names_explicit = seq_names is not None
        self.seq_names = list(seq_names) if seq_names else [f"seq_{i}" for i in range(len(seqs))]
        if len(self.seq_names) != len(self.seqs):
            raise ValueError("seq_names must have same length as seqs")
        match mode:
            case 'sequential':
                num_states = len(seqs)
            case 'random':
                # num_states stays None for pure random mode
                pass
            case _:
                num_states = 1
        # Use lengths without markers (includes all chars except marker tags)
        lengths = [dna.get_length_without_markers(s) for s in self.seqs]
        seq_length = lengths[0] if all(L == lengths[0] for L in lengths) else None
        
        parent_pools = [bg_pool] if bg_pool is not None else []
        super().__init__(
            parent_pools=parent_pools,
            num_values=num_states,
            mode=mode,
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=region,
        )
    
    def compute(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
        parent_styles: list[StyleList] | None = None,
    ) -> dict:
        """Return design card and sequence together."""
        if self.mode == 'random':
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)")
            idx = rng.integers(0, len(self.seqs))
        elif self.state is None:
            # Fixed mode - always use index 0
            idx = 0
        else:
            # Sequential mode - use state value (0 when inactive)
            state = self.state.value
            idx = (0 if state is None else state) % len(self.seqs)
        
        seq = self.seqs[idx]
        
        # Apply style to all positions if specified
        output_styles: StyleList = []
        if self._style is not None:
            positions = np.arange(len(seq), dtype=np.int64)
            output_styles.append((self._style, positions))
        
        return {
            'seq_name': self.seq_names[idx],
            'seq_index': idx,
            'seq': seq,
            'style': output_styles,
        }
    
    def compute_seq_names(
        self,
        parent_names: list[Optional[str]],
        card: dict,
    ) -> Optional[str]:
        """Return name based on explicit seq_names or name_prefix."""
        # Block name if _block_seq_names is set
        if self._block_seq_names:
            return None
        # If explicit seq_names were provided, use them directly
        if self._seq_names_explicit:
            idx = card['seq_index']
            return self.seq_names[idx]
        # Otherwise fall back to prefix logic
        if self.name_prefix is None:
            return None
        state = self.state.value
        if state is None:
            return None
        return f'{self.name_prefix}{state}'
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'seqs': self.seqs,
            'bg_pool': self.parent_pools[0] if self.parent_pools else None,
            'region': self._region,
            'style': self._style,
            'seq_names': self.seq_names if self._seq_names_explicit else None,
            'prefix': self.name_prefix,
            'mode': self.mode,
            'num_states': self.num_values if self.mode == 'random' and self.num_values is not None and self.num_values > 1 else None,
            'name': None,
            'iter_order': self.iter_order,
        }
