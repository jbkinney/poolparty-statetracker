"""FromIupac operation - generate DNA sequences from IUPAC notation."""
from numbers import Real
from ..types import Pool_type, Sequence, ModeType, Optional, Union, RegionType, beartype
from ..operation import Operation
from ..pool import Pool
from .. import dna
import numpy as np


@beartype
def from_iupac(
    iupac_seq: str,
    bg_pool: Optional[Union[Pool, str]] = None,
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    spacer_str: str = '',
    mark_changes: Optional[bool] = None,
    seq_name_prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[int] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool_type:
    """
    Create a Pool that generates DNA sequences from IUPAC notation.

    Parameters
    ----------
    iupac_seq : str
        IUPAC sequence string (e.g., 'RN' for purine + any base).
        Valid characters: A, C, G, T, U, R, Y, S, W, K, M, B, D, H, V, N.
    bg_pool : Optional[Union[Pool, str]], default=None
        Background pool or sequence. If provided with region, generated sequence
        replaces the region content.
    region : RegionType, default=None
        Region to replace in bg_pool. Can be a marker name or [start, stop] interval.
        Required if bg_pool is provided.
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
    mark_changes : Optional[bool], default=None
        If True, apply swapcase() to degenerate positions. If None, uses party default.
    mode : ModeType, default='random'
        Sequence selection mode: 'sequential', 'random', or 'hybrid'.
    num_hybrid_states : Optional[int], default=None
        Number of pool states when using 'hybrid' mode.
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Real, default=0
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operation.

    Returns
    -------
    Pool_type
        A Pool yielding DNA sequences from the IUPAC pattern.
    
    Raises
    ------
    ValueError
        If bg_pool is provided without region.
    """
    from ..fixed_ops.from_seq import from_seq
    bg_pool_obj = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool
    op = FromIupacOp(
        iupac_seq=iupac_seq,
        bg_pool=bg_pool_obj,
        region=region,
        remove_marker=remove_marker,
        spacer_str=spacer_str,
        mark_changes=mark_changes,
        seq_name_prefix=seq_name_prefix,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=op_name,
        iter_order=op_iter_order,
    )
    pool = Pool(operation=op, name=name, iter_order=iter_order)
    return pool


@beartype
class FromIupacOp(Operation):
    """Generate DNA sequences from IUPAC notation."""
    factory_name = "from_iupac"
    design_card_keys = ['iupac_state']

    def __init__(
        self,
        iupac_seq: str,
        bg_pool: Optional[Pool] = None,
        region: RegionType = None,
        remove_marker: Optional[bool] = None,
        spacer_str: str = '',
        mark_changes: Optional[bool] = None,
        seq_name_prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize FromIupacOp."""
        from ..party import get_active_party
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "from_iupac requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )
        
        # Validate bg_pool/region combination
        if bg_pool is not None and region is None:
            raise ValueError(
                "region is required when bg_pool is provided. "
                "Specify which region of bg_pool to replace with the generated sequence."
            )
        
        if not iupac_seq:
            raise ValueError("iupac_seq must be a non-empty string")
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")

        # Validate and build position options
        # Handle ignore chars (e.g., '.', '-', ' ') as pass-through positions
        iupac_seq_upper = iupac_seq.upper()
        invalid_chars = set()
        position_options = []
        for char, char_upper in zip(iupac_seq, iupac_seq_upper):
            if char in dna.IGNORE_CHARS:
                # Pass through ignore chars unchanged
                position_options.append([char])
            elif char_upper in dna.IUPAC_TO_DNA:
                opts = dna.IUPAC_TO_DNA[char]
                position_options.append(opts)
            else:
                invalid_chars.add(char)

        if invalid_chars:
            raise ValueError(
                f"iupac_seq contains invalid IUPAC character(s): {sorted(invalid_chars)}. "
                f"Valid IUPAC characters are: {sorted(dna.IUPAC_TO_DNA.keys())} "
                f"(plus ignore characters: {sorted(dna.IGNORE_CHARS)})"
            )

        self.iupac_seq = iupac_seq
        self.position_options = position_options
        # Resolve mark_changes from party defaults if not explicitly set
        if mark_changes is None:
            mark_changes = party.get_default('mark_changes', False)
        self.mark_changes = mark_changes

        # Compute total states as product of possibilities at each position
        total_states = 1
        for options in position_options:
            total_states *= len(options)

        match mode:
            case 'sequential':
                num_states = total_states
            case 'hybrid':
                num_states = num_hybrid_states
            case _:
                num_states = 1

        self._total_states = total_states
        # Use length without markers for consistency
        seq_length = dna.get_length_without_markers(iupac_seq)
        
        parent_pools = [bg_pool] if bg_pool is not None else []
        super().__init__(
            parent_pools=parent_pools,
            num_states=num_states,
            mode=mode,
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
            seq_name_prefix=seq_name_prefix,
            region=region,
            remove_marker=remove_marker,
            spacer_str=spacer_str,
        )

    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with IUPAC state index."""
        if self.mode in ('random', 'hybrid'):
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")
            state = rng.integers(0, self._total_states)
        else:
            counter_state = self.counter.state
            state = (0 if counter_state is None else counter_state) % self._total_states
        return {'iupac_state': state}

    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Return the DNA sequence for the given state."""
        state = card['iupac_state']
        # Mixed-radix conversion: map state to specific sequence
        result = []
        remaining = state
        for position_opts in reversed(self.position_options):
            result.append(position_opts[remaining % len(position_opts)])
            remaining //= len(position_opts)
        result = list(reversed(result))
        seq = ''.join(result)
        
        # Apply mark_changes swapcase only when inserting into a region
        if self.mark_changes and self._region is not None:
            seq = seq.swapcase()
        
        return {'seq_0': seq}

    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'iupac_seq': self.iupac_seq,
            'bg_pool': self.parent_pools[0] if self.parent_pools else None,
            'region': self._region,
            'remove_marker': self._remove_marker,
            'spacer_str': self._spacer_str,
            'mark_changes': self.mark_changes,
            'seq_name_prefix': self.name_prefix,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }
