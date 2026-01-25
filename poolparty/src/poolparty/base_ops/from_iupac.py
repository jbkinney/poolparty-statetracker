"""FromIupac operation - generate DNA sequences from IUPAC notation."""
from numbers import Real
from ..types import Pool_type, Sequence, ModeType, Optional, Union, RegionType, beartype
from ..operation import Operation
from ..pool import Pool
from ..utils import dna_utils
import numpy as np


@beartype
def from_iupac(
    iupac_seq: str,
    bg_pool: Optional[Union[Pool, str]] = None,
    region: RegionType = None,
    prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_states: Optional[int] = None,
    iter_order: Optional[Real] = None,
    style: Optional[str] = None,
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
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : ModeType, default='random'
        Sequence selection mode: 'sequential' or 'random'.
    num_states : Optional[int], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    style : Optional[str], default=None
        Style to apply to generated sequences (e.g., 'red', 'blue bold').

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
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        name=None,
        iter_order=iter_order,
        style=style,
    )
    pool = Pool(operation=op)
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
        prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
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

        # Validate and build position options
        # Handle ignore chars (e.g., '.', '-', ' ') as pass-through positions
        iupac_seq_upper = iupac_seq.upper()
        invalid_chars = set()
        position_options = []
        for char, char_upper in zip(iupac_seq, iupac_seq_upper):
            if char in dna_utils.IGNORE_CHARS:
                # Pass through ignore chars unchanged
                position_options.append([char])
            elif char_upper in dna_utils.IUPAC_TO_DNA:
                opts = dna_utils.IUPAC_TO_DNA[char]
                position_options.append(opts)
            else:
                invalid_chars.add(char)

        if invalid_chars:
            raise ValueError(
                f"iupac_seq contains invalid IUPAC character(s): {sorted(invalid_chars)}. "
                f"Valid IUPAC characters are: {sorted(dna_utils.IUPAC_TO_DNA.keys())} "
                f"(plus ignore characters: {sorted(dna_utils.IGNORE_CHARS)})"
            )

        self.iupac_seq = iupac_seq
        self.position_options = position_options
        self._style = style

        # Compute total states as product of possibilities at each position
        total_states = 1
        for options in position_options:
            total_states *= len(options)

        match mode:
            case 'sequential':
                num_states = total_states
            case 'random':
                # num_states stays None for pure random mode
                pass
            case _:
                num_states = 1

        self._total_states = total_states
        # Use length without markers for consistency
        seq_length = dna_utils.get_length_without_markers(iupac_seq)
        
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
        parent_styles: list | None = None,
    ) -> dict:
        """Return design card and DNA sequence together."""
        if self.mode == 'random':
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")
            state = rng.integers(0, self._total_states)
        else:
            state_value = self.state.value
            state = (0 if state_value is None else state_value) % self._total_states
        
        # Mixed-radix conversion: map state to specific sequence
        result = []
        remaining = state
        for position_opts in reversed(self.position_options):
            result.append(position_opts[remaining % len(position_opts)])
            remaining //= len(position_opts)
        result = list(reversed(result))
        seq = ''.join(result)
        
        # Apply styling if requested
        from ..types import StyleList
        output_styles: StyleList = []
        if self._style:
            # Style all positions of the generated sequence
            positions = np.arange(len(seq), dtype=np.int64)
            output_styles = [(self._style, positions)]
        
        return {
            'iupac_state': state,
            'seq': seq,
            'style': output_styles,
        }

    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'iupac_seq': self.iupac_seq,
            'bg_pool': self.parent_pools[0] if self.parent_pools else None,
            'region': self._region,
            'prefix': self.name_prefix,
            'mode': self.mode,
            'num_states': self.num_values if self.mode == 'random' and self.num_values is not None and self.num_values > 1 else None,
            'name': None,
            'iter_order': self.iter_order,
            'style': self._style,
        }
