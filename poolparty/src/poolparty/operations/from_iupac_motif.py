"""FromIupacMotif operation - generate DNA sequences from IUPAC notation."""
from numbers import Real
from ..types import Pool_type, Sequence, ModeType, Optional, beartype
from ..operation import Operation
from ..pool import Pool
from ..alphabet import IUPAC_TO_DNA
import numpy as np


@beartype
def from_iupac_motif(
    iupac_seq: str,
    mark_changes: Optional[bool] = None,
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
    """
    op = FromIupacMotifOp(
        iupac_seq=iupac_seq,
        mark_changes=mark_changes,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=op_name,
        iter_order=op_iter_order,
    )
    pool = Pool(operation=op, name=name, iter_order=iter_order)
    return pool


@beartype
class FromIupacMotifOp(Operation):
    """Generate DNA sequences from IUPAC notation."""
    factory_name = "from_iupac_motif"
    design_card_keys = ['iupac_state']

    def __init__(
        self,
        iupac_seq: str,
        mark_changes: Optional[bool] = None,
        mode: ModeType = 'random',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize FromIupacMotifOp."""
        from ..party import get_active_party
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "from_iupac_motif requires an active Party context. "
                "Use 'with pp.Party() as party:' to create one."
            )
        if not iupac_seq:
            raise ValueError("iupac_seq must be a non-empty string")
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")

        # Validate and build position options
        # Handle ignore chars (e.g., '.', '-', ' ') as pass-through positions
        ignore_chars = party._alphabet.ignore_chars
        iupac_seq_upper = iupac_seq.upper()
        invalid_chars = set()
        position_options = []
        degenerate_positions = []
        pos_idx = 0  # Track position in position_options
        for char, char_upper in zip(iupac_seq, iupac_seq_upper):
            if char in ignore_chars:
                # Pass through ignore chars unchanged
                position_options.append([char])
                pos_idx += 1
            elif char_upper in IUPAC_TO_DNA:
                opts = IUPAC_TO_DNA[char]
                position_options.append(opts)
                if len(opts) > 1:
                    degenerate_positions.append(pos_idx)
                pos_idx += 1
            else:
                invalid_chars.add(char)

        if invalid_chars:
            raise ValueError(
                f"iupac_seq contains invalid IUPAC character(s): {sorted(invalid_chars)}. "
                f"Valid IUPAC characters are: {sorted(IUPAC_TO_DNA.keys())} "
                f"(plus ignore characters: {sorted(ignore_chars)})"
            )

        self.iupac_seq = iupac_seq
        self.position_options = position_options
        # Resolve mark_changes from party defaults if not explicitly set
        if mark_changes is None:
            mark_changes = party.get_default('mark_changes', False)
        self.mark_changes = mark_changes
        self.degenerate_positions = set(degenerate_positions)

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
        seq_length = party._alphabet.get_length_without_markers(iupac_seq)
        super().__init__(
            parent_pools=[],
            num_states=num_states,
            mode=mode,
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
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
        
        # Apply swapcase to degenerate positions if mark_changes is True
        if self.mark_changes:
            for i in self.degenerate_positions:
                result[i] = result[i].swapcase()
        
        seq = ''.join(result)
        return {'seq_0': seq}

    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'iupac_seq': self.iupac_seq,
            'mark_changes': self.mark_changes,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }
