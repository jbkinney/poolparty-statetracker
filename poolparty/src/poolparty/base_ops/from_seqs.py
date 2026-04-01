"""FromSeqs operation - create a pool from a list of sequences."""

from numbers import Real

import numpy as np

from ..dna_pool import DnaPool
from ..operation import Operation
from ..pool import Pool
from ..types import CardsType, Integral, ModeType, Optional, Pool_type, RegionType, Seq, Sequence, Union, beartype
from ..utils import dna_utils
from ..utils.dna_seq import DnaSeq


@beartype
def from_seqs(
    seqs: Sequence[str],
    pool: Optional[Union[Pool, str]] = None,
    region: RegionType = None,
    style: Optional[str] = None,
    seq_names: Optional[Sequence[str]] = None,
    prefix: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = None,
) -> DnaPool:
    """
    Create a Pool containing the specified sequences.

    Parameters
    ----------
    seqs : Sequence[str]
        Sequence of string sequences to include in the pool.
    pool : Optional[Union[Pool, str]], default=None
        Background pool or sequence. If provided with region, selected sequence
        replaces the region content.
    region : RegionType, default=None
        Region to replace in pool. Can be a marker name or [start, stop] interval.
        Required if pool is provided.
    seq_names : Optional[Sequence[str]], default=None
        Explicit names for each sequence. If provided, these are used directly.
    prefix : Optional[str], default=None
        Prefix for auto-generated names (e.g., 'seq_' produces 'seq_0', 'seq_1', ...).
        Cannot be used together with seq_names.
    mode : ModeType, default='random'
        Sequence selection mode: 'sequential' or 'random'.
    num_states : Optional[int], default=None
        Number of states. In sequential mode, overrides the computed count
        (cycling if greater, clipping if less). In random mode, if None
        defaults to 1 (pure random sampling).
    style : Optional[str], default=None
        Style to apply to output sequences (e.g., 'red', 'blue bold').
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    cards : list[str] or dict, optional
        Design card keys to include. Available keys: ``'seq_name'``, ``'seq_index'``.

    Returns
    -------
    Pool_type
        A Pool object yielding the provided sequences using the specified selection mode.

    Raises
    ------
    TypeError
        If seqs is a bare string instead of a list of strings.
    ValueError
        If pool is provided without region.
    """
    if isinstance(seqs, str):
        raise TypeError(
            "from_seqs expects a list of sequences, not a bare string. "
            "Use from_seqs(['...']) for a single sequence, or from_seq('...') for a fixed pool."
        )

    from ..fixed_ops.from_seq import from_seq

    pool_obj = from_seq(pool) if isinstance(pool, str) else pool
    op = FromSeqsOp(
        seqs,
        parent_pool=pool_obj,
        region=region,
        style=style,
        seq_names=seq_names,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        name=None,
        iter_order=iter_order,
        cards=cards,
        _factory_name=_factory_name,
    )
    result_pool = DnaPool(operation=op)
    return result_pool


class FromSeqsOp(Operation):
    """Create a pool from a list of sequences."""

    factory_name = "from_seqs"
    design_card_keys = ["seq_name", "seq_index"]

    def __init__(
        self,
        seqs: Sequence[str],
        parent_pool: Optional[Pool] = None,
        region: RegionType = None,
        style: Optional[str] = None,
        seq_names: Optional[Sequence[str]] = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
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

        # Validate parent_pool/region combination
        if parent_pool is not None and region is None:
            raise ValueError(
                "region is required when parent_pool is provided. "
                "Specify which region of parent_pool to replace with the selected sequence."
            )

        self._style = style

        if len(seqs) == 0:
            raise ValueError("seqs must not be empty")
        if mode == "fixed" and len(seqs) != 1:
            raise ValueError("mode='fixed' requires exactly 1 sequence")
        if seq_names is not None and prefix is not None:
            raise ValueError("Cannot specify both seq_names and prefix")
        self.seqs = list(seqs)
        # Track whether explicit seq_names were provided (for compute_name_contributions)
        self._seq_names_explicit = seq_names is not None
        self.seq_names = list(seq_names) if seq_names else [f"seq_{i}" for i in range(len(seqs))]
        # Store current index for name computation
        self._current_idx: int = 0
        if len(self.seq_names) != len(self.seqs):
            raise ValueError("seq_names must have same length as seqs")

        # Determine num_states based on mode
        natural_num_states = None
        match mode:
            case "sequential":
                # Natural count is the number of sequences
                natural_num_states = len(seqs)
                # Use user-provided num_states if given, else natural count
                if num_states is None:
                    num_states = natural_num_states
            case "random":
                # num_states stays as provided (or None for pure random mode)
                pass
            case _:
                num_states = 1

        # Use lengths without markers (includes all chars except marker tags)
        lengths = [dna_utils.get_length_without_tags(s) for s in self.seqs]
        seq_length = lengths[0] if all(L == lengths[0] for L in lengths) else None

        parent_pools_list = [parent_pool] if parent_pool is not None else []
        super().__init__(
            parent_pools=parent_pools_list,
            num_states=num_states,
            mode=mode,
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=region,
            _natural_num_states=natural_num_states,
            cards=cards,
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return Seq and design card."""
        if self.mode == "random":
            if rng is None:
                raise RuntimeError(
                    f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)"
                )
            idx = int(rng.integers(0, len(self.seqs)))
        elif self.mode == "fixed":
            # Fixed mode - always use index 0
            idx = 0
        else:
            # Sequential mode - use state value (0 when inactive)
            state = self.state.value
            idx = (0 if state is None else state) % len(self.seqs)

        # Store index for name computation
        self._current_idx = idx

        seq_string = self.seqs[idx]

        # Apply style to all positions if specified
        from ..utils.style_utils import SeqStyle, styles_suppressed

        if styles_suppressed():
            output_seq = DnaSeq(seq_string, None)
        else:
            output_style = SeqStyle.full(len(seq_string), self._style)
            output_seq = DnaSeq(seq_string, output_style)

        return output_seq, {
            "seq_name": self.seq_names[int(idx)],
            "seq_index": int(idx),
        }

    def compute_name_contributions(self, global_state=None, max_global_state=None) -> list[str]:
        """Compute name contributions - explicit seq_names or prefix pattern."""
        # Check if state is inactive (for branch selection)
        if not self.state.is_active:
            return []
        if self._seq_names_explicit:
            # Use explicit seq_name for current index
            return [self.seq_names[self._current_idx]]
        # Otherwise use default prefix logic from base class
        return super().compute_name_contributions(global_state, max_global_state)

    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        params = super()._get_copy_params()
        # Only include seq_names if explicitly set by user
        params["seq_names"] = self.seq_names if self._seq_names_explicit else None
        return params
