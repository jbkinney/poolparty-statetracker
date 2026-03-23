"""Orientation operation - produce forward or reverse complement variants."""

from numbers import Real

import numpy as np

from ..operation import Operation
from ..pool import Pool
from ..types import CardsType, ModeType, Optional, Pool_type, RegionType, Seq, Union, beartype
from ..utils import dna_utils
from ..utils.dna_seq import DnaSeq


@beartype
def orientation(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    rc_rate: Real = 0.5,
    prefix: Optional[str] = None,
    mode: ModeType = "sequential",
    num_states: Optional[int] = None,
    iter_order: Optional[Real] = None,
    style: Optional[str] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = "orientation",
) -> Pool:
    """
    Create a Pool that produces forward or reverse-complement variants.

    Unlike ``rc`` (which always reverse complements), this operation tracks
    orientation as state: sequential mode enumerates both orientations,
    random mode coin-flips with configurable ``rc_rate``.

    Parameters
    ----------
    pool : Union[Pool_type, str]
        Parent pool or sequence string.
    region : RegionType, default=None
        Region to apply transformation to. Can be marker name (str),
        [start, stop], or None for the entire sequence.
    rc_rate : Real, default=0.5
        Probability of reverse complement in random mode. Ignored in
        sequential mode.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : ModeType, default='sequential'
        Selection mode: 'sequential' enumerates forward then rc;
        'random' coin-flips per sample.
    num_states : Optional[int], default=None
        Number of states. In sequential mode, overrides the natural
        count of 2 (cycling). In random mode, if None defaults to 1
        (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    style : Optional[str], default=None
        Style to apply to reverse-complemented sequences.
    cards : CardsType, default=None
        Design card column specification.

    Returns
    -------
    Pool
        A Pool that generates forward or reverse-complement variants.

    Raises
    ------
    ValueError
        If ``rc_rate`` is not in [0, 1] or ``mode='fixed'``.
    """
    from ..fixed_ops.from_seq import from_seq

    pool = (
        from_seq(pool, _factory_name=f"{_factory_name}(from_seq)")
        if isinstance(pool, str)
        else pool
    )
    op = OrientationOp(
        pool=pool,
        region=region,
        rc_rate=rc_rate,
        style=style,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        cards=cards,
        _factory_name=_factory_name,
    )
    pool_class = type(pool)
    return pool_class(operation=op)


class OrientationOp(Operation):
    """Produce forward or reverse-complement variants with state tracking."""

    factory_name = "orientation"
    design_card_keys = ["orientation"]

    def __init__(
        self,
        pool: Pool,
        region: RegionType = None,
        rc_rate: Real = 0.5,
        style: Optional[str] = None,
        prefix: Optional[str] = None,
        mode: ModeType = "sequential",
        num_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
        _factory_name: Optional[str] = "orientation",
    ) -> None:
        if _factory_name is not None:
            self.factory_name = _factory_name

        if mode == "fixed":
            raise ValueError(
                "mode='fixed' is not supported for orientation. "
                "Use rc() for fixed reverse complement or from_seq() for fixed forward."
            )

        if rc_rate < 0 or rc_rate > 1:
            raise ValueError(f"rc_rate must be between 0 and 1, got {rc_rate}")

        self._rc_rate = rc_rate
        self._style = style

        natural_num_states = None
        match mode:
            case "sequential":
                natural_num_states = 2
                if num_states is None:
                    num_states = natural_num_states
            case "random":
                pass

        super().__init__(
            parent_pools=[pool],
            num_states=num_states,
            mode=mode,
            seq_length=pool.seq_length,
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
        """Return Seq and orientation design card."""
        from ..utils.parsing_utils import strip_all_tags
        from ..utils.style_utils import SeqStyle, styles_suppressed

        if self.mode == "random":
            if rng is None:
                raise RuntimeError("Random mode requires RNG - use Party.generate(seed=...)")
            is_rc = rng.random() < self._rc_rate
        else:
            state = self.state.value
            is_rc = ((0 if state is None else state) % 2) == 1

        parent_seq = parents[0]

        if is_rc:
            clean_seq = strip_all_tags(parent_seq.string)
            rc_string = dna_utils.reverse_complement(clean_seq)

            if styles_suppressed():
                output_seq = DnaSeq(rc_string, None)
            else:
                output_style = SeqStyle.full(len(rc_string), self._style)
                output_seq = DnaSeq(rc_string, output_style)
        else:
            output_seq = parent_seq

        return output_seq, {"orientation": "rc" if is_rc else "forward"}
