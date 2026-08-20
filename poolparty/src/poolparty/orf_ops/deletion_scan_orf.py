"""Codon-aware deletion scans for ORF sequences."""

from numbers import Integral, Real

import numpy as np

from ..operation import Operation
from ..pool import Pool
from ..types import (
    CardsType,
    ModeType,
    Optional,
    PositionsType,
    RegionType,
    Seq,
    Union,
    beartype,
)
from ..utils import reverse_complement, validate_positions
from ..utils.dna_seq import DnaSeq
from ..utils.parsing_utils import strip_all_tags, validate_single_region
from ._frame import frame_offset, resolve_frame


def _split_cards(cards: CardsType) -> tuple[CardsType, CardsType]:
    """Route universal cards to the stateful scan and ORF cards to the card op."""
    if cards is None:
        return None, None
    requested = set(cards if isinstance(cards, list) else cards.keys())
    valid = {"seq", "state", "codon_positions", "wt_codons", "start", "end"}
    invalid = requested - valid
    if invalid:
        raise ValueError(
            f"Invalid card key(s) {sorted(invalid)} for deletion_scan_orf. "
            f"Valid keys: {sorted(valid)}"
        )
    if isinstance(cards, list):
        return (
            [key for key in cards if key in {"seq", "state"}],
            [key for key in cards if key not in {"seq", "state"}],
        )
    return (
        {key: value for key, value in cards.items() if key in {"seq", "state"}},
        {key: value for key, value in cards.items() if key not in {"seq", "state"}},
    )


def _resolve_region_span(pool: Pool, region: RegionType) -> int:
    """Return the fixed nucleotide span targeted by an ORF operation."""
    if region is None:
        span = pool.seq_length
    elif isinstance(region, str):
        region_obj = next((item for item in pool.regions if item.name == region), None)
        if region_obj is None:
            raise ValueError(f"Region '{region}' is not present in the input pool")
        span = region_obj.seq_length
    else:
        if len(region) != 2:
            raise ValueError(f"region must have exactly 2 elements, got {len(region)}")
        start, end = int(region[0]), int(region[1])
        if start < 0:
            raise ValueError(f"region start must be >= 0, got {start}")
        if end <= start:
            raise ValueError(f"region end ({end}) must be greater than start ({start})")
        if pool.seq_length is not None and end > pool.seq_length:
            raise ValueError(
                f"region end ({end}) cannot exceed pool.seq_length ({pool.seq_length})"
            )
        span = end - start

    if span is None:
        raise ValueError("deletion_scan_orf requires a fixed-length input region")
    return int(span)


def _codon_windows_to_nt(
    codon_positions: PositionsType,
    *,
    span: int,
    frame: int,
    deletion_codons: int,
) -> tuple[list[int], dict[int, int]]:
    """Map coding-order deletion windows to physical nucleotide starts."""
    offset = frame_offset(frame)
    num_codons = max(0, (span - offset) // 3)
    num_windows = num_codons - deletion_codons + 1
    if num_windows <= 0:
        raise ValueError(
            f"deletion_codons ({deletion_codons}) exceeds the number of complete "
            f"codons ({num_codons}) in the selected frame"
        )

    coding_starts = validate_positions(
        codon_positions,
        min_position=0,
        max_position=num_windows - 1,
    )
    if frame > 0:
        physical_starts = [offset + 3 * pos for pos in coding_starts]
    else:
        coding_end = span - offset
        physical_starts = [
            coding_end - 3 * (pos + deletion_codons) for pos in coding_starts
        ]

    return physical_starts, dict(zip(physical_starts, coding_starts))


@beartype
def deletion_scan_orf(
    pool: Union[Pool, str],
    deletion_codons: Integral,
    deletion_marker: Optional[str] = "-",
    codon_positions: PositionsType = None,
    region: RegionType = None,
    frame: Optional[int] = None,
    prefix: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    style: Optional[str] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
) -> Pool:
    """Delete whole-codon windows from an ORF in coding order.

    ``codon_positions`` contains 0-based window starts in coding orientation.
    For negative frames, codon 0 is the rightmost complete codon in the stored
    plus/reference sequence. Orphan bases outside the complete-codon grid are
    never selected.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent DNA pool or sequence string.
    deletion_codons : Integral
        Number of consecutive codons to delete in each state.
    deletion_marker : Optional[str], default='-'
        One-character marker used to replace every deleted nucleotide. Pass
        ``None`` to excise the selected codons instead.
    codon_positions : PositionsType, default=None
        Eligible deletion-window starts in coding-order codon units. ``None``
        selects every valid window; explicit sequences preserve their order.
    region : RegionType, default=None
        ORF region to scan: a named region, ``[start, stop]`` interval, or the
        entire sequence when ``None``.
    frame : Optional[int], default=None
        Reading frame (+1/+2/+3/-1/-2/-3). A named OrfRegion supplies its frame
        when this argument is omitted; other inputs default to +1.
    prefix : Optional[str], default=None
        Prefix for generated sequence names.
    mode : ModeType, default='random'
        Position-selection mode: ``'random'`` or ``'sequential'``.
    num_states : Optional[Integral], default=None
        Number of states. Sequential mode naturally uses the number of eligible
        codon windows; random mode defaults to one stateless draw.
    style : Optional[str], default=None
        Style applied to deletion markers. Ignored for true deletions.
    iter_order : Optional[Real], default=None
        Enumeration priority when combined with other stateful operations.
    cards : CardsType, default=None
        ORF design-card keys to include: ``'codon_positions'``, ``'wt_codons'``,
        ``'start'``, and ``'end'``.

    Returns
    -------
    Pool
        A DNA pool containing one whole-codon deletion per scan state.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..region_ops import region_scan, replace_region

    pool = from_seq(pool) if isinstance(pool, str) else pool

    deletion_codons = int(deletion_codons)
    if deletion_codons <= 0:
        raise ValueError(f"deletion_codons must be > 0, got {deletion_codons}")
    if deletion_marker is not None and len(deletion_marker) != 1:
        raise ValueError("deletion_marker must be None or exactly one character")

    resolved_frame = resolve_frame(region, frame)
    span = _resolve_region_span(pool, region)
    deletion_nt = deletion_codons * 3
    physical_positions, coding_position_by_nt = _codon_windows_to_nt(
        codon_positions,
        span=span,
        frame=resolved_frame,
        deletion_codons=deletion_codons,
    )
    scan_cards, orf_cards = _split_cards(cards)

    validation_op = _DeletionScanOrfValidateOp(
        pool,
        target_region=region,
        target_span=span,
        iter_order=iter_order,
    )
    validated = type(pool)(operation=validation_op)

    marker_name = f"_del_len{deletion_nt}"
    marked = region_scan(
        validated,
        tag_name=marker_name,
        positions=physical_positions,
        region=region,
        remove_tags=False,
        region_length=deletion_nt,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        cards=scan_cards,
        _factory_name="deletion_scan_orf(region_scan)",
    )

    card_op = _DeletionScanOrfCardOp(
        marked,
        marker_name=marker_name,
        target_span=span,
        deletion_codons=deletion_codons,
        frame=resolved_frame,
        coding_position_by_nt=coding_position_by_nt,
        target_region=region,
        cards=orf_cards,
        iter_order=iter_order,
    )
    carded = type(marked)(operation=card_op)

    replacement = "" if deletion_marker is None else deletion_marker * deletion_nt
    replacement_pool = from_seq(
        replacement,
        _factory_name="deletion_scan_orf(from_seq)",
    )
    return replace_region(
        carded,
        replacement_pool,
        marker_name,
        sync=False,
        keep_tags=False,
        iter_order=iter_order,
        _factory_name="deletion_scan_orf(replace_region)",
        _style=style if deletion_marker is not None else None,
    )


class _DeletionScanOrfValidateOp(Operation):
    """Reject unsupported target content before temporary tags are inserted."""

    factory_name = "deletion_scan_orf(validate)"
    design_card_keys = []

    def __init__(
        self,
        parent_pool: Pool,
        *,
        target_region: RegionType,
        target_span: int,
        iter_order: Optional[Real],
        name: Optional[str] = None,
    ) -> None:
        self.target_region = target_region
        self.target_span = target_span
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,
            mode="fixed",
            seq_length=parent_pool.seq_length,
            name=name,
            iter_order=iter_order,
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        parent = parents[0]
        parsed_parent = DnaSeq.from_string(parent.string, parent.style)

        if isinstance(self.target_region, str):
            target_literal = validate_single_region(
                parsed_parent.string, self.target_region
            ).content
        elif self.target_region is None:
            target_literal = parent.string
        else:
            start = int(self.target_region[0])
            literal_start = parsed_parent.nontag_to_literal(start)
            literal_end = parsed_parent.nontag_to_literal(
                start + self.target_span - 1
            ) + 1
            target_literal = parent.string[literal_start:literal_end]

        if "<" in target_literal or ">" in target_literal:
            raise ValueError(
                "deletion_scan_orf does not yet support nested region tags "
                "inside the target ORF"
            )

        target_seq = strip_all_tags(target_literal)
        if len(target_seq) != self.target_span or any(
            char not in "ACGTacgt" for char in target_seq
        ):
            raise ValueError(
                "deletion_scan_orf currently requires a fixed-length, ungapped ACGT region"
            )

        return parent, {}


class _DeletionScanOrfCardOp(Operation):
    """Pass through a marked ORF and report coding-aware deletion cards."""

    factory_name = "deletion_scan_orf(cards)"
    design_card_keys = ["codon_positions", "wt_codons", "start", "end"]

    def __init__(
        self,
        parent_pool: Pool,
        *,
        marker_name: str,
        target_span: int,
        deletion_codons: int,
        frame: int,
        coding_position_by_nt: dict[int, int],
        target_region: RegionType,
        cards: CardsType,
        iter_order: Optional[Real],
        name: Optional[str] = None,
    ) -> None:
        self.marker_name = marker_name
        self.target_span = target_span
        self.deletion_codons = deletion_codons
        self.frame = frame
        self.reverse = frame < 0
        self.coding_position_by_nt = coding_position_by_nt
        self.target_region = target_region
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,
            mode="fixed",
            seq_length=parent_pool.seq_length,
            name=name,
            iter_order=iter_order,
            region=None,
            remove_tags=False,
            cards=cards,
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        parent = parents[0]
        parsed_parent = DnaSeq.from_string(parent.string, parent.style)
        clean_parent = strip_all_tags(parent.string)

        if isinstance(self.target_region, str):
            target = validate_single_region(parsed_parent.string, self.target_region)
            target_start = len(strip_all_tags(parent.string[: target.content_start]))
            target_seq = strip_all_tags(target.content)
        elif self.target_region is None:
            target_start = 0
            target_seq = clean_parent
        else:
            target_start = int(self.target_region[0])
            target_seq = clean_parent[target_start : target_start + self.target_span]

        if len(target_seq) != self.target_span or any(
            char not in "ACGTacgt" for char in target_seq
        ):
            raise ValueError(
                "deletion_scan_orf currently requires a fixed-length, ungapped ACGT region"
            )

        marked_region = validate_single_region(parsed_parent.string, self.marker_name)
        physical_seq = strip_all_tags(marked_region.content).upper()
        marker_start = len(strip_all_tags(parent.string[: marked_region.content_start]))
        start = marker_start - target_start
        end = start + len(physical_seq)
        if start < 0 or end > self.target_span:
            raise RuntimeError("Selected ORF deletion interval falls outside the target region")

        try:
            coding_start = self.coding_position_by_nt[start]
        except KeyError as exc:
            raise RuntimeError(
                f"Unexpected physical deletion start {start} for {self.marker_name}"
            ) from exc

        coding_seq = reverse_complement(physical_seq) if self.reverse else physical_seq
        wt_codons = tuple(
            coding_seq[i : i + 3] for i in range(0, len(coding_seq), 3)
        )
        codon_positions = tuple(
            range(coding_start, coding_start + self.deletion_codons)
        )

        return parent, {
            "codon_positions": codon_positions,
            "wt_codons": wt_codons,
            "start": start,
            "end": end,
        }
