"""Codon-aware insertion scans for ORF sequences."""

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
from ..utils.dna_seq import DnaSeq
from ..utils.dna_utils import reverse_complement
from ..utils.parsing_utils import strip_all_tags, validate_single_region
from ._frame import complete_codon_count, frame_offset, resolve_frame
from ._scan import (
    codon_starts_to_nt,
    codons_to_aas,
    get_target_start_and_seq,
    resolve_codon_starts,
    resolve_region_span,
    validate_orf_scan_input,
)


def _split_cards(cards: CardsType, *, replace: bool) -> tuple[CardsType, CardsType, CardsType]:
    """Route cards to the position, insert-content, and target card ops."""
    if cards is None:
        return None, None, None

    mode_keys = (
        {"codon_positions", "wt_codons", "wt_aas", "start", "end"}
        if replace
        else {"codon_slot", "start", "end"}
    )
    content_keys = {"mut_codons", "mut_aas"}
    valid = {"seq", "state"} | content_keys | mode_keys
    requested = set(cards if isinstance(cards, list) else cards.keys())
    invalid = requested - valid
    if invalid:
        raise ValueError(
            f"Invalid card key(s) {sorted(invalid)} for insertion_scan_orf. "
            f"Valid keys for replace={replace}: {sorted(valid)}"
        )

    if isinstance(cards, list):
        return (
            [key for key in cards if key in {"seq", "state"}],
            [key for key in cards if key in content_keys],
            [key for key in cards if key in mode_keys],
        )
    return (
        {key: value for key, value in cards.items() if key in {"seq", "state"}},
        {key: value for key, value in cards.items() if key in content_keys},
        {key: value for key, value in cards.items() if key in mode_keys},
    )


@beartype
def insertion_scan_orf(
    pool: Union[Pool, str],
    insertion_pool: Union[Pool, str],
    codon_positions: PositionsType = None,
    region: RegionType = None,
    frame: Optional[int] = None,
    replace: bool = False,
    style: Optional[str] = None,
    prefix: Optional[str] = None,
    prefix_position: Optional[str] = None,
    prefix_insert: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
) -> Pool:
    """Insert coding-oriented whole codons at ORF positions.

    With ``replace=False``, ``codon_positions`` selects boundaries between
    codons and the card key is ``codon_slot``. With ``replace=True``, positions
    select whole-codon overwrite windows. Insert cards report ``mut_codons``
    and ``mut_aas``; overwrite cards can additionally report ``wt_codons`` and
    ``wt_aas``. For negative frames, the complete selected insert is
    reverse-complemented once before physical placement.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent DNA pool or sequence string.
    insertion_pool : Union[Pool, str]
        Fixed-length coding-oriented DNA pool to insert. Its length must be a
        multiple of three. Upstream cards remain coding-oriented.
    codon_positions : PositionsType, default=None
        Eligible coding-order splice slots or overwrite-window starts.
    region : RegionType, default=None
        ORF region to scan: a named region, ``[start, stop]`` interval, or the
        entire sequence when ``None``.
    frame : Optional[int], default=None
        Reading frame (+1/+2/+3/-1/-2/-3). A named OrfRegion supplies its frame
        when omitted; other inputs default to +1.
    replace : bool, default=False
        If ``False``, splice at codon boundaries. If ``True``, overwrite the
        same number of whole codons as supplied by ``insertion_pool``.
    style : Optional[str], default=None
        Style applied to inserted content.
    prefix : Optional[str], default=None
        Prefix for the combined position-by-insert state index.
    prefix_position : Optional[str], default=None
        Prefix for the selected position-state index.
    prefix_insert : Optional[str], default=None
        Prefix for the insertion-pool state index.
    mode : ModeType, default='random'
        Position-selection mode: ``'random'`` or ``'sequential'``.
    num_states : Optional[Integral], default=None
        Number of position states. Sequential mode naturally uses all eligible
        slots/windows; random mode defaults to one stateless draw.
    iter_order : Optional[Real], default=None
        Enumeration priority when combined with other stateful operations.
    cards : CardsType, default=None
        Splice keys: ``'codon_slot'``, ``'mut_codons'``, ``'mut_aas'``,
        ``'start'``, ``'end'``. Overwrite keys: ``'codon_positions'``,
        ``'wt_codons'``, ``'wt_aas'``, ``'mut_codons'``, ``'mut_aas'``,
        ``'start'``, ``'end'``.

    Returns
    -------
    Pool
        A DNA pool containing coding-aware insertions or overwrites.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.passthrough import passthrough
    from ..region_ops import region_scan, replace_region

    pool = from_seq(pool) if isinstance(pool, str) else pool
    insertion_pool = from_seq(insertion_pool) if isinstance(insertion_pool, str) else insertion_pool

    insertion_length = insertion_pool.seq_length
    if insertion_length is None:
        raise ValueError("insertion_pool must have a defined seq_length")
    insertion_length = int(insertion_length)
    if insertion_length <= 0:
        raise ValueError("insertion_pool must contain at least one complete codon")
    if insertion_length % 3 != 0:
        raise ValueError(f"insertion_pool.seq_length ({insertion_length}) must be divisible by 3")

    resolved_frame = resolve_frame(region, frame)
    span = resolve_region_span(pool, region, "insertion_scan_orf")
    item_codons = insertion_length // 3
    target_codons = complete_codon_count(span, resolved_frame)
    if replace:
        num_slots = target_codons - item_codons + 1
    else:
        num_slots = target_codons + 1 if span >= frame_offset(resolved_frame) else 0
    if replace and num_slots <= 0:
        raise ValueError(
            f"insertion_pool contains {item_codons} codon(s), which exceeds the "
            f"number of complete target codons ({target_codons})"
        )

    coding_starts = resolve_codon_starts(
        codon_positions,
        num_slots=num_slots,
        error_context="ORF insertion",
    )
    physical_positions = codon_starts_to_nt(
        coding_starts,
        span=span,
        frame=resolved_frame,
        item_codons=item_codons,
        splice=not replace,
    )
    coding_position_by_nt = dict(zip(physical_positions, coding_starts))
    scan_cards, content_cards, target_cards = _split_cards(cards, replace=replace)

    validated = validate_orf_scan_input(
        pool,
        target_region=region,
        target_span=span,
        operation_name="insertion_scan_orf",
        iter_order=iter_order,
    )

    insertion_state = insertion_pool.state
    insertion_num_states = insertion_pool.num_states
    oriented_op = _InsertionScanOrfContentOp(
        insertion_pool,
        expected_length=insertion_length,
        reverse=resolved_frame < 0,
        cards=content_cards,
        iter_order=iter_order,
    )
    oriented_insert = type(insertion_pool)(operation=oriented_op)

    marker_name = f"_rep_len{insertion_length}" if replace else "_ins"
    marker_length = insertion_length if replace else 0
    marked = region_scan(
        validated,
        tag_name=marker_name,
        positions=physical_positions,
        region=region,
        remove_tags=False,
        region_length=marker_length,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        cards=scan_cards,
        _factory_name="insertion_scan_orf(region_scan)",
    )
    marked = marked.named(f"{marked.name}:insertion_scan_orf(intermediate)")
    position_state = marked.operation.state

    card_op = _InsertionScanOrfCardOp(
        marked,
        marker_name=marker_name,
        target_span=span,
        item_codons=item_codons,
        frame=resolved_frame,
        coding_position_by_nt=coding_position_by_nt,
        target_region=region,
        replace=replace,
        cards=target_cards,
        iter_order=iter_order,
    )
    carded = type(marked)(operation=card_op)

    result = replace_region(
        carded,
        oriented_insert,
        marker_name,
        sync=False,
        keep_tags=False,
        iter_order=iter_order,
        _factory_name="insertion_scan_orf(replace_region)",
        _style=style,
    )

    if any([prefix, prefix_position, prefix_insert]):
        num_insert_states = insertion_num_states or 1

        def compute_names():
            if not position_state.is_active:
                return []
            if insertion_state is not None and not insertion_state.is_active:
                return []

            position_index = position_state.value
            insert_index = insertion_state.value if insertion_state else 0
            contributions = []
            if prefix:
                combined_index = position_index * num_insert_states + insert_index
                contributions.append(f"{prefix}_{combined_index}")
            if prefix_position:
                contributions.append(f"{prefix_position}_{position_index}")
            if prefix_insert:
                contributions.append(f"{prefix_insert}_{insert_index}")
            return contributions

        result = passthrough(
            result,
            _name_fn=compute_names,
            iter_order=iter_order,
            _factory_name="insertion_scan_orf(naming)",
        )

    return result


class _InsertionScanOrfContentOp(Operation):
    """Validate an insert state and orient it for physical placement."""

    factory_name = "insertion_scan_orf(orient_insert)"
    design_card_keys = ["mut_codons", "mut_aas"]

    def __init__(
        self,
        parent_pool: Pool,
        *,
        expected_length: int,
        reverse: bool,
        cards: CardsType,
        iter_order: Optional[Real],
        name: Optional[str] = None,
    ) -> None:
        self.expected_length = expected_length
        self.reverse = reverse
        self.report_mut_cards = cards is not None and any(
            key in cards for key in ("mut_codons", "mut_aas")
        )
        self.report_mut_aas = cards is not None and "mut_aas" in cards
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,
            mode="fixed",
            seq_length=parent_pool.seq_length,
            name=name,
            iter_order=iter_order,
            cards=cards,
        )
        self.codon_table = self._party.codon_table

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        parent = parents[0]
        if "<" in parent.string or ">" in parent.string:
            raise ValueError("insertion_scan_orf does not support tagged insertion content")
        clean = strip_all_tags(parent.string)
        if len(clean) != self.expected_length or any(char not in "ACGTacgt" for char in clean):
            raise ValueError(
                "insertion_scan_orf requires every insertion state to contain "
                "ungapped ACGT sequence"
            )

        insert = DnaSeq.from_string(parent.string, parent.style)
        card = {}
        if self.report_mut_cards:
            coding_seq = clean.upper()
            mut_codons = tuple(coding_seq[i : i + 3] for i in range(0, len(coding_seq), 3))
            card["mut_codons"] = mut_codons
            if self.report_mut_aas:
                card["mut_aas"] = codons_to_aas(mut_codons, self.codon_table)
        if not self.reverse:
            return insert, card

        reversed_style = insert.style.reversed() if insert.style is not None else None
        return DnaSeq.from_string(reverse_complement(insert.string), reversed_style), card


class _InsertionScanOrfCardOp(Operation):
    """Pass through a marked ORF and report coding-aware insertion cards."""

    factory_name = "insertion_scan_orf(cards)"
    design_card_keys = [
        "codon_slot",
        "codon_positions",
        "wt_codons",
        "wt_aas",
        "start",
        "end",
    ]

    def __init__(
        self,
        parent_pool: Pool,
        *,
        marker_name: str,
        target_span: int,
        item_codons: int,
        frame: int,
        coding_position_by_nt: dict[int, int],
        target_region: RegionType,
        replace: bool,
        cards: CardsType,
        iter_order: Optional[Real],
        name: Optional[str] = None,
    ) -> None:
        self.marker_name = marker_name
        self.target_span = target_span
        self.item_codons = item_codons
        self.frame = frame
        self.reverse = frame < 0
        self.coding_position_by_nt = coding_position_by_nt
        self.target_region = target_region
        self.replace = replace
        self.report_wt_aas = cards is not None and "wt_aas" in cards
        super().__init__(
            parent_pools=[parent_pool],
            num_states=1,
            mode="fixed",
            seq_length=parent_pool.seq_length,
            name=name,
            iter_order=iter_order,
            cards=cards,
        )
        self.codon_table = self._party.codon_table

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        parent = parents[0]
        target_start, target_seq = get_target_start_and_seq(
            parent,
            target_region=self.target_region,
            target_span=self.target_span,
        )
        if len(target_seq) != self.target_span or any(
            char not in "ACGTacgt" for char in target_seq
        ):
            raise ValueError(
                "insertion_scan_orf currently requires a fixed-length, ungapped ACGT region"
            )

        marked_region = validate_single_region(parent.string, self.marker_name)
        physical_seq = strip_all_tags(marked_region.content).upper()
        marker_start = len(strip_all_tags(parent.string[: marked_region.content_start]))
        start = marker_start - target_start
        end = start + len(physical_seq)
        if start < 0 or end > self.target_span:
            raise RuntimeError("Selected ORF insertion interval falls outside the target region")

        try:
            coding_start = self.coding_position_by_nt[start]
        except KeyError as exc:
            raise RuntimeError(
                f"Unexpected physical insertion start {start} for {self.marker_name}"
            ) from exc

        if not self.replace:
            return parent, {"codon_slot": coding_start, "start": start, "end": end}

        coding_seq = physical_seq
        if self.reverse:
            coding_seq = reverse_complement(physical_seq)
        wt_codons = tuple(coding_seq[i : i + 3] for i in range(0, len(coding_seq), 3))
        wt_aas = (
            codons_to_aas(wt_codons, self.codon_table) if self.report_wt_aas else tuple()
        )
        codon_positions = tuple(range(coding_start, coding_start + self.item_codons))
        return parent, {
            "codon_positions": codon_positions,
            "wt_codons": wt_codons,
            "wt_aas": wt_aas,
            "start": start,
            "end": end,
        }
