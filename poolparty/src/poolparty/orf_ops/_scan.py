"""Shared geometry and validation helpers for ORF scan operations."""

from numbers import Real

import numpy as np

from ..operation import Operation
from ..pool import Pool
from ..types import Optional, PositionsType, RegionType, Seq
from ..utils import validate_positions
from ..utils.dna_seq import DnaSeq
from ..utils.parsing_utils import strip_all_tags, validate_single_region
from ._frame import frame_offset


def resolve_region_span(pool: Pool, region: RegionType, operation_name: str) -> int:
    """Return the fixed nucleotide span targeted by an ORF scan."""
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
        raise ValueError(f"{operation_name} requires a fixed-length input region")
    return int(span)


def num_complete_codons(span: int, frame: int) -> int:
    """Return the number of complete codons in a framed nucleotide span."""
    return max(0, (span - frame_offset(frame)) // 3)


def resolve_codon_starts(
    codon_positions: PositionsType,
    *,
    num_slots: int,
    error_context: str,
) -> list[int]:
    """Resolve a coding-order position specification against a slot count."""
    if num_slots <= 0:
        raise ValueError(f"No valid codon positions for {error_context}")
    return validate_positions(
        codon_positions,
        min_position=0,
        max_position=num_slots - 1,
    )


def codon_starts_to_nt(
    coding_starts: list[int],
    *,
    span: int,
    frame: int,
    item_codons: int,
    splice: bool,
) -> list[int]:
    """Map coding-order starts or splice slots to physical nucleotide positions."""
    offset = frame_offset(frame)
    if frame > 0:
        return [offset + 3 * pos for pos in coding_starts]

    coding_end = span - offset
    if splice:
        return [coding_end - 3 * pos for pos in coding_starts]
    return [coding_end - 3 * (pos + item_codons) for pos in coding_starts]


def get_target_start_and_seq(
    parent: Seq,
    *,
    target_region: RegionType,
    target_span: int,
) -> tuple[int, str]:
    """Return global nontag start and clean content for a target region."""
    parsed_parent = DnaSeq.from_string(parent.string, parent.style)
    clean_parent = strip_all_tags(parent.string)

    if isinstance(target_region, str):
        target = validate_single_region(parsed_parent.string, target_region)
        target_start = len(strip_all_tags(parent.string[: target.content_start]))
        target_seq = strip_all_tags(target.content)
    elif target_region is None:
        target_start = 0
        target_seq = clean_parent
    else:
        target_start = int(target_region[0])
        target_seq = clean_parent[target_start : target_start + target_span]

    return target_start, target_seq


def validate_orf_scan_input(
    pool: Pool,
    *,
    target_region: RegionType,
    target_span: int,
    operation_name: str,
    iter_order: Optional[Real],
) -> Pool:
    """Add a fixed validation pass before a temporary scan marker is inserted."""
    op = OrfScanValidateOp(
        pool,
        target_region=target_region,
        target_span=target_span,
        operation_name=operation_name,
        iter_order=iter_order,
    )
    return type(pool)(operation=op)


class OrfScanValidateOp(Operation):
    """Reject unsupported target content before temporary tags are inserted."""

    design_card_keys = []

    def __init__(
        self,
        parent_pool: Pool,
        *,
        target_region: RegionType,
        target_span: int,
        operation_name: str,
        iter_order: Optional[Real],
        name: Optional[str] = None,
    ) -> None:
        self.target_region = target_region
        self.target_span = target_span
        self.operation_name = operation_name
        self.factory_name = f"{operation_name}(validate)"
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
                f"{self.operation_name} does not yet support nested region tags "
                "inside the target ORF"
            )

        target_seq = strip_all_tags(target_literal)
        if len(target_seq) != self.target_span or any(
            char not in "ACGTacgt" for char in target_seq
        ):
            raise ValueError(
                f"{self.operation_name} currently requires a fixed-length, "
                "ungapped ACGT region"
            )

        return parent, {}
