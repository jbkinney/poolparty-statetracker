"""StylizeOrf operation - apply ORF-aware inline styling to sequences."""

from numbers import Real

import numpy as np

from ..dna_pool import DnaPool
from ..operation import Operation
from ..pool import Pool
from ..region import VALID_FRAMES, OrfRegion
from ..types import Optional, Pool_type, RegionType, Seq, Union, beartype
from ..utils.dna_utils import VALID_CHARS


def _resolve_frame(region: RegionType, frame: Optional[int]) -> int:
    """Resolve the frame value, looking up from OrfRegion if needed.

    Backward compatibility: defaults to frame=1 when region is None or an interval.
    When region is a named OrfRegion, uses the stored frame.
    When region is a named plain Region, raises an error (must specify frame).
    """
    from ..party import get_active_party

    # If frame is explicitly provided, validate and use it
    if frame is not None:
        if frame not in VALID_FRAMES:
            raise ValueError(f"frame must be one of {sorted(VALID_FRAMES)}, got {frame}")
        return frame

    # frame is None - try to get from OrfRegion or use default
    if region is None or not isinstance(region, str):
        # Backward compatibility: default to frame=1 for non-named regions
        return 1

    # region is a string (region name) - look it up
    party = get_active_party()
    if party is None:
        raise RuntimeError("No active Party context.")

    if not party.has_region(region):
        # Region doesn't exist yet - use default frame=1
        return 1

    registered_region = party.get_region(region)
    if isinstance(registered_region, OrfRegion):
        return registered_region.frame
    else:
        raise ValueError(
            f"Region '{region}' is a plain Region, not an OrfRegion. "
            f"frame must be specified explicitly, or use annotate_orf() to "
            f"upgrade the region to an OrfRegion with a frame."
        )


@beartype
def stylize_orf(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    *,
    style_codons: Optional[list[str]] = None,
    style_frames: Optional[list[str]] = None,
    frame: Optional[int] = None,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
) -> Pool:
    """
    Apply ORF-aware inline styling to sequences.

    Styles are attached directly to sequences based on codon boundaries or reading frame.

    Parameters
    ----------
    pool : Union[Pool_type, str]
        Parent pool or sequence to style.
    region : RegionType, default=None
        Region to restrict styling. Can be marker name or [start, stop].
        If None, styles the entire sequence.
    style_codons : Optional[list[str]], default=None
        List of styles to apply to codons in sequence, cycling through the list.
        Mutually exclusive with style_frames.
    style_frames : Optional[list[str]], default=None
        List of styles with length a multiple of 3. Each group of 3 styles
        is applied to frames 0, 1, 2 of a codon, cycling through groups.
        Mutually exclusive with style_codons.
    frame : Optional[int], default=None
        Reading frame and orientation. Valid values: +1, +2, +3, -1, -2, -3.
        Positive values indicate left-to-right orientation (5'->3'),
        negative values indicate right-to-left orientation (3'->5').
        The absolute value indicates the frame of the boundary base (1-indexed).
        If None and region is a named OrfRegion, uses the OrfRegion's frame.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool with ORF-aware inline styling attached to sequences.

    Raises
    ------
    ValueError
        If frame is None and region is not a named OrfRegion.
    """
    from ..fixed_ops.from_seq import from_seq

    pool_obj = from_seq(pool) if isinstance(pool, str) else pool

    # Resolve frame (may look up from OrfRegion)
    resolved_frame = _resolve_frame(region, frame)

    op = StylizeOrfOp(
        pool=pool_obj,
        region=region,
        style_codons=style_codons,
        style_frames=style_frames,
        frame=resolved_frame,
        name=None,
        iter_order=iter_order,
        prefix=prefix,
    )
    return DnaPool(operation=op)


class StylizeOrfOp(Operation):
    """Apply ORF-aware inline styling to sequences."""

    factory_name = "stylize_orf"
    design_card_keys: list[str] = []

    def __init__(
        self,
        pool: Pool,
        region: RegionType = None,
        style_codons: Optional[list[str]] = None,
        style_frames: Optional[list[str]] = None,
        frame: int = 1,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """Initialize StylizeOrfOp."""
        from ..party import get_active_party

        get_active_party()  # Ensure we're in a Party context

        # Validate frame
        if frame not in VALID_FRAMES:
            raise ValueError(f"frame must be one of {sorted(VALID_FRAMES)}, got {frame}")

        # Validate mutual exclusivity
        if style_codons is not None and style_frames is not None:
            raise ValueError("style_codons and style_frames are mutually exclusive")
        if style_codons is None and style_frames is None:
            raise ValueError("Either style_codons or style_frames must be provided")

        # Validate style_frames has length that is a non-zero multiple of 3
        if style_frames is not None:
            if len(style_frames) == 0:
                raise ValueError("style_frames must not be empty")
            if len(style_frames) % 3 != 0:
                raise ValueError(
                    f"style_frames length must be a multiple of 3, got {len(style_frames)}"
                )

        # Validate style_codons is non-empty
        if style_codons is not None and len(style_codons) == 0:
            raise ValueError("style_codons must not be empty")

        self.style_codons = style_codons
        self.style_frames = style_frames
        self.frame = frame
        self.reverse = frame < 0  # Derive reverse from frame sign
        self.region_frame = abs(frame) - 1  # Convert to 0-indexed internally

        # Store region locally - we handle it ourselves, not via base class
        self._style_region = region

        super().__init__(
            parent_pools=[pool],
            num_states=1,
            mode="fixed",
            seq_length=pool.seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
        )

    def _get_tag_positions(self, text: str) -> set[int]:
        """Get positions of all characters inside XML tags."""
        from ..utils.parsing_utils import TAG_PATTERN

        tag_positions: set[int] = set()
        for match in TAG_PATTERN.finditer(text):
            for i in range(match.start(), match.end()):
                tag_positions.add(i)
        return tag_positions

    def _get_region_bounds(self, text: str) -> tuple[int, int]:
        """Get the start/end positions of the region in text."""
        if self._style_region is None:
            return (0, len(text))

        # Handle [start, stop] interval — convert molecular to literal positions
        if not isinstance(self._style_region, str):
            from ..utils.parsing_utils import nontag_pos_to_literal_pos

            start, stop = int(self._style_region[0]), int(self._style_region[1])
            return (nontag_pos_to_literal_pos(text, start), nontag_pos_to_literal_pos(text, stop))

        # Handle region name
        from ..utils.parsing_utils import find_all_regions

        try:
            regions = find_all_regions(text)
        except ValueError:
            return (0, len(text))

        for r in regions:
            if r.name == self._style_region:
                return (r.content_start, r.content_end)

        return (0, len(text))

    def _get_molecular_positions_in_region(
        self, seq: str, region_start: int, region_end: int
    ) -> np.ndarray:
        """Get molecular positions within the region bounds.

        Only includes positions with valid DNA characters (ACGTacgt).
        Tag characters and non-molecular characters (gaps, etc.) are skipped.
        """
        tag_positions = self._get_tag_positions(seq)

        positions = []
        for i in range(region_start, region_end):
            if i not in tag_positions and seq[i] in VALID_CHARS:
                positions.append(i)

        return np.array(positions, dtype=np.int64)

    def _compute_codon_styles(
        self, molecular_positions: np.ndarray
    ) -> list[tuple[str, np.ndarray]]:
        """Compute styles for codon-based styling."""
        if len(molecular_positions) == 0:
            return []

        num_styles = len(self.style_codons)
        style_positions: dict[str, list[int]] = {s: [] for s in self.style_codons}

        if self.reverse:
            # Process from end to start
            positions = molecular_positions[::-1]
        else:
            positions = molecular_positions

        # Group positions into codons and assign styles
        # Apply region_frame offset to determine codon boundaries
        for idx, pos in enumerate(positions):
            adjusted_idx = idx + self.region_frame
            codon_index = adjusted_idx // 3
            style = self.style_codons[codon_index % num_styles]
            style_positions[style].append(pos)

        # Build result list
        result = []
        for style in self.style_codons:
            if style_positions[style]:
                result.append((style, np.array(sorted(style_positions[style]), dtype=np.int64)))

        return result

    def _compute_frame_styles(
        self, molecular_positions: np.ndarray
    ) -> list[tuple[str, np.ndarray]]:
        """Compute styles for frame-based styling.

        Cycles through groups of 3 styles per codon. For example, with 6 styles:
        - Codon 0: frames 0,1,2 get styles[0,1,2]
        - Codon 1: frames 0,1,2 get styles[3,4,5]
        - Codon 2: frames 0,1,2 get styles[0,1,2] (cycles back)
        """
        if len(molecular_positions) == 0:
            return []

        num_style_groups = len(self.style_frames) // 3
        # Use set to track unique styles, dict to collect positions
        style_positions: dict[str, list[int]] = {}

        if self.reverse:
            # Process from end to start - frame is relative to reverse direction
            positions = molecular_positions[::-1]
        else:
            positions = molecular_positions

        for idx, pos in enumerate(positions):
            codon_index = idx // 3
            frame = (idx + self.region_frame) % 3
            # Select which group of 3 styles to use, cycling through groups
            style_group = codon_index % num_style_groups
            style = self.style_frames[style_group * 3 + frame]

            if style not in style_positions:
                style_positions[style] = []
            style_positions[style].append(pos)

        # Build result list (preserve order of unique styles as they appear)
        result = []
        seen_styles = set()
        for style in self.style_frames:
            if style in style_positions and style not in seen_styles:
                result.append((style, np.array(sorted(style_positions[style]), dtype=np.int64)))
                seen_styles.add(style)

        return result

    def _compute_core(
        self,
        parents: list[Seq],
        rng=None,
    ) -> tuple[Seq, dict]:
        """Return unchanged Seq with ORF-aware styling applied."""
        from ..utils.style_utils import styles_suppressed

        parent_seq = parents[0]

        # If styles suppressed, pass through unchanged
        if styles_suppressed():
            return parent_seq, {}

        # Get region bounds
        region_start, region_end = self._get_region_bounds(parent_seq.string)

        # Get molecular positions within the region
        molecular_positions = self._get_molecular_positions_in_region(
            parent_seq.string, region_start, region_end
        )

        # Compute styles based on mode
        if self.style_codons is not None:
            style_list = self._compute_codon_styles(molecular_positions)
        else:
            style_list = self._compute_frame_styles(molecular_positions)

        # Apply styles to parent Seq
        output_seq = parent_seq
        for style_spec, positions in style_list:
            if len(positions) > 0:
                output_seq = output_seq.add_style(style_spec, positions)

        return output_seq, {}

    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        params = super()._get_copy_params()
        # region parameter is stored as _style_region (non-standard naming)
        params["region"] = self._style_region
        return params
