"""Stylize operation - apply inline styling to sequences without modification."""

import re
from numbers import Real

import numpy as np

from ..operation import Operation
from ..pool import Pool
from ..types import Literal, Optional, Pool_type, RegionType, Seq, Union, beartype

# Reuse constants from style
from ..utils.style_utils import DEFAULT_GAP_CHARS

WhichType = Literal["all", "upper", "lower", "gap", "tags", "contents"]


@beartype
def stylize(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    *,
    style: str,
    which: WhichType = "contents",
    regex: Optional[str] = None,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
) -> Pool:
    """
    Apply inline styling to sequences without modifying them.

    Styles are attached directly to sequences as they flow through the pool chain.

    Parameters
    ----------
    pool : Union[Pool_type, str]
        Parent pool or sequence to style.
    region : RegionType, default=None
        Region to restrict styling. Can be marker name or [start, stop].
        If None, styles the entire sequence.
    style : str
        Style spec string (e.g., 'red bold', 'lower cyan').
        Can include 'upper'/'lower' for case transforms.
    which : WhichType, default='contents'
        Pattern selector: 'all', 'upper', 'lower', 'gap', 'tags', 'contents'.
    regex : Optional[str], default=None
        Custom regex pattern. If specified, overrides `which`.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.

    Returns
    -------
    Pool
        A Pool with inline styling attached to sequences.
    """
    from .from_seq import from_seq

    pool_obj = from_seq(pool) if isinstance(pool, str) else pool

    op = StylizeOp(
        pool=pool_obj,
        style=style,
        region=region,
        which=which,
        regex=regex,
        name=None,
        iter_order=iter_order,
        prefix=prefix,
    )
    pool_class = type(pool_obj)
    return pool_class(operation=op)


class StylizeOp(Operation):
    """Apply inline styling to sequences without modification."""

    factory_name = "stylize"
    design_card_keys: list[str] = []

    def __init__(
        self,
        pool: Pool,
        style: str,
        region: RegionType = None,
        which: WhichType = "contents",
        regex: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """Initialize StylizeOp."""
        from ..party import get_active_party

        get_active_party()  # Ensure we're in a Party context

        self.style = style
        self.which = which if regex is None else None
        self.regex = regex

        # Store region locally - we handle it ourselves, not via base class
        # (base class region handling modifies sequences, which we don't want)
        self._style_region = region

        # These patterns only apply to molecular characters (outside tags)
        self._excludes_tags = self.which in ("upper", "lower", "gap", "contents")

        # Build the internal regex pattern
        self._pattern = self._build_pattern()

        super().__init__(
            parent_pools=[pool],
            num_states=1,
            mode="fixed",
            seq_length=pool.seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            # Don't pass region - we handle it ourselves for styling only
        )

    def _build_pattern(self) -> re.Pattern:
        """Build the regex pattern based on which/regex."""
        if self.regex is not None:
            return re.compile(self.regex)

        match self.which:
            case "all" | "contents":
                return re.compile(r".")
            case "upper":
                return re.compile(r"[A-Z]")
            case "lower":
                return re.compile(r"[a-z]")
            case "gap":
                escaped = re.escape(DEFAULT_GAP_CHARS)
                return re.compile(f"[{escaped}]")
            case "tags":
                if self._style_region is None or not isinstance(self._style_region, str):
                    from ..utils.parsing_utils import TAG_PATTERN

                    return TAG_PATTERN
                else:
                    name = re.escape(self._style_region)
                    return re.compile(rf"</?{name}(?:\s[^>]*)?>|<{name}(?:\s[^>]*)?/>")
            case _:
                raise ValueError(f"Unknown 'which' value: {self.which}")

    def _get_tag_positions(self, text: str) -> set[int]:
        """Get positions of all characters inside XML tags."""
        from ..utils.parsing_utils import TAG_PATTERN

        tag_positions: set[int] = set()
        for match in TAG_PATTERN.finditer(text):
            for i in range(match.start(), match.end()):
                tag_positions.add(i)
        return tag_positions

    def _get_region_bounds(self, text: str) -> Optional[tuple[int, int]]:
        """Get the start/end positions of the region in text."""
        if self._style_region is None:
            return None

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
            return None

        for r in regions:
            if r.name == self._style_region:
                if self.which == "contents":
                    return (r.content_start, r.content_end)
                else:
                    return (r.start, r.end)

        return None

    def _get_matching_positions(self, seq: str) -> np.ndarray:
        """Get positions matching the pattern within region bounds."""
        n = len(seq)
        if n == 0:
            return np.array([], dtype=np.int64)

        # Determine bounds
        bounds = self._get_region_bounds(seq)
        if bounds is None:
            if self._style_region is not None:
                # Region specified but not found
                return np.array([], dtype=np.int64)
            eligible_start, eligible_end = 0, n
        else:
            eligible_start, eligible_end = bounds

        # Get tag positions if needed
        tag_positions = self._get_tag_positions(seq) if self._excludes_tags else set()

        # Find matching positions
        positions = []
        search_text = seq[eligible_start:eligible_end]
        for match in self._pattern.finditer(search_text):
            for i in range(match.start(), match.end()):
                pos = eligible_start + i
                if self._excludes_tags and pos in tag_positions:
                    continue
                positions.append(pos)

        return np.array(positions, dtype=np.int64)

    def _compute_core(
        self,
        parents: list[Seq],
        rng=None,
    ) -> tuple[Seq, dict]:
        """Return unchanged Seq with styling applied."""
        from ..utils.style_utils import styles_suppressed

        parent_seq = parents[0]

        # If styles suppressed, pass through unchanged
        if styles_suppressed():
            return parent_seq, {}

        # Get positions matching the pattern
        positions = self._get_matching_positions(parent_seq.string)

        # Add new style to parent Seq
        if len(positions) > 0:
            output_seq = parent_seq.add_style(self.style, positions)
        else:
            output_seq = parent_seq

        return output_seq, {}

    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        params = super()._get_copy_params()
        # region parameter is stored as _style_region (non-standard naming)
        params["region"] = self._style_region
        return params
