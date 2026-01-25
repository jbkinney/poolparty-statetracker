"""Stylize operation - apply inline styling to sequences without modification."""
import re
from numbers import Real
import numpy as np
from ..types import Pool_type, Union, Optional, RegionType, Literal, beartype, StyleList
from ..operation import Operation
from ..pool import Pool

# Reuse constants from style
from ..utils.style_utils import DEFAULT_GAP_CHARS

WhichType = Literal['all', 'upper', 'lower', 'gap', 'tags', 'contents']


@beartype
def stylize(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    *,
    style: str,
    which: WhichType = 'contents',
    regex: Optional[str] = None,
    iter_order: Optional[Real] = None,
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
    )
    return Pool(operation=op)


@beartype
class StylizeOp(Operation):
    """Apply inline styling to sequences without modification."""
    factory_name = "stylize"
    design_card_keys: list[str] = []

    def __init__(
        self,
        pool: Pool,
        style: str,
        region: RegionType = None,
        which: WhichType = 'contents',
        regex: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
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
        self._excludes_tags = self.which in ('upper', 'lower', 'gap', 'contents')

        # Build the internal regex pattern
        self._pattern = self._build_pattern()

        super().__init__(
            parent_pools=[pool],
            num_values=1,
            mode='fixed',
            seq_length=pool.seq_length,
            name=name,
            iter_order=iter_order,
            # Don't pass region - we handle it ourselves for styling only
        )

    def _build_pattern(self) -> re.Pattern:
        """Build the regex pattern based on which/regex."""
        if self.regex is not None:
            return re.compile(self.regex)

        match self.which:
            case 'all' | 'contents':
                return re.compile(r'.')
            case 'upper':
                return re.compile(r'[A-Z]')
            case 'lower':
                return re.compile(r'[a-z]')
            case 'gap':
                escaped = re.escape(DEFAULT_GAP_CHARS)
                return re.compile(f'[{escaped}]')
            case 'tags':
                if self._style_region is None or not isinstance(self._style_region, str):
                    from ..marker_ops.parsing import TAG_PATTERN
                    return TAG_PATTERN
                else:
                    name = re.escape(self._style_region)
                    return re.compile(rf'</?{name}(?:\s[^>]*)?>|<{name}(?:\s[^>]*)?/>')
            case _:
                raise ValueError(f"Unknown 'which' value: {self.which}")

    def _get_tag_positions(self, text: str) -> set[int]:
        """Get positions of all characters inside XML tags."""
        from ..marker_ops.parsing import TAG_PATTERN
        tag_positions: set[int] = set()
        for match in TAG_PATTERN.finditer(text):
            for i in range(match.start(), match.end()):
                tag_positions.add(i)
        return tag_positions

    def _get_region_bounds(self, text: str) -> Optional[tuple[int, int]]:
        """Get the start/end positions of the region in text."""
        if self._style_region is None:
            return None

        # Handle [start, stop] interval
        if not isinstance(self._style_region, str):
            start, stop = int(self._style_region[0]), int(self._style_region[1])
            return (start, stop)

        # Handle marker name
        from ..marker_ops.parsing import find_all_markers

        try:
            markers = find_all_markers(text)
        except ValueError:
            return None

        for m in markers:
            if m.name == self._style_region:
                if self.which == 'contents':
                    return (m.content_start, m.content_end)
                else:
                    return (m.start, m.end)

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

    def compute(
        self,
        parent_seqs: list[str],
        rng=None,
        parent_styles: list[StyleList] | None = None,
    ) -> dict:
        """Return unchanged sequence with styling applied."""
        seq = parent_seqs[0]

        # Get positions matching the pattern
        positions = self._get_matching_positions(seq)

        # Build output styles: pass through parent + add new style
        output_styles: StyleList = []
        if parent_styles and len(parent_styles) > 0:
            output_styles.extend(parent_styles[0])

        if len(positions) > 0:
            output_styles.append((self.style, positions))

        return {'seq': seq, 'style': output_styles}

    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'pool': self.parent_pools[0],
            'style': self.style,
            'region': self._style_region,
            'which': self.which,
            'regex': self.regex,
            'name': None,
            'iter_order': self.iter_order,
        }
