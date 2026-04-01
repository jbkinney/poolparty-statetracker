"""Recombine operation - simulate evolutionary recombination across aligned sequences."""

from itertools import combinations
from math import comb

import numpy as np

from ..dna_pool import DnaPool
from ..operation import Operation
from ..pool import Pool
from ..types import (
    CardsType,
    Integral,
    ModeType,
    Optional,
    Real,
    RegionType,
    Seq,
    Sequence,
    StyleByForRecombineType,
    Union,
    beartype,
)
from ..utils.dna_seq import DnaSeq


@beartype
def recombine(
    pool: Union[Pool, str, None] = None,
    region: RegionType = None,
    sources: Sequence[Union[Pool, str]] = (),
    num_breakpoints: Integral = 1,
    positions: Optional[Sequence[Integral]] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    prefix: Optional[str] = None,
    styles: Optional[list[str]] = None,
    style_by: StyleByForRecombineType = "order",
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = "recombine",
) -> Pool:
    """
    Create a Pool that recombines segments from multiple source pools at breakpoints.

    Parameters
    ----------
    pool : Union[Pool, str, None], default=None
        Parent pool for region-based recombination. If provided with region,
        the recombined sequences replace the region content.
    region : Union[str, Sequence[Integral], None], default=None
        Region in pool where recombined sequences will be inserted. Region content
        is discarded (not used as a source pool).
    sources : Sequence[Union[Pool, str]], default=()
        Source pools for recombination. All must have the same seq_length.
    num_breakpoints : Integral, default=1
        Number of recombination breakpoints. Must be <= seq_length - 1.
    positions : Optional[Sequence[Integral]], default=None
        Valid breakpoint positions. If None, defaults to range(seq_length - 1).
        Position i means "breakpoint after index i".
    mode : ModeType, default='random'
        Selection mode: 'random' (random breakpoints and pool assignments) or
        'sequential' (enumerate all combinations).
    num_states : Optional[int], default=None
        Number of states. In sequential mode, overrides the computed count
        (cycling if greater, clipping if less). In random mode, if None
        defaults to 1 (pure random sampling).
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    styles : Optional[list[str]], default=None
        List of styles to apply to segments. Both modes accept any non-empty list and cycle:
        - If style_by='order': cycles through styles for segments by position
          (e.g., with 2 styles and 5 segments: style[0], style[1], style[0], style[1], style[0])
        - If style_by='source': cycles through styles based on source pool index
          (e.g., with 2 styles and 3 sources: source[0]→style[0], source[1]→style[1], source[2]→style[0])
        Use empty string '' for segments that shouldn't have additional styling.
        Styles overlay on top of inherited source pool styles.
    style_by : StyleByForRecombineType, default='order'
        Determines how styles are assigned to segments:
        - 'order': styles[i % len(styles)] applied to segment i (cycles through styles by position)
        - 'source': styles[j % len(styles)] applied to segments from sources[j] (cycles by source index)
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    cards : list[str] or dict, optional
        Design card keys to include. Available keys: ``'breakpoints'``,
        ``'pool_assignments'``.

    Returns
    -------
    Pool
        A Pool that generates recombined sequences.
    """
    from ..fixed_ops.from_seq import from_seq

    # Convert string sources to Pool objects
    converted_sources = []
    for i, sp in enumerate(sources):
        if isinstance(sp, str):
            converted_sources.append(from_seq(sp, _factory_name=f"{_factory_name}(from_seq_{i})"))
        else:
            converted_sources.append(sp)
    sources = converted_sources

    # Validate sources has at least 2 pools
    if len(sources) < 2:
        raise ValueError("sources must contain at least 2 pools for recombination")

    # Validate all source pools have the same fixed seq_length
    seq_lengths = [sp.seq_length for sp in sources]
    if any(sl is None for sl in seq_lengths):
        raise ValueError("All sources must have a fixed seq_length (not None)")
    if len(set(seq_lengths)) > 1:
        raise ValueError(f"All sources must have the same seq_length. Found lengths: {seq_lengths}")

    # Create the operation
    op = RecombineOp(
        parent_pool=pool,
        sources=sources,
        num_breakpoints=num_breakpoints,
        positions=positions,
        region=region,
        styles=styles,
        style_by=style_by,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        cards=cards,
        _factory_name=_factory_name,
    )

    # Preserve the pool type from the first source
    pool_class = type(sources[0]) if sources else DnaPool
    result_pool = pool_class(operation=op)
    return result_pool


class RecombineOp(Operation):
    """Recombine segments from multiple source pools at specified breakpoints.

    In sequential mode, enumerates all breakpoint positions × pool assignment combinations.
    In random mode, randomly selects breakpoints and pool assignments.
    """

    factory_name = "recombine"
    design_card_keys = ["breakpoints", "pool_assignments"]

    def __init__(
        self,
        parent_pool: Optional[Pool],
        sources: Sequence[Pool],
        num_breakpoints: Integral = 1,
        positions: Optional[Sequence[Integral]] = None,
        region: RegionType = None,
        styles: Optional[list[str]] = None,
        style_by: StyleByForRecombineType = "order",
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
        _factory_name: Optional[str] = "recombine",
    ) -> None:
        # Set factory name if provided
        if _factory_name is not None:
            self.factory_name = _factory_name

        # Store source pools and validate
        self.sources = list(sources)
        self.num_sources = len(self.sources)

        if self.num_sources < 2:
            raise ValueError("sources must contain at least 2 pools for recombination")

        # Get and validate seq_length
        seq_length = self.sources[0].seq_length
        if seq_length is None:
            raise ValueError("All sources must have a fixed seq_length (not None)")

        for i, sp in enumerate(self.sources):
            if sp.seq_length != seq_length:
                raise ValueError(
                    f"All sources must have the same seq_length. "
                    f"Pool {i} has length {sp.seq_length}, expected {seq_length}"
                )

        self._seq_length = seq_length
        self.num_breakpoints = int(num_breakpoints)

        # Validate num_breakpoints
        if self.num_breakpoints < 1:
            raise ValueError(f"num_breakpoints must be >= 1, got {self.num_breakpoints}")
        if self.num_breakpoints > seq_length - 1:
            raise ValueError(
                f"num_breakpoints={self.num_breakpoints} exceeds seq_length - 1 = {seq_length - 1}. "
                f"Each segment must have at least 1 nucleotide."
            )

        # Set up positions
        if positions is None:
            self.positions = list(range(seq_length - 1))  # [0, 1, ..., L-2]
        else:
            self.positions = list(positions)
            # Validate positions
            for pos in self.positions:
                if pos < 0 or pos >= seq_length - 1:
                    raise ValueError(
                        f"Invalid position {pos}. Positions must be in range [0, {seq_length - 1})"
                    )

        if len(self.positions) < self.num_breakpoints:
            raise ValueError(
                f"Not enough positions ({len(self.positions)}) for num_breakpoints={self.num_breakpoints}"
            )

        # Validate and store styles
        self._styles = styles
        self._style_by = style_by
        if styles is not None:
            # Both modes accept any non-empty list (will cycle through)
            if len(styles) == 0:
                raise ValueError("styles must be non-empty, got empty list")

        self._mode = mode
        self._sequential_cache = None

        # Determine parent_pools for Operation base class
        if parent_pool is not None:
            # Region-based: parent_pool + sources
            parent_pools = [parent_pool] + self.sources
        else:
            # Direct recombination: only sources
            parent_pools = self.sources

        # Determine num_states based on mode
        natural_num_states = None
        if mode == "sequential":
            # Total states = C(P, K) × N × (N-1)^K
            # where P = number of valid positions, K = num_breakpoints, N = num_sources
            # First segment has N choices, each subsequent segment has N-1 choices
            # (can't use same pool as previous segment)
            num_breakpoint_combos = comb(len(self.positions), self.num_breakpoints)
            num_pool_assignments = self.num_sources * (
                (self.num_sources - 1) ** self.num_breakpoints
            )
            natural_num_states = num_breakpoint_combos * num_pool_assignments

            # Build cache for sequential enumeration
            self._build_cache()

            # Use user-provided num_states if given, else natural count
            if num_states is None:
                num_states = natural_num_states
        elif mode == "random":
            # num_states stays as provided (or None for pure random)
            pass
        else:
            # fixed mode
            num_states = 1

        if parent_pool is None:
            output_seq_length = self._seq_length
        else:
            parent_seq_length = parent_pool.seq_length
            if isinstance(region, str):
                from ..party import get_active_party

                party = get_active_party()
                try:
                    region_obj = party.get_region_by_name(region)
                    region_length = region_obj.seq_length
                except ValueError:
                    region_length = None
            else:
                region_length = region[1] - region[0] if region is not None else None

            if parent_seq_length is None or region_length is None:
                output_seq_length = None
            else:
                output_seq_length = parent_seq_length - region_length + self._seq_length

        super().__init__(
            parent_pools=parent_pools,
            num_states=num_states,
            mode=mode,
            seq_length=output_seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=region,
            _natural_num_states=natural_num_states,
            cards=cards,
        )

    def _build_cache(self) -> None:
        """Build cache for sequential enumeration of breakpoint positions and pool assignments.

        Consecutive segments must come from different pools (no self-recombination).
        First segment: N choices, subsequent segments: N-1 choices each.
        """
        cache = []

        # Enumerate all breakpoint combinations
        for breakpoint_combo in combinations(self.positions, self.num_breakpoints):
            breakpoint_combo = tuple(sorted(breakpoint_combo))

            # Generate all valid pool assignments where consecutive segments differ
            pool_assignments_list = self._enumerate_pool_assignments()

            for pool_assignments in pool_assignments_list:
                cache.append((breakpoint_combo, pool_assignments))

        self._sequential_cache = cache

    def _enumerate_pool_assignments(self) -> list[tuple[int, ...]]:
        """Enumerate all valid pool assignments for segments.

        Consecutive segments must come from different pools.
        Returns list of tuples, each tuple is a valid assignment.
        """
        num_segments = self.num_breakpoints + 1
        N = self.num_sources

        # Build assignments recursively
        def build(current: list[int]) -> list[tuple[int, ...]]:
            if len(current) == num_segments:
                return [tuple(current)]

            results = []
            if len(current) == 0:
                # First segment: any pool
                for pool_idx in range(N):
                    results.extend(build(current + [pool_idx]))
            else:
                # Subsequent segments: any pool except the previous one
                prev_pool = current[-1]
                for pool_idx in range(N):
                    if pool_idx != prev_pool:
                        results.extend(build(current + [pool_idx]))
            return results

        return build([])

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Generate recombined Seq.

        When region is specified:
        - parents[0] is the region content (which we ignore)
        - parents[1:] are the source pool sequences

        When region is not specified:
        - parents are the source pool sequences directly
        """
        from ..utils.style_utils import SeqStyle, styles_suppressed

        # Cache party attributes at start (avoid repeated function calls)
        _suppress_styles = self._party.suppress_styles
        _suppress_cards = self._party.suppress_cards

        # Determine which parents are source sequences
        if self._region is not None:
            # Region-based: skip first parent (region content)
            sources = parents[1:]
        else:
            # Direct: all parents are source sequences
            sources = parents

        # Extract strings and styles for compatibility with segment logic
        source_seqs = [s.string for s in sources]
        source_styles = [s.style for s in sources]

        # Get breakpoints and pool assignments
        if self.mode == "sequential":
            # Use cached combinations
            state_val = self.state.value if self.state.value is not None else 0
            breakpoints, pool_assignments = self._sequential_cache[
                state_val % len(self._sequential_cache)
            ]
        elif self.mode == "random":
            if rng is None:
                raise RuntimeError(
                    f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)"
                )

            # Randomly select breakpoints (sorted)
            breakpoint_indices = rng.choice(
                len(self.positions), size=self.num_breakpoints, replace=False
            )
            breakpoints = tuple(sorted([self.positions[i] for i in breakpoint_indices]))

            # Randomly assign pools to segments (consecutive segments must differ)
            num_segments = self.num_breakpoints + 1
            pool_assignments = []
            for i in range(num_segments):
                if i == 0:
                    # First segment: any pool
                    pool_assignments.append(int(rng.integers(0, self.num_sources)))
                else:
                    # Subsequent segments: any pool except the previous one
                    prev_pool = pool_assignments[-1]
                    choices = [p for p in range(self.num_sources) if p != prev_pool]
                    pool_assignments.append(choices[int(rng.integers(0, len(choices)))])
            pool_assignments = tuple(pool_assignments)
        else:
            # fixed mode - use first positions and alternating pool assignment
            breakpoints = tuple(sorted(self.positions[: self.num_breakpoints]))
            num_segments = self.num_breakpoints + 1
            # Alternate between pool 0 and pool 1 (guarantees consecutive segments differ)
            pool_assignments = tuple(i % 2 for i in range(num_segments))

        # Build recombined sequence from segments
        segments = []
        segment_styles = []

        # Breakpoints define segment boundaries
        # Breakpoint at position i means "after index i"
        # So segment ranges are: [0:b0+1], [b0+1:b1+1], ..., [bK+1:L]
        start = 0
        for seg_idx, (breakpoint, pool_idx) in enumerate(zip(breakpoints, pool_assignments)):
            end = breakpoint + 1
            # Extract segment from assigned source pool
            segment = source_seqs[pool_idx][start:end]
            segments.append(segment)

            # Extract and offset style from source pool
            if (
                source_styles
                and pool_idx < len(source_styles)
                and source_styles[pool_idx] is not None
            ):
                seg_style = source_styles[pool_idx][start:end]
            else:
                seg_style = None if _suppress_styles else SeqStyle.empty(len(segment))
            segment_styles.append(seg_style)

            start = end

        # Last segment (from last breakpoint to end)
        last_pool_idx = pool_assignments[-1]
        segment = source_seqs[last_pool_idx][start:]
        segments.append(segment)

        if (
            source_styles
            and last_pool_idx < len(source_styles)
            and source_styles[last_pool_idx] is not None
        ):
            seg_style = source_styles[last_pool_idx][start:]
        else:
            seg_style = None if _suppress_styles else SeqStyle.empty(len(segment))
        segment_styles.append(seg_style)

        # Build segments as DnaSeq objects
        seq_segments = []
        for seg, seg_style in zip(segments, segment_styles):
            seq_segments.append(DnaSeq(seg, seg_style))

        # Join segments (use fast path since segments are tag-free)
        output_seq = DnaSeq._join_fast(seq_segments)

        # Overlay additional styles if provided
        if self._styles is not None:
            offset = 0
            for seg_idx in range(len(seq_segments)):
                # Determine which style to apply based on style_by mode
                if self._style_by == "order":
                    # Apply style based on segment position (cycle through styles)
                    style_spec = self._styles[seg_idx % len(self._styles)]
                else:  # style_by == 'source'
                    # Apply style based on source pool (cycle through styles)
                    pool_idx = pool_assignments[seg_idx]
                    style_spec = self._styles[pool_idx % len(self._styles)]

                if style_spec and style_spec != "":
                    # Apply style to this segment
                    seg_len = len(seq_segments[seg_idx])
                    positions = np.arange(offset, offset + seg_len, dtype=np.int64)
                    output_seq = output_seq.add_style(style_spec, positions)
                offset += len(seq_segments[seg_idx])

        if _suppress_cards:
            return output_seq, {}

        return output_seq, {
            "breakpoints": breakpoints,
            "pool_assignments": pool_assignments,
        }
