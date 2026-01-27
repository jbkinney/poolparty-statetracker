"""Recombine operation - simulate evolutionary recombination across aligned sequences."""
from itertools import combinations
from math import comb
from ..types import Union, ModeType, Optional, Real, Integral, Sequence, RegionType, beartype, Seq
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def recombine(
    pool: Union[Pool, str, None] = None,
    region: RegionType = None,
    source_pools: Sequence[Union[Pool, str]] = (),
    num_breakpoints: Integral = 1,
    positions: Optional[Sequence[Integral]] = None,
    mode: ModeType = 'random',
    num_states: Optional[int] = None,
    prefix: Optional[str] = None,
    styles: Optional[list[str]] = None,
    iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = 'recombine',
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
    source_pools : Sequence[Union[Pool, str]], default=()
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
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    styles : Optional[list[str]], default=None
        List of styles to apply to each segment (length must equal num_breakpoints + 1).
        Use empty string '' for segments that shouldn't have additional styling.
        Styles overlay on top of inherited source pool styles.
    iter_order : Optional[Real], default=None
        Iteration order for the Operation.

    Returns
    -------
    Pool
        A Pool that generates recombined sequences.
    """
    from ..fixed_ops.from_seq import from_seq
    
    # Convert string source_pools to Pool objects
    converted_source_pools = []
    for i, sp in enumerate(source_pools):
        if isinstance(sp, str):
            converted_source_pools.append(from_seq(sp, _factory_name=f'{_factory_name}(from_seq_{i})'))
        else:
            converted_source_pools.append(sp)
    source_pools = converted_source_pools
    
    # Validate source_pools has at least 2 pools
    if len(source_pools) < 2:
        raise ValueError("source_pools must contain at least 2 pools for recombination")
    
    # Validate all source pools have the same fixed seq_length
    seq_lengths = [sp.seq_length for sp in source_pools]
    if any(sl is None for sl in seq_lengths):
        raise ValueError("All source_pools must have a fixed seq_length (not None)")
    if len(set(seq_lengths)) > 1:
        raise ValueError(
            f"All source_pools must have the same seq_length. "
            f"Found lengths: {seq_lengths}"
        )
    
    # Create the operation
    op = RecombineOp(
        parent_pool=pool,
        source_pools=source_pools,
        num_breakpoints=num_breakpoints,
        positions=positions,
        region=region,
        styles=styles,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        _factory_name=_factory_name,
    )
    
    result_pool = Pool(operation=op)
    return result_pool


@beartype
class RecombineOp(Operation):
    """Recombine segments from multiple source pools at specified breakpoints.
    
    In sequential mode, enumerates all breakpoint positions × pool assignment combinations.
    In random mode, randomly selects breakpoints and pool assignments.
    """
    factory_name = "recombine"
    design_card_keys = ['breakpoints', 'pool_assignments']
    
    def __init__(
        self,
        parent_pool: Optional[Pool],
        source_pools: Sequence[Pool],
        num_breakpoints: Integral = 1,
        positions: Optional[Sequence[Integral]] = None,
        region: RegionType = None,
        styles: Optional[list[str]] = None,
        prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        _factory_name: Optional[str] = 'recombine',
    ) -> None:
        # Set factory name if provided
        if _factory_name is not None:
            self.factory_name = _factory_name
        
        # Store source pools and validate
        self.source_pools = list(source_pools)
        self.num_source_pools = len(self.source_pools)
        
        if self.num_source_pools < 2:
            raise ValueError("source_pools must contain at least 2 pools for recombination")
        
        # Get and validate seq_length
        seq_length = self.source_pools[0].seq_length
        if seq_length is None:
            raise ValueError("All source_pools must have a fixed seq_length (not None)")
        
        for i, sp in enumerate(self.source_pools):
            if sp.seq_length != seq_length:
                raise ValueError(
                    f"All source_pools must have the same seq_length. "
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
        if styles is not None:
            num_segments = self.num_breakpoints + 1
            if len(styles) != num_segments:
                raise ValueError(
                    f"styles must have length {num_segments} (num_breakpoints + 1), "
                    f"got {len(styles)}"
                )
        
        self._mode = mode
        self._sequential_cache = None
        
        # Determine parent_pools for Operation base class
        if parent_pool is not None:
            # Region-based: parent_pool + source_pools
            parent_pools = [parent_pool] + self.source_pools
        else:
            # Direct recombination: only source_pools
            parent_pools = self.source_pools
        
        # Determine num_states based on mode
        if mode == 'sequential':
            # Total states = C(P, K) × N × (N-1)^K
            # where P = number of valid positions, K = num_breakpoints, N = num_source_pools
            # First segment has N choices, each subsequent segment has N-1 choices
            # (can't use same pool as previous segment)
            num_breakpoint_combos = comb(len(self.positions), self.num_breakpoints)
            num_pool_assignments = self.num_source_pools * ((self.num_source_pools - 1) ** self.num_breakpoints)
            num_states = num_breakpoint_combos * num_pool_assignments
            
            # Build cache for sequential enumeration
            self._build_cache()
        elif mode == 'random':
            # num_states stays as provided (or None for pure random)
            pass
        else:
            # fixed mode
            num_states = 1
        
        super().__init__(
            parent_pools=parent_pools,
            num_states=num_states,
            mode=mode,
            seq_length=self._seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=region,
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
        N = self.num_source_pools
        
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
    
    def compute(
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
        if self.mode == 'sequential':
            # Use cached combinations
            state_val = self.state.value if self.state.value is not None else 0
            breakpoints, pool_assignments = self._sequential_cache[state_val % len(self._sequential_cache)]
        elif self.mode == 'random':
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)")
            
            # Randomly select breakpoints (sorted)
            breakpoint_indices = rng.choice(
                len(self.positions),
                size=self.num_breakpoints,
                replace=False
            )
            breakpoints = tuple(sorted([self.positions[i] for i in breakpoint_indices]))
            
            # Randomly assign pools to segments (consecutive segments must differ)
            num_segments = self.num_breakpoints + 1
            pool_assignments = []
            for i in range(num_segments):
                if i == 0:
                    # First segment: any pool
                    pool_assignments.append(int(rng.integers(0, self.num_source_pools)))
                else:
                    # Subsequent segments: any pool except the previous one
                    prev_pool = pool_assignments[-1]
                    choices = [p for p in range(self.num_source_pools) if p != prev_pool]
                    pool_assignments.append(choices[int(rng.integers(0, len(choices)))])
            pool_assignments = tuple(pool_assignments)
        else:
            # fixed mode - use first positions and alternating pool assignment
            breakpoints = tuple(sorted(self.positions[:self.num_breakpoints]))
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
            if source_styles and pool_idx < len(source_styles):
                seg_style = source_styles[pool_idx][start:end]
            else:
                seg_style = SeqStyle.empty(len(segment))
            segment_styles.append(seg_style)
            
            start = end
        
        # Last segment (from last breakpoint to end)
        last_pool_idx = pool_assignments[-1]
        segment = source_seqs[last_pool_idx][start:]
        segments.append(segment)
        
        if source_styles and last_pool_idx < len(source_styles):
            seg_style = source_styles[last_pool_idx][start:]
        else:
            seg_style = SeqStyle.empty(len(segment))
        segment_styles.append(seg_style)
        
        # Build segments as Seq objects
        seq_segments = []
        for seg, seg_style in zip(segments, segment_styles):
            seq_segments.append(Seq(seg, seg_style, None))
        
        # Join segments
        output_seq = Seq.join(seq_segments)
        
        # Overlay additional styles if provided
        if self._styles is not None:
            offset = 0
            for seg_idx, style_spec in enumerate(self._styles):
                if style_spec and style_spec != '':
                    # Apply style to this segment
                    seg_len = len(seq_segments[seg_idx])
                    positions = np.arange(offset, offset + seg_len, dtype=np.int64)
                    output_seq = output_seq.add_style(style_spec, positions)
                offset += len(seq_segments[seg_idx])
        
        # Compute name
        output_seq = output_seq.with_name(self._default_name(parents))
        
        return output_seq, {
            'breakpoints': breakpoints,
            'pool_assignments': pool_assignments,
        }
