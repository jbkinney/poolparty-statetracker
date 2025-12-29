"""SpacingScanPool: Enumerate spacing combinations for multiple inserts relative to an anchor."""

import itertools
import warnings
from typing import Union, List, Tuple, Dict, Any, Literal
from .pool import Pool


class SpacingScanPool(Pool):
    """Enumerate spacing combinations for multiple inserts relative to an anchor position.
    
    Pre-computes all valid (non-overlapping, properly ordered) combinations at construction,
    giving a finite state space for sequential enumeration. Designed for testing regulatory
    grammar with controlled transcription factor spacing.
    
    Supports two mutually exclusive interfaces for specifying insert positions:
    
    **Range-based interface (insert_scan_ranges):**
        Defines a region where each insert can be placed. The region boundaries specify
        where the insert's left and right edges can reach.
        
        - `(start, end)`: Insert can be placed anywhere in region [anchor+start, anchor+end],
          with the left edge at leftmost position and right edge at rightmost position.
          Step defaults to 1.
        - `(start, end, step)`: Same as above, but positions are sampled at given step.
        
        The valid left-edge positions are automatically computed accounting for insert length.
    
    **Distance-based interface (insert_distances):**
        Specifies explicit left-edge positions (distances from anchor) for each insert.
        
        - `[d1, d2, d3, ...]`: Insert's left edge can be at anchor+d1, anchor+d2, etc.
    
    Both interfaces can be mixed (some inserts use ranges, others use explicit distances).
    
    Example:
        # Test SP1 and AP1 spacing relative to TSS
        pool = SpacingScanPool(
            background_seq="N" * 150,
            insert_seqs=[sp1_motif, ap1_motif],
            insert_names=["SP1", "AP1"],
            anchor_pos=100,
            insert_scan_ranges=[
                (-80, -20, 10),  # SP1: region [-80, -20] from anchor, step 10
                (10, 60, 10),    # AP1: region [+10, +60] from anchor, step 10
            ],
            min_spacing=5,
            mode='sequential',
        )
        seqs = pool.generate_seqs(num_complete_iterations=1)
    """
    
    def __init__(
        self,
        background_seq: Union[str, Pool],
        insert_seqs: List[Union[str, Pool]],
        anchor_pos: int,
        insert_scan_ranges: List[Union[Tuple[int, int], Tuple[int, int, int], None]] = None,
        insert_distances: List[Union[List[int], None]] = None,
        insert_names: List[str] = None,
        min_spacing: int = 0,
        enforce_order: bool = True,
        insert_or_overwrite: Literal['insert', 'overwrite'] = 'overwrite',
        mark_changes: bool = False,
        mode: str = 'sequential',
        max_num_states: int = None,
        iteration_order: int = None,
        name: str = None,
        metadata: str = 'features',
    ):
        """Initialize a SpacingScanPool.
        
        Args:
            background_seq: Background sequence (str or Pool) to insert into.
            insert_seqs: List of sequences (str or Pool) to insert. Order matters
                if enforce_order=True.
            anchor_pos: Reference position (0-indexed) in background for distance
                calculations. All distances are relative to this position.
            insert_scan_ranges: For each insert, a region where it can be placed:
                - (start, end): Region [anchor+start, anchor+end], step=1
                - (start, end, step): Same with explicit step
                - None: Use insert_distances for this insert instead
                Must have same length as insert_seqs.
            insert_distances: For each insert, explicit left-edge distances from anchor:
                - [d1, d2, ...]: Left edge at anchor+d1, anchor+d2, etc.
                - None: Use insert_scan_ranges for this insert instead
                Must have same length as insert_seqs.
            insert_names: Names for each insert (for design card columns).
                Defaults to ["insert_0", "insert_1", ...].
            min_spacing: Minimum gap (bp) between adjacent inserts. Default 0
                (non-overlapping). Must be >= 0.
            enforce_order: If True (default), inserts must appear in list order
                (5' to 3'). Combinations where order would flip are filtered out.
            insert_or_overwrite: 'overwrite' (default) replaces background;
                'insert' shifts the sequence.
            mark_changes: If True, apply swapcase() to inserted regions.
            mode: 'sequential' (default) or 'random'.
            max_num_states: Maximum states before treating as infinite.
            iteration_order: Order for sequential iteration in composites.
            name: Pool name for design cards.
            metadata: Metadata level ('core', 'features', 'complete').
        
        Raises:
            ValueError: If insert_seqs is empty.
            ValueError: If insert_scan_ranges and insert_distances have wrong length.
            ValueError: If an insert has neither scan_range nor distances specified.
            ValueError: If an insert has both scan_range and distances specified.
            ValueError: If min_spacing < 0.
            ValueError: If no valid combinations exist after filtering.
            ValueError: If anchor_pos is out of background bounds.
        """
        # Store raw inputs
        self.background_seq = background_seq
        self.insert_seqs = insert_seqs
        self.anchor_pos = anchor_pos
        self.min_spacing = min_spacing
        self.enforce_order = enforce_order
        self.insert_or_overwrite = insert_or_overwrite
        self.mark_changes = mark_changes
        
        # Validate basic inputs
        if len(insert_seqs) == 0:
            raise ValueError("insert_seqs cannot be empty")
        
        if min_spacing < 0:
            raise ValueError(f"min_spacing must be >= 0, got {min_spacing}")
        
        if insert_or_overwrite not in ['insert', 'overwrite']:
            raise ValueError(
                f"insert_or_overwrite must be 'insert' or 'overwrite', "
                f"got '{insert_or_overwrite}'"
            )
        
        n_inserts = len(insert_seqs)
        
        # Normalize insert_scan_ranges and insert_distances to lists of correct length
        if insert_scan_ranges is None:
            insert_scan_ranges = [None] * n_inserts
        if insert_distances is None:
            insert_distances = [None] * n_inserts
        
        if len(insert_scan_ranges) != n_inserts:
            raise ValueError(
                f"insert_scan_ranges length ({len(insert_scan_ranges)}) must match "
                f"insert_seqs length ({n_inserts})"
            )
        if len(insert_distances) != n_inserts:
            raise ValueError(
                f"insert_distances length ({len(insert_distances)}) must match "
                f"insert_seqs length ({n_inserts})"
            )
        
        # Set up insert names
        if insert_names is None:
            self._insert_names = [f"insert_{i}" for i in range(n_inserts)]
        else:
            if len(insert_names) != n_inserts:
                raise ValueError(
                    f"insert_names length ({len(insert_names)}) must match "
                    f"insert_seqs length ({n_inserts})"
                )
            # Check for duplicate names
            if len(set(insert_names)) != len(insert_names):
                duplicates = [n for n in insert_names if insert_names.count(n) > 1]
                raise ValueError(
                    f"insert_names must be unique. Found duplicates: {set(duplicates)}"
                )
            self._insert_names = list(insert_names)
        
        # Get insert lengths
        self._insert_lengths = []
        for i, ins in enumerate(insert_seqs):
            if isinstance(ins, Pool):
                length = ins.seq_length
            else:
                length = len(ins)
            if length == 0:
                raise ValueError(f"Insert {i} ({self._insert_names[i]}) has zero length")
            self._insert_lengths.append(length)
        
        # Get background length
        if isinstance(background_seq, Pool):
            self._bg_length = background_seq.seq_length
        else:
            self._bg_length = len(background_seq)
        
        # Validate anchor
        if anchor_pos < 0 or anchor_pos >= self._bg_length:
            raise ValueError(
                f"anchor_pos ({anchor_pos}) must be in range [0, {self._bg_length})"
            )
        
        # Convert specifications to distance lists for each insert
        self._distance_lists = []
        for i in range(n_inserts):
            scan_range = insert_scan_ranges[i]
            distances = insert_distances[i]
            
            if scan_range is not None and distances is not None:
                raise ValueError(
                    f"Insert {i} ({self._insert_names[i]}) has both scan_range and "
                    f"distances specified. Choose one."
                )
            
            if scan_range is None and distances is None:
                raise ValueError(
                    f"Insert {i} ({self._insert_names[i]}) has neither scan_range nor "
                    f"distances specified. Must provide one."
                )
            
            if scan_range is not None:
                # Range-based: (start, end) or (start, end, step)
                dist_list = self._scan_range_to_distances(scan_range, i)
            else:
                # Distance-based: explicit list
                if len(distances) == 0:
                    raise ValueError(
                        f"Insert {i} ({self._insert_names[i]}) has empty distances list"
                    )
                dist_list = list(distances)
            
            self._distance_lists.append(dist_list)
        
        # Collect parents for computation graph
        parents = []
        if isinstance(background_seq, Pool):
            parents.append(background_seq)
        for ins in insert_seqs:
            if isinstance(ins, Pool):
                parents.append(ins)
        
        # Early check: estimate max combinations before full enumeration
        max_combos = 1
        for dist_list in self._distance_lists:
            max_combos *= len(dist_list)
        
        if max_combos > 10_000_000:
            raise ValueError(
                f"Cartesian product too large ({max_combos:,} potential combinations). "
                "This would consume too much memory. Consider: reducing distance ranges, "
                "using fewer inserts, or using explicit insert_distances with fewer positions."
            )
        
        # Pre-compute all valid combinations
        self._valid_combinations = self._enumerate_valid_combinations()
        
        if len(self._valid_combinations) == 0:
            raise ValueError(
                "No valid distance combinations found. Check for: "
                "overlaps, ordering violations (if enforce_order=True), "
                "or out-of-bounds positions."
            )
        
        if len(self._valid_combinations) > 1_000_000:
            raise ValueError(
                f"State space too large ({len(self._valid_combinations):,} combinations). "
                "Consider: reducing distance ranges, using fewer inserts, "
                "or using explicit insert_distances with fewer positions."
            )
        
        if len(self._valid_combinations) > 100_000:
            warnings.warn(
                f"Large state space ({len(self._valid_combinations):,} combinations). "
                "Consider reducing distance ranges for faster enumeration."
            )
        
        super().__init__(
            parents=tuple(parents) if parents else (),
            op='spacing_scan',
            max_num_states=max_num_states,
            mode=mode,
            iteration_order=iteration_order,
            name=name,
            metadata=metadata,
        )
        
        # Cache for design cards
        self._cached_combo: Tuple[int, ...] = None
        self._cached_output_positions: Dict[int, Tuple[int, int]] = {}
    
    def _scan_range_to_distances(
        self, 
        scan_range: Union[Tuple[int, int], Tuple[int, int, int]], 
        insert_idx: int
    ) -> List[int]:
        """Convert a scan range to a list of valid left-edge distances.
        
        The scan range defines where the insert can be placed:
        - start: leftmost position the LEFT edge can reach (relative to anchor)
        - end: rightmost position the RIGHT edge can reach (relative to anchor)
        
        Args:
            scan_range: (start, end) or (start, end, step)
            insert_idx: Index of the insert (for error messages)
        
        Returns:
            List of valid left-edge distances from anchor.
        """
        if len(scan_range) == 2:
            start, end = scan_range
            step = 1
        elif len(scan_range) == 3:
            start, end, step = scan_range
        else:
            raise ValueError(
                f"Insert {insert_idx} ({self._insert_names[insert_idx]}): "
                f"scan_range must be (start, end) or (start, end, step), "
                f"got {scan_range}"
            )
        
        if step <= 0:
            raise ValueError(
                f"Insert {insert_idx} ({self._insert_names[insert_idx]}): "
                f"step must be > 0, got {step}"
            )
        
        # The insert occupies [anchor+dist, anchor+dist+length)
        # Left edge can be at: anchor + start (leftmost)
        # Right edge can be at: anchor + end (rightmost)
        # So left edge can range from: start to end - length
        insert_length = self._insert_lengths[insert_idx]
        left_edge_max = end - insert_length
        
        if left_edge_max < start:
            raise ValueError(
                f"Insert {insert_idx} ({self._insert_names[insert_idx]}): "
                f"scan range [{start}, {end}] is too small for insert of length "
                f"{insert_length}. Need at least {insert_length}bp range."
            )
        
        # Generate left-edge distances
        distances = list(range(start, left_edge_max + 1, step))
        
        if len(distances) == 0:
            raise ValueError(
                f"Insert {insert_idx} ({self._insert_names[insert_idx]}): "
                f"scan range produces no valid positions"
            )
        
        return distances
    
    def _enumerate_valid_combinations(self) -> List[Tuple[int, ...]]:
        """Enumerate all valid distance combinations.
        
        Filters out combinations that:
        1. Place inserts out of background bounds
        2. Have overlapping inserts (gap < min_spacing)
        3. Violate ordering constraint (if enforce_order=True)
        
        Returns:
            List of valid (distance_0, distance_1, ...) tuples.
        """
        valid = []
        
        for combo in itertools.product(*self._distance_lists):
            if self._is_valid_combination(combo):
                valid.append(combo)
        
        return valid
    
    def _is_valid_combination(self, combo: Tuple[int, ...]) -> bool:
        """Check if a distance combination is valid.
        
        Args:
            combo: Tuple of distances, one per insert.
        
        Returns:
            True if valid, False otherwise.
        """
        n = len(combo)
        
        # Build intervals: (start, end, original_index)
        intervals = []
        for i, dist in enumerate(combo):
            start = self.anchor_pos + dist
            end = start + self._insert_lengths[i]
            intervals.append((start, end, i))
        
        # Check bounds (all inserts within background)
        # For insert mode, we allow inserting at position bg_length (appending)
        # For overwrite mode, the window must fit within background
        for start, end, _ in intervals:
            if start < 0:
                return False
            if self.insert_or_overwrite == 'overwrite':
                if end > self._bg_length:
                    return False
            else:  # insert mode
                # In insert mode, start position must be <= bg_length
                if start > self._bg_length:
                    return False
        
        # Sort by start position
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        
        # Check ordering constraint
        if self.enforce_order:
            original_indices = [x[2] for x in sorted_intervals]
            if original_indices != list(range(n)):
                return False  # Order would flip
        
        # Check min_spacing between adjacent (in sorted order)
        for j in range(len(sorted_intervals) - 1):
            gap = sorted_intervals[j + 1][0] - sorted_intervals[j][1]
            if gap < self.min_spacing:
                return False
        
        return True
    
    def _calculate_num_internal_states(self) -> int:
        """Number of valid combinations."""
        return len(self._valid_combinations)
    
    def _calculate_seq_length(self) -> int:
        """Calculate output sequence length."""
        if self.insert_or_overwrite == 'overwrite':
            return self._bg_length
        else:
            # Insert mode: adds all insert lengths
            return self._bg_length + sum(self._insert_lengths)
    
    def _get_background(self) -> str:
        """Get current background sequence."""
        if isinstance(self.background_seq, Pool):
            return self.background_seq.seq
        return self.background_seq
    
    def _get_insert(self, idx: int) -> str:
        """Get current sequence for insert at index."""
        ins = self.insert_seqs[idx]
        if isinstance(ins, Pool):
            return ins.seq
        return ins
    
    def _compute_seq(self) -> str:
        """Compute sequence with inserts placed at current state's positions."""
        # Get current combination
        state = self.get_state() % len(self._valid_combinations)
        combo = self._valid_combinations[state]
        self._cached_combo = combo
        
        # Get background
        bg = self._get_background()
        
        # Get insert sequences
        insert_seqs = [self._get_insert(i) for i in range(len(self.insert_seqs))]
        
        # Build placements: (bg_position, insert_seq, length, original_idx)
        placements = []
        for i, dist in enumerate(combo):
            pos = self.anchor_pos + dist
            seq = insert_seqs[i]
            if self.mark_changes:
                seq = seq.swapcase()
            placements.append((pos, seq, self._insert_lengths[i], i))
        
        # Sort by position (5' to 3')
        placements.sort(key=lambda x: x[0])
        
        # Calculate output positions for insert mode
        # (For overwrite mode, output positions = background positions)
        # Store as dict: original_idx -> (output_start, output_end)
        self._cached_output_positions = {}
        
        if self.insert_or_overwrite == 'insert':
            # Track cumulative shift from earlier insertions
            cumulative_shift = 0
            for bg_pos, seq, length, orig_idx in placements:
                output_start = bg_pos + cumulative_shift
                output_end = output_start + length
                self._cached_output_positions[orig_idx] = (output_start, output_end)
                cumulative_shift += length
        else:
            # Overwrite mode: positions don't shift
            for bg_pos, seq, length, orig_idx in placements:
                self._cached_output_positions[orig_idx] = (bg_pos, bg_pos + length)
        
        # Apply insertions from right to left to preserve positions
        result = bg
        for pos, seq, length, _ in reversed(placements):
            if self.insert_or_overwrite == 'overwrite':
                result = result[:pos] + seq + result[pos + length:]
            else:
                result = result[:pos] + seq + result[pos:]
        
        return result
    
    # =========================================================================
    # Design Card Metadata
    # =========================================================================
    
    def get_metadata(self, abs_start: int, abs_end: int) -> Dict[str, Any]:
        """Return metadata for this SpacingScanPool at the current state.
        
        Extends base Pool metadata with per-insert position information
        and pairwise spacing.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end only
            - 'features': core + per-insert coords + pairwise spacings (default)
            - 'complete': features + value
        
        For each insert, reports:
            - {name}_dist: Distance from anchor (in background coordinate space)
            - {name}_pos_start, {name}_pos_end: Position in output sequence
            - {name}_abs_pos_start, {name}_abs_pos_end: Absolute position in composite
        
        Note: For insert mode, pos_start/end account for shifts from earlier inserts.
        For overwrite mode, positions are the same as background coordinates.
        
        For each pair of inserts (i, j where i < j in list order), reports:
            - spacing_{name_i}_{name_j}: Signed gap between closest boundaries.
        
        Spacing sign convention:
            - Positive: Insert i is upstream (5') of insert j (expected order).
              Value = j_start - i_end (gap from i's end to j's start).
            - Negative: Insert j is upstream (5') of insert i (flipped order).
              Value = -(i_start - j_end) (negated gap from j's end to i's start).
            - abs(spacing) always gives the actual gap between closest boundaries.
            - When enforce_order=True, spacing is always positive.
        
        Args:
            abs_start: Absolute start position of this pool in the final sequence.
            abs_end: Absolute end position of this pool in the final sequence.
        
        Returns:
            Dictionary with metadata fields based on metadata level.
        """
        # Ensure cached combo is current by always triggering computation
        # This is necessary because get_metadata may be called multiple times
        # at different states without accessing .seq in between
        _ = self.seq
        
        # Get base metadata
        metadata = super().get_metadata(abs_start, abs_end)
        
        if self._metadata_level in ('features', 'complete'):
            combo = self._cached_combo
            n = len(combo)
            
            # Per-insert metadata using OUTPUT positions (from _cached_output_positions)
            # These are the actual positions in the generated sequence
            positions = []  # (start, end) in output sequence
            for i, dist in enumerate(combo):
                name = self._insert_names[i]
                
                # Output positions (accounting for shifts in insert mode)
                pos_start, pos_end = self._cached_output_positions[i]
                positions.append((pos_start, pos_end))
                
                # Distance from anchor (always in background coordinate space)
                metadata[f'{name}_dist'] = dist
                
                # Position in output sequence
                metadata[f'{name}_pos_start'] = pos_start
                metadata[f'{name}_pos_end'] = pos_end
                
                # Absolute position in composite (if abs_start is known)
                if abs_start is not None:
                    metadata[f'{name}_abs_pos_start'] = abs_start + pos_start
                    metadata[f'{name}_abs_pos_end'] = abs_start + pos_end
                else:
                    metadata[f'{name}_abs_pos_start'] = None
                    metadata[f'{name}_abs_pos_end'] = None
            
            # Pairwise spacings (based on output positions)
            # Uses signed distance: positive = expected order (i upstream of j),
            # negative = flipped order (j upstream of i).
            # Magnitude is always the gap between closest boundaries.
            for i in range(n):
                for j in range(i + 1, n):
                    name_i = self._insert_names[i]
                    name_j = self._insert_names[j]
                    
                    if positions[i][0] <= positions[j][0]:
                        # i is upstream of j (expected order)
                        # Gap = j_start - i_end (positive)
                        spacing = positions[j][0] - positions[i][1]
                    else:
                        # j is upstream of i (flipped order)
                        # Gap = i_start - j_end, but negate to indicate flip
                        spacing = -(positions[i][0] - positions[j][1])
                    
                    metadata[f'spacing_{name_i}_{name_j}'] = spacing
        
        return metadata
    
    def __repr__(self) -> str:
        n = len(self.insert_seqs)
        names = ", ".join(self._insert_names[:3])
        if n > 3:
            names += f", ... ({n} total)"
        return (
            f"SpacingScanPool(inserts=[{names}], anchor_pos={self.anchor_pos}, "
            f"valid_combinations={len(self._valid_combinations)})"
        )

