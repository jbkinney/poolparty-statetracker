"""Insert multiple XML markers into a sequence."""
from poolparty.types import Union, Optional, Sequence, Literal
from numbers import Integral, Real
import numpy as np

from .parsing import build_marker_tag, get_nonmarker_positions, nonmarker_pos_to_literal_pos
from ..operation import Operation

# Type aliases
PositionsType = Union[list[int], tuple[int, ...], slice, None]


def _validate_positions(positions: PositionsType, max_position: int, min_position: int = 0) -> list[int]:
    """Validate and normalize position specification."""
    if positions is None:
        return list(range(min_position, max_position + 1))
    
    if isinstance(positions, slice):
        start = positions.start if positions.start is not None else min_position
        stop = positions.stop if positions.stop is not None else max_position + 1
        step = positions.step if positions.step is not None else 1
        return list(range(start, stop, step))
    
    positions_list = list(positions)
    for p in positions_list:
        if p < min_position or p > max_position:
            raise ValueError(
                f"Position {p} out of range [{min_position}, {max_position}]"
            )
    return positions_list


def marker_multiscan(
    pool,
    markers,
    num_insertions: int,
    positions: PositionsType = None,
    strand: str = '+',
    marker_length: int = 0,
    insertion_mode: Literal['ordered', 'unordered'] = 'ordered',
    seq_name_prefix: Optional[str] = None,
    mode: str = 'random',
    num_hybrid_states: Optional[int] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
):
    """
    Insert multiple XML-style markers into a sequence.

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string to insert markers into.
    markers : Sequence[str] or str
        Marker name(s) to insert. If a single string, used for all insertions.
    num_insertions : Integral
        Number of markers to insert.
    positions : PositionsType, default=None
        Valid insertion positions (0-based). If None, all positions are valid.
    strand : StrandType, default='+'
        Strand for markers: '+', '-', or 'both'.
    marker_length : Integral, default=0
        Length of sequence to encompass per marker. 0 for zero-length markers.
    insertion_mode : str, default='ordered'
        How to assign markers to positions:
        - 'ordered': markers[i] goes to the i-th selected position
        - 'unordered': randomly assign markers to positions
    mode : ModeType, default='random'
        Position selection mode: 'random' or 'hybrid'.
    num_hybrid_states : Optional[Integral], default=None
        Number of pool states when using 'hybrid' mode.
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operation.

    Returns
    -------
    Pool
        A Pool yielding sequences with multiple markers inserted.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..pool import Pool
    from ..party import get_active_party

    pool = from_seq(pool) if isinstance(pool, str) else pool
    
    # Register all markers with the Party
    party = get_active_party()
    marker_names = [markers] if isinstance(markers, str) else list(markers)
    registered_markers = []
    for marker_name in marker_names:
        registered_marker = party.register_marker(marker_name, marker_length)
        registered_markers.append(registered_marker)
    
    op = MarkerMultiScanOp(
        parent_pool=pool,
        markers=markers,
        num_insertions=int(num_insertions),
        positions=positions,
        strand=strand,
        marker_length=int(marker_length),
        insertion_mode=insertion_mode,
        seq_name_prefix=seq_name_prefix,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=op_name,
        iter_order=op_iter_order,
    )
    result_pool = Pool(operation=op, name=name, iter_order=iter_order)
    
    # Add all registered markers to the pool
    for registered_marker in registered_markers:
        result_pool.add_marker(registered_marker)
    
    return result_pool


class MarkerMultiScanOp(Operation):
    """Insert multiple XML markers at selected positions."""

    factory_name = "marker_multiscan"
    design_card_keys = ['indices', 'strands', 'marker_tags']

    def __init__(
        self,
        parent_pool,
        markers,
        num_insertions: int,
        positions: PositionsType = None,
        strand: str = '+',
        marker_length: int = 0,
        insertion_mode: str = 'ordered',
        seq_name_prefix: Optional[str] = None,
        mode: str = 'random',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        if num_insertions < 1:
            raise ValueError(f"num_insertions must be >= 1, got {num_insertions}")
        if mode not in ('random', 'hybrid'):
            raise ValueError("marker_multiscan supports only mode='random' or 'hybrid'")
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")
        if marker_length < 0:
            raise ValueError(f"marker_length must be >= 0, got {marker_length}")

        self._positions = positions
        self._mode = mode
        self._strand = strand
        self._marker_length = marker_length
        self._seq_length = parent_pool.seq_length
        self.num_insertions = num_insertions
        self.insertion_mode = insertion_mode
        self._marker_names = self._coerce_markers(markers)
        self._validate_marker_counts()

        num_states = 1 if mode == 'random' else num_hybrid_states
        super().__init__(
            parent_pools=[parent_pool],
            num_values=num_states,
            mode=mode,
            seq_length=None,
            name=name,
            iter_order=iter_order,
            seq_name_prefix=seq_name_prefix,
        )

    def _coerce_markers(
        self, markers: Union[Sequence[str], str]
    ) -> list[str]:
        """Normalize markers input to a list of marker names."""
        if isinstance(markers, str):
            markers = [markers]
        if not markers:
            raise ValueError("markers must not be empty")
        return list(markers)

    def _validate_marker_counts(self) -> None:
        """Validate marker counts against insertion_mode."""
        if self.insertion_mode not in ('ordered', 'unordered'):
            raise ValueError(
                "insertion_mode must be one of 'ordered', 'unordered'"
            )
        if self.insertion_mode == 'ordered' and len(self._marker_names) != self.num_insertions:
            raise ValueError(
                "insertion_mode='ordered' requires len(markers) == num_insertions"
            )
        if self.insertion_mode == 'unordered' and len(self._marker_names) < self.num_insertions:
            raise ValueError(
                "insertion_mode='unordered' requires len(markers) >= num_insertions"
            )

    def _get_valid_marker_indices(self, seq: str) -> list[int]:
        """Return valid nonmarker indices (0 to n-1) for marker insertion.
        
        Returns logical indices into the non-marker character positions,
        not literal string positions. This allows proper handling of
        multiple marker insertions without position corruption.
        """
        nonmarker_positions = get_nonmarker_positions(seq)
        num_nonmarker = len(nonmarker_positions)
        
        if self._marker_length > 0:
            # For region markers, ensure room for content
            # Valid indices are 0 to (num_nonmarker - marker_length)
            max_valid_idx = num_nonmarker - self._marker_length
            if max_valid_idx < 0:
                return []
            all_valid = list(range(max_valid_idx + 1))
        else:
            # For zero-length markers, can insert at any position including end
            all_valid = list(range(num_nonmarker + 1))

        if self._positions is not None:
            indices = _validate_positions(
                self._positions,
                max_position=len(all_valid) - 1,
                min_position=0,
            )
            return [all_valid[i] for i in indices]

        return all_valid

    def _select_indices(
        self, valid_indices: list[int], rng: np.random.Generator
    ) -> list[int]:
        """Select nonmarker indices for marker insertion.
        
        For region markers (marker_length > 0), ensures selected indices
        are at least marker_length apart to prevent overlapping regions.
        """
        if len(valid_indices) < self.num_insertions:
            raise ValueError(
                f"Not enough valid positions ({len(valid_indices)}) "
                f"for {self.num_insertions} insertions"
            )
        
        if self._marker_length > 0:
            # For region markers, need non-overlapping selection
            # Use greedy algorithm with random shuffling
            shuffled = list(valid_indices)
            rng.shuffle(shuffled)
            
            chosen = []
            for idx in shuffled:
                # Check if this idx overlaps with any already chosen
                overlaps = False
                for c in chosen:
                    if abs(idx - c) < self._marker_length:
                        overlaps = True
                        break
                if not overlaps:
                    chosen.append(idx)
                    if len(chosen) == self.num_insertions:
                        break
            
            if len(chosen) < self.num_insertions:
                raise ValueError(
                    f"Cannot select {self.num_insertions} non-overlapping positions "
                    f"with marker_length={self._marker_length} from "
                    f"{len(valid_indices)} valid positions"
                )
            return sorted(chosen)
        else:
            # Zero-length markers don't overlap
            chosen = rng.choice(
                valid_indices,
                size=self.num_insertions,
                replace=False,
            )
            return sorted(int(x) for x in chosen)

    def _select_strands(self, rng: np.random.Generator) -> list[str]:
        """Select strands for each insertion."""
        if self._strand == 'both':
            return ['+' if rng.random() < 0.5 else '-' for _ in range(self.num_insertions)]
        else:
            return [self._strand] * self.num_insertions

    def _select_marker_tags(
        self, seq: str, indices: list[int], strands: list[str], rng: np.random.Generator
    ) -> list[str]:
        """Build marker tags for each nonmarker index.
        
        Args:
            seq: The original sequence string.
            indices: Nonmarker indices (logical positions, not literal).
            strands: Strand for each marker.
            rng: Random number generator.
        
        Returns:
            List of marker tag strings.
        """
        if self.insertion_mode == 'ordered':
            names = self._marker_names
        else:
            # unordered: randomly select from available markers
            idxs = rng.choice(len(self._marker_names), size=self.num_insertions, replace=False)
            names = [self._marker_names[int(i)] for i in idxs]
        
        tags = []
        for idx, strand, name in zip(indices, strands, names):
            if self._marker_length > 0:
                # Convert nonmarker index to literal positions for content extraction
                literal_start = nonmarker_pos_to_literal_pos(seq, idx)
                literal_end = nonmarker_pos_to_literal_pos(seq, idx + self._marker_length)
                content = seq[literal_start:literal_end]
                # Strip any existing marker tags from content (keep only actual characters)
                from .parsing import strip_all_markers
                content = strip_all_markers(content)
            else:
                content = ''
            tags.append(build_marker_tag(name, content, strand))
        return tags

    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        seq = parent_seqs[0]
        if rng is None:
            raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")

        valid_indices = self._get_valid_marker_indices(seq)
        indices = self._select_indices(valid_indices, rng)
        strands = self._select_strands(rng)
        marker_tags = self._select_marker_tags(seq, indices, strands, rng)

        return {
            'indices': indices,  # nonmarker indices, not literal positions
            'strands': strands,
            'marker_tags': marker_tags,
        }

    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Build result sequence with markers inserted at nonmarker indices.
        
        Uses single-pass construction to avoid position corruption issues
        when inserting multiple overlapping markers.
        """
        seq = parent_seqs[0]
        indices = list(card['indices'])
        marker_tags = list(card['marker_tags'])
        
        # Sort by index ascending for left-to-right processing
        inserts = sorted(zip(indices, marker_tags), key=lambda x: x[0])
        
        # Build result string from left to right
        result_parts = []
        prev_end_idx = 0  # Next nonmarker index to copy from
        
        for nm_idx, tag in inserts:
            # Copy characters from prev_end_idx to nm_idx (exclusive)
            if prev_end_idx < nm_idx:
                start_literal = nonmarker_pos_to_literal_pos(seq, prev_end_idx)
                end_literal = nonmarker_pos_to_literal_pos(seq, nm_idx)
                result_parts.append(seq[start_literal:end_literal])
            
            # Add the marker tag (which contains content for region markers)
            result_parts.append(tag)
            
            # Update prev_end_idx based on marker type
            if self._marker_length > 0:
                # Region marker: skip the characters that are now inside the tag
                prev_end_idx = nm_idx + self._marker_length
            else:
                # Zero-length marker: don't skip any characters
                prev_end_idx = nm_idx
        
        # Add remaining characters after the last marker
        nonmarker_positions = get_nonmarker_positions(seq)
        if prev_end_idx < len(nonmarker_positions):
            start_literal = nonmarker_pos_to_literal_pos(seq, prev_end_idx)
            result_parts.append(seq[start_literal:])
        elif prev_end_idx == len(nonmarker_positions):
            # Edge case: last marker ends exactly at end of sequence
            # There might be trailing marker tags to preserve
            last_nonmarker_literal = nonmarker_positions[-1] if nonmarker_positions else 0
            if last_nonmarker_literal + 1 < len(seq):
                result_parts.append(seq[last_nonmarker_literal + 1:])
        
        return {'seq_0': ''.join(result_parts)}

    def _get_copy_params(self) -> dict:
        return {
            'parent_pool': self.parent_pools[0],
            'markers': self._marker_names,
            'num_insertions': self.num_insertions,
            'positions': self._positions,
            'strand': self._strand,
            'marker_length': self._marker_length,
            'insertion_mode': self.insertion_mode,
            'seq_name_prefix': self.name_prefix,
            'mode': self.mode,
            'num_hybrid_states': self.num_values if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }
