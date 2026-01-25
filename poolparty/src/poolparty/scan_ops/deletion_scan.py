"""Deletion scan operation - delete a segment at scanning positions."""
from numbers import Integral, Real
import numpy as np

from ..types import Union, ModeType, Optional, PositionsType, RegionType, StyleList, beartype
from ..party import get_active_party
from ..pool import Pool
from ..operation import Operation
from ..marker_ops.parsing import get_nonmarker_positions


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


@beartype
def deletion_scan(
    pool: Union[Pool, str],
    deletion_length: Integral,
    deletion_marker: Optional[str] = '-',
    region: RegionType = None,
    positions: PositionsType = None,
    min_spacing: Optional[Integral] = None,
    max_spacing: Optional[Integral] = None,
    prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_states: Optional[Integral] = None,
    style_deletion: Optional[str] = None,
    iter_order: Optional[Real] = None,
) -> Pool:
    """
    Scan a pool for all possible single deletions of a fixed length.

    Parameters
    ----------
    pool : Pool or str
        Source pool or sequence string to delete from.
    deletion_length : Integral
        Number of characters to delete at each valid position.
    deletion_marker : Optional[str], default='-'
        Character to insert at the deletion site. If None, segment is removed.
    region : RegionType, default=None
        Region to constrain the scan to. Can be a marker name (str) or [start, stop].
    positions : PositionsType, default=None
        Positions to consider for the start of the deletion (0-based, relative to region).
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : ModeType, default='random'
        Deletion mode: 'random' or 'sequential'.
    num_states : Optional[Integral], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    style_deletion : Optional[str], default=None
        Style to apply to deletion gap characters (e.g., 'gray', 'red bold').
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool yielding sequences where a segment of the specified length is removed
        from the source at each allowed position, optionally with a marker inserted.
    """
    from ..fixed_ops.from_seq import from_seq

    # Validate min_spacing/max_spacing not supported
    if min_spacing is not None or max_spacing is not None:
        raise ValueError(
            "min_spacing and max_spacing are not supported."
        )

    # Convert string to pool
    pool = from_seq(pool) if isinstance(pool, str) else pool

    # Validate bg_pool has defined seq_length (only when no region specified)
    bg_length = pool.seq_length
    if bg_length is None and region is None:
        raise ValueError("pool must have a defined seq_length")

    # Validate deletion_length
    if deletion_length <= 0:
        raise ValueError(f"del_length must be > 0, got {deletion_length}")
    if bg_length is not None and deletion_length >= bg_length:
        raise ValueError(
            f"del_length ({deletion_length}) must be < pool.seq_length ({bg_length})"
        )

    # Determine gap character
    gap_char = deletion_marker if deletion_marker is not None else None

    op = DeletionScanOp(
        parent_pool=pool,
        deletion_length=int(deletion_length),
        gap_char=gap_char,
        positions=positions,
        region=region,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        style_deletion=style_deletion,
        name=None,
        iter_order=iter_order,
    )
    result_pool = Pool(operation=op)
    
    return result_pool


class DeletionScanOp(Operation):
    """Delete a segment at scanning positions."""
    factory_name = "deletion_scan"
    design_card_keys = ['position_index', 'start', 'stop', 'length', 'deleted_content']
    
    def __init__(
        self,
        parent_pool,
        deletion_length: int,
        gap_char: Optional[str] = None,
        positions: PositionsType = None,
        region: RegionType = None,
        prefix: Optional[str] = None,
        mode: str = 'random',
        num_states: Optional[int] = None,
        style_deletion: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize DeletionScanOp."""
        self._deletion_length = deletion_length
        self._gap_char = gap_char
        self._positions = positions
        self._style_deletion = style_deletion
        
        # Determine effective seq_length for state calculation:
        # If region is a marker name, use the marker's registered length
        # Otherwise, use the parent pool's seq_length
        if isinstance(region, str):
            party = get_active_party()
            try:
                region_marker = party.get_marker_by_name(region)
                self._seq_length = region_marker.seq_length
            except (ValueError, KeyError):
                # Marker not yet registered, fall back to parent seq_length
                self._seq_length = parent_pool.seq_length
        else:
            self._seq_length = parent_pool.seq_length
        
        # Calculate number of states
        if mode == 'sequential':
            if self._seq_length is not None:
                num_states = self._build_caches()
            else:
                num_states = 1
        elif mode == 'random':
            # num_states stays None for pure random mode
            pass
        else:
            num_states = 1
        
        # Initialize as Operation
        super().__init__(
            parent_pools=[parent_pool],
            num_values=num_states,
            mode=mode,
            seq_length=None,  # Variable due to deletion/gap insertion
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=region,
        )
    
    def _build_caches(self) -> int:
        """Build caches for sequential enumeration based on seq_length."""
        if self._seq_length is None:
            if self._positions is not None:
                positions_list = _validate_positions(
                    self._positions,
                    max_position=1000000,
                    min_position=0,
                )
                return max(1, len(positions_list))
            return 1
        
        # For deletions, we need room for deletion_length bases
        max_start = self._seq_length - self._deletion_length
        if max_start < 0:
            max_start = 0
        
        num_all_positions = max_start + 1
        if self._positions is not None:
            indices = _validate_positions(
                self._positions,
                max_position=num_all_positions - 1,
                min_position=0,
            )
            num_states = len(indices)
        else:
            num_states = num_all_positions
        
        if num_states == 0:
            raise ValueError("No valid positions for deletion")
        return num_states
    
    def _get_valid_deletion_positions(self, seq: str) -> tuple[list[int], list[int]]:
        """Get valid deletion positions, excluding marker interiors.
        
        Returns tuple of (valid_nonmarker_indices, nonmarker_positions) where:
        - valid_nonmarker_indices: indices into nonmarker_positions that are valid start positions
        - nonmarker_positions: literal positions of all non-marker characters
        """
        # Get positions not inside existing markers
        nonmarker_positions = get_nonmarker_positions(seq)
        
        # For deletions, ensure there's room for deletion_length consecutive non-marker chars
        max_valid_idx = len(nonmarker_positions) - self._deletion_length
        if max_valid_idx < 0:
            all_valid_indices = []
        else:
            all_valid_indices = list(range(max_valid_idx + 1))
        
        # Apply user position filter
        if self._positions is not None:
            indices = _validate_positions(
                self._positions,
                max_position=len(all_valid_indices) - 1,
                min_position=0,
            )
            filtered_indices = [all_valid_indices[i] for i in indices]
            return filtered_indices, nonmarker_positions
        
        return all_valid_indices, nonmarker_positions
    
    def compute(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
        parent_styles: list | None = None,
    ) -> dict:
        """Return design card and sequence with deletion applied.
        
        Note: When region is specified, Operation.wrapped_compute() extracts
        the region and passes only the region content here. So seq is already
        the region content (or full sequence if no region).
        """
        seq = parent_seqs[0]
        
        # Get valid positions in the sequence
        valid_indices, nonmarker_positions = self._get_valid_deletion_positions(seq)
        if len(valid_indices) == 0:
            raise ValueError("No valid positions for deletion")
        
        # Select position based on mode
        if self.mode in ('random', 'hybrid'):
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")
            position_index = int(rng.integers(0, len(valid_indices)))
        else:
            state = self.state.value
            state = 0 if state is None else state
            position_index = state % len(valid_indices)
        
        # Calculate literal start/end positions
        start_idx = valid_indices[position_index]
        start_literal = nonmarker_positions[start_idx]
        end_nonmarker_idx = start_idx + self._deletion_length
        if end_nonmarker_idx < len(nonmarker_positions):
            end_literal = nonmarker_positions[end_nonmarker_idx]
        else:
            # One past the last non-marker character
            end_literal = nonmarker_positions[-1] + 1 if nonmarker_positions else len(seq)
        
        # Extract deleted content for design card
        deleted_content = seq[start_literal:end_literal]
        
        # Build gap content
        gap_content = self._gap_char * self._deletion_length if self._gap_char else ''
        
        # Build result sequence
        result_seq = seq[:start_literal] + gap_content + seq[end_literal:]
        
        # Adjust styles in one pass
        output_styles = self._adjust_styles(
            parent_styles, 
            start_literal, 
            end_literal, 
            gap_content
        )
        
        return {
            'position_index': position_index,
            'start': start_idx,
            'stop': start_idx + self._deletion_length,
            'length': self._deletion_length,
            'deleted_content': deleted_content,
            'seq': result_seq,
            'style': output_styles,
        }
    
    def _adjust_styles(
        self, 
        parent_styles: list | None, 
        del_start: int, 
        del_end: int, 
        gap_content: str
    ) -> StyleList:
        """Adjust style positions for deletion + optional gap insertion."""
        output_styles: StyleList = []
        
        if not parent_styles or len(parent_styles) == 0:
            # No parent styles, just add style_deletion if specified
            if self._style_deletion and gap_content:
                gap_positions = np.arange(del_start, del_start + len(gap_content), dtype=np.int64)
                output_styles.append((self._style_deletion, gap_positions))
            return output_styles
        
        input_styles = parent_styles[0]
        length_delta = len(gap_content) - (del_end - del_start)
        
        # Adjust existing styles
        for spec, positions in input_styles:
            adjusted_positions = []
            for pos in positions:
                if pos < del_start:
                    # Before deletion: unchanged
                    adjusted_positions.append(pos)
                elif pos >= del_end:
                    # After deletion: shift by length change
                    adjusted_positions.append(pos + length_delta)
                # Positions inside deleted region are discarded
            if adjusted_positions:
                output_styles.append((spec, np.array(adjusted_positions, dtype=np.int64)))
        
        # Add style_deletion for gap characters
        if self._style_deletion and gap_content:
            gap_positions = np.arange(del_start, del_start + len(gap_content), dtype=np.int64)
            output_styles.append((self._style_deletion, gap_positions))
        
        return output_styles
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'deletion_length': self._deletion_length,
            'gap_char': self._gap_char,
            'positions': self._positions,
            'region': self._region,
            'prefix': self.name_prefix,
            'mode': self.mode,
            'num_states': self.num_values if self.mode == 'random' and self.num_values is not None and self.num_values > 1 else None,
            'style_deletion': self._style_deletion,
            'name': None,
            'iter_order': self.iter_order,
        }
