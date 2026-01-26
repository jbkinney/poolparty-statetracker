"""Deletion scan operation - delete a segment at scanning positions."""
from numbers import Integral, Real
import numpy as np

from ..types import Union, ModeType, Optional, PositionsType, RegionType, SeqStyle, beartype
from ..party import get_active_party
from ..pool import Pool
from ..operation import Operation
from ..utils.parsing_utils import get_nontag_positions
from ..utils import validate_positions, build_scan_cache


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
    style: Optional[str] = None,
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
    style : Optional[str], default=None
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
        style=style,
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
        style: Optional[str] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize DeletionScanOp."""
        self._deletion_length = deletion_length
        self._gap_char = gap_char
        self._positions = positions
        self._style = style
        
        # Determine effective seq_length for state calculation:
        # If region is a marker name, use the marker's registered length
        # Otherwise, use the parent pool's seq_length
        if isinstance(region, str):
            party = get_active_party()
            try:
                region_marker = party.get_region_by_name(region)
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
        return build_scan_cache(
            seq_length=self._seq_length,
            item_length=self._deletion_length,
            positions=self._positions,
            error_context="deletion",
        )
    
    def _get_valid_deletion_positions(self, seq: str) -> tuple[list[int], list[int]]:
        """Get valid deletion positions, excluding marker interiors.
        
        Returns tuple of (valid_nontag_indices, nontag_positions) where:
        - valid_nontag_indices: indices into nontag_positions that are valid start positions
        - nontag_positions: literal positions of all non-tag characters
        """
        # Get positions not inside existing tags
        nontag_positions = get_nontag_positions(seq)
        
        # For deletions, ensure there's room for deletion_length consecutive non-tag chars
        max_valid_idx = len(nontag_positions) - self._deletion_length
        if max_valid_idx < 0:
            all_valid_indices = []
        else:
            all_valid_indices = list(range(max_valid_idx + 1))
        
        # Apply user position filter
        if self._positions is not None:
            indices = validate_positions(
                self._positions,
                max_position=len(all_valid_indices) - 1,
                min_position=0,
            )
            filtered_indices = [all_valid_indices[i] for i in indices]
            return filtered_indices, nontag_positions
        
        return all_valid_indices, nontag_positions
    
    def compute(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
        parent_styles: list[SeqStyle] | None = None,
    ) -> dict:
        """Return design card and sequence with deletion applied.
        
        Note: When region is specified, Operation.wrapped_compute() extracts
        the region and passes only the region content here. So seq is already
        the region content (or full sequence if no region).
        """
        seq = parent_seqs[0]
        
        # Get valid positions in the sequence
        valid_indices, nontag_positions = self._get_valid_deletion_positions(seq)
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
        start_literal = nontag_positions[start_idx]
        end_nontag_idx = start_idx + self._deletion_length
        if end_nontag_idx < len(nontag_positions):
            end_literal = nontag_positions[end_nontag_idx]
        else:
            # One past the last non-tag character
            end_literal = nontag_positions[-1] + 1 if nontag_positions else len(seq)
        
        # Extract deleted content for design card
        deleted_content = seq[start_literal:end_literal]
        
        # Build gap content
        gap_content = self._gap_char * self._deletion_length if self._gap_char else ''
        
        # Build result sequence
        result_seq = seq[:start_literal] + gap_content + seq[end_literal:]
        
        # Adjust styles in one pass
        output_style = self._adjust_styles(
            parent_styles, 
            start_literal, 
            end_literal, 
            gap_content,
            len(seq)
        )
        
        return {
            'position_index': position_index,
            'start': start_idx,
            'stop': start_idx + self._deletion_length,
            'length': self._deletion_length,
            'deleted_content': deleted_content,
            'seq': result_seq,
            'style': output_style,
        }
    
    def _adjust_styles(
        self, 
        parent_styles: list[SeqStyle] | None, 
        del_start: int, 
        del_end: int, 
        gap_content: str,
        seq_len: int,
    ) -> SeqStyle:
        """Adjust style positions for deletion + optional gap insertion."""
        input_style = SeqStyle.from_parent(parent_styles, 0, seq_len)
        
        output_style = SeqStyle.join([
            input_style[:del_start],           # Prefix
            SeqStyle.empty(len(gap_content)),  # Gap spacer
            input_style[del_end:],             # Suffix
        ])
        
        # Add gap style if specified
        if self._style and gap_content:
            gap_positions = np.arange(del_start, del_start + len(gap_content), dtype=np.int64)
            output_style = output_style.add_style(self._style, gap_positions)
        
        return output_style
    
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
            'style': self._style,
            'name': None,
            'iter_order': self.iter_order,
        }
