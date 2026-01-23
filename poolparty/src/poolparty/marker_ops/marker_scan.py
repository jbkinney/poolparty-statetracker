"""MarkerScan operation - insert XML markers at scanning positions."""
from poolparty.types import Union, Optional, Literal, RegionType
from numbers import Integral, Real
import numpy as np

from .parsing import build_marker_tag, TAG_PATTERN, get_nonmarker_positions
from ..operation import Operation

# Type aliases
PositionsType = Union[list[int], tuple[int, ...], slice, None]
ModeType = Literal['random', 'sequential']
StrandType = Literal['+', '-', 'both']


def marker_scan(
    pool,
    marker: str = 'marker',
    positions: PositionsType = None,
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    spacer_str: str = '',
    strand: str = '+',
    marker_length: int = 0,
    seq_name_prefix: Optional[str] = None,
    mode: str = 'random',
    num_states: Optional[int] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = None,
):
    """
    Insert XML-style markers at scanning positions in a sequence.

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string to insert marker into.
    marker : str, default='marker'
        Name for the marker to insert.
    positions : PositionsType, default=None
        Valid insertion positions (0-based). If None, all positions are valid.
    region : RegionType, default=None
        Region to constrain the scan to. Can be marker name (str) or [start, stop].
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
    strand : StrandType, default='+'
        Strand for the marker: '+', '-', or 'both'.
    marker_length : Integral, default=0
        Length of sequence to encompass. 0 creates zero-length markers (<name/>),
        >0 creates region markers (<name>BASES</name>).
    mode : ModeType, default='random'
        Position selection mode: 'random' or 'sequential'.
    _factory_name : Optional[str], default=None
        Sets default name of the resulting operation

    Returns
    -------
    Pool
        A Pool yielding sequences with the marker inserted at selected positions.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..pool import Pool
    from ..party import get_active_party
    
    # Convert string input to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool
    
    # Validate marker_length
    if marker_length < 0:
        raise ValueError(f"marker_length must be >= 0, got {marker_length}")
    
    # Register the marker with the Party
    party = get_active_party()
    registered_marker = party.register_marker(marker, marker_length)
    
    op = MarkerScanOp(
        parent_pool=pool,
        marker_name=marker,
        positions=positions,
        region=region,
        remove_marker=remove_marker,
        spacer_str=spacer_str,
        strand=strand,
        marker_length=int(marker_length),
        seq_name_prefix=seq_name_prefix,
        mode=mode,
        num_states=num_states,
        name=op_name,
        iter_order=op_iter_order,
        _factory_name=_factory_name,
    )
    result_pool = Pool(operation=op, name=name, iter_order=iter_order)
    
    # Add the marker to the pool's marker set
    result_pool.add_marker(registered_marker)
    
    return result_pool


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


class MarkerScanOp(Operation):
    """Insert XML markers at scanning positions."""
    factory_name = "marker_scan"
    design_card_keys = ['position_index', 'start', 'stop', 'length', 'region_name', 'region_content', 'strand', 'region_seq']
    
    def __init__(
        self,
        parent_pool,
        marker_name: str,
        positions: PositionsType = None,
        region: RegionType = None,
        remove_marker: Optional[bool] = None,
        spacer_str: str = '',
        strand: str = '+',
        marker_length: int = 0,
        seq_name_prefix: Optional[str] = None,
        mode: str = 'random',
        num_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        _factory_name: Optional[str] = None,
    ) -> None:
        """Initialize MarkerScanOp."""
        from ..party import get_active_party
        
        
        self.marker_name = marker_name
        self._positions = positions
        self._mode = mode
        self._strand = strand
        self._marker_length = marker_length
        self._region = region  # Store early for logging
        self._valid_positions = None
        self._sequential_cache = None
        
        # Determine effective seq_length for cache building:
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
        
        # Set factory name if provided
        if _factory_name is not None:
            self.factory_name = _factory_name
 
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
        
        # If strand='both', double the number of states
        if strand == 'both' and mode == 'sequential':
            num_states *= 2
        
        # Initialize as Operation
        super().__init__(
            parent_pools=[parent_pool],
            num_values=num_states,
            mode=mode,
            seq_length=None,  # Variable due to marker tags
            name=name,
            iter_order=iter_order,
            seq_name_prefix=seq_name_prefix,
            region=region,
            remove_marker=remove_marker,
            spacer_str=spacer_str,
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
        
        # For region markers, we need room for marker_length bases
        max_start = self._seq_length - self._marker_length
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
            raise ValueError("No valid positions for marker insertion")
        return num_states
    
    def _get_valid_marker_positions(self, seq: str) -> tuple[list[int], list[int]]:
        """Get valid marker insertion positions, excluding marker interiors.
        
        Returns tuple of (valid_nonmarker_indices, nonmarker_positions) where:
        - valid_nonmarker_indices: indices into nonmarker_positions that are valid start positions
        - nonmarker_positions: literal positions of all non-marker characters
        """
        # Get positions not inside existing markers
        nonmarker_positions = get_nonmarker_positions(seq)
        
        # For region markers, ensure there's room for marker_length bases
        if self._marker_length > 0:
            # Valid indices are those where we have room for marker_length consecutive non-marker chars
            max_valid_idx = len(nonmarker_positions) - self._marker_length
            if max_valid_idx < 0:
                all_valid_indices = []
            else:
                all_valid_indices = list(range(max_valid_idx + 1))
        else:
            # For zero-length markers, all positions are valid plus end (len of seq)
            all_valid_indices = list(range(len(nonmarker_positions) + 1))
        
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
    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with insertion position and strand."""
        seq = parent_seqs[0]
        
        valid_indices, nonmarker_positions = self._get_valid_marker_positions(seq)
        if len(valid_indices) == 0:
            raise ValueError("No valid positions for marker insertion")
        
        # Determine strand for this state
        if self._strand == 'both':
            # For sequential mode, alternate strands
            if self.mode == 'sequential':
                state = self.state.value
                state = 0 if state is None else state
                num_pos = len(valid_indices)
                position_index = (state // 2) % num_pos
                strand = '+' if (state % 2) == 0 else '-'
            else:
                # Random mode: randomly choose strand
                if rng is None:
                    raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")
                position_index = int(rng.integers(0, len(valid_indices)))
                strand = '+' if rng.random() < 0.5 else '-'
        else:
            strand = self._strand
            if self.mode in ('random', 'hybrid'):
                if rng is None:
                    raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")
                position_index = int(rng.integers(0, len(valid_indices)))
            else:
                state = self.state.value
                state = 0 if state is None else state
                position_index = state % len(valid_indices)
        
        # Build marker tag - extract content using non-marker indices
        nonmarker_idx = valid_indices[position_index]
        explicit_strand = (self._strand == 'both')
        if self._marker_length > 0:
            # Extract content from non-marker characters only
            content = ''.join(
                seq[nonmarker_positions[i]]
                for i in range(nonmarker_idx, nonmarker_idx + self._marker_length)
            )
            marker_tag = build_marker_tag(self.marker_name, content, strand, explicit_strand=explicit_strand)
            start = nonmarker_idx
            stop = nonmarker_idx + self._marker_length
            # Get raw sequence from literal start to end (including markers/gaps, excluding new marker_tag)
            start_literal = nonmarker_positions[nonmarker_idx]
            end_nonmarker_idx = nonmarker_idx + self._marker_length
            if end_nonmarker_idx < len(nonmarker_positions):
                end_literal = nonmarker_positions[end_nonmarker_idx]
            else:
                end_literal = nonmarker_positions[-1] + 1 if nonmarker_positions else len(seq)
            marked_seq = seq[start_literal:end_literal]
        else:
            marker_tag = build_marker_tag(self.marker_name, '', strand, explicit_strand=explicit_strand)
            start = nonmarker_idx
            stop = nonmarker_idx
            marked_seq = ''
        
        return {
            'position_index': position_index,
            'start': start,
            'stop': stop,
            'length': self._marker_length,
            'region_name': self.marker_name,
            'region_content': marked_seq,
            'strand': strand,
            'region_seq': marker_tag,
        }
    
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Insert marker at position based on design card."""
        seq = parent_seqs[0]
        position_index = card['position_index']
        marker_tag = card['region_seq']
        
        valid_indices, nonmarker_positions = self._get_valid_marker_positions(seq)
        nonmarker_idx = valid_indices[position_index]
        
        if self._marker_length > 0:
            # Region marker: replace content with marker
            # Get literal start and end positions from non-marker indices
            start_literal = nonmarker_positions[nonmarker_idx]
            end_nonmarker_idx = nonmarker_idx + self._marker_length
            # End position is the literal position of the first char AFTER the region
            if end_nonmarker_idx < len(nonmarker_positions):
                end_literal = nonmarker_positions[end_nonmarker_idx]
            else:
                # One past the last non-marker character (preserves trailing marker tags)
                end_literal = nonmarker_positions[-1] + 1 if nonmarker_positions else len(seq)
            result_seq = seq[:start_literal] + marker_tag + seq[end_literal:]
        else:
            # Zero-length marker: insert at position
            if nonmarker_idx < len(nonmarker_positions):
                raw_position = nonmarker_positions[nonmarker_idx]
            else:
                raw_position = len(seq)  # Insert at end
            result_seq = seq[:raw_position] + marker_tag + seq[raw_position:]
        
        return {'seq_0': result_seq}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'marker_name': self.marker_name,
            'positions': self._positions,
            'region': self._region,
            'remove_marker': self._remove_marker,
            'spacer_str': self._spacer_str,
            'strand': self._strand,
            'marker_length': self._marker_length,
            'seq_name_prefix': self.name_prefix,
            'mode': self.mode,
            'num_states': self.num_values if self.mode == 'random' and self.num_values is not None and self.num_values > 1 else None,
            'name': None,
            'iter_order': self.iter_order,
        }
