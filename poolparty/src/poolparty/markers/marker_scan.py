"""MarkerScan operation - insert XML markers at scanning positions."""
from poolparty.types import Union, Optional, Literal
from numbers import Integral, Real
import numpy as np

from .parsing import build_marker_tag, MARKER_PATTERN, get_positions_without_markers
from ..operation import Operation

# Type aliases
PositionsType = Union[list[int], tuple[int, ...], slice, None]
ModeType = Literal['random', 'sequential', 'hybrid']
StrandType = Literal['+', '-', 'both']


def marker_scan(
    pool,
    marker: str = 'marker',
    positions: PositionsType = None,
    strand: str = '+',
    marker_length: int = 0,
    mode: str = 'random',
    num_hybrid_states: Optional[int] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
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
    strand : StrandType, default='+'
        Strand for the marker: '+', '-', or 'both'.
        If 'both', creates 2x states scanning both strands.
    marker_length : Integral, default=0
        Length of sequence to encompass. 0 creates zero-length markers (<name/>),
        >0 creates region markers (<name>BASES</name>).
    mode : ModeType, default='random'
        Position selection mode: 'random', 'sequential', or 'hybrid'.
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
        A Pool yielding sequences with the marker inserted at selected positions.

    Examples
    --------
    >>> with pp.Party():
    ...     # Zero-length marker at all positions
    ...     result = pp.marker_scan('ACGT', marker='ins', mode='sequential')
    ...     # Creates: '<ins/>ACGT', 'A<ins/>CGT', 'AC<ins/>GT', etc.
    ...
    ...     # Region marker of length 3
    ...     result = pp.marker_scan('ACGTACGT', marker='region',
    ...                              positions=[2], marker_length=3)
    ...     # Result: 'AC<region>GTA</region>CGT'
    ...
    ...     # Scan both strands
    ...     result = pp.marker_scan('ACGT', marker='site', strand='both',
    ...                              mode='sequential')
    ...     # Creates states for strand='+' and strand='-' at each position
    """
    from ..operations.from_seq import from_seq
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
        strand=strand,
        marker_length=int(marker_length),
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=op_name,
        iter_order=op_iter_order,
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
    design_card_keys = ['position', 'strand', 'marker_tag']
    
    def __init__(
        self,
        parent_pool,
        marker_name: str,
        positions: PositionsType = None,
        strand: str = '+',
        marker_length: int = 0,
        mode: str = 'random',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize MarkerScanOp."""
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")
        
        self.marker_name = marker_name
        self._positions = positions
        self._mode = mode
        self._strand = strand
        self._marker_length = marker_length
        self._seq_length = parent_pool.seq_length
        self._valid_positions = None
        self._sequential_cache = None
        
        # Calculate number of states
        if mode == 'sequential':
            if self._seq_length is not None:
                num_states = self._build_caches()
            else:
                num_states = 1
        elif mode == 'hybrid':
            num_states = num_hybrid_states
        else:
            num_states = 1
        
        # If strand='both', double the number of states
        if strand == 'both' and mode == 'sequential':
            num_states *= 2
        
        # Initialize as Operation
        super().__init__(
            parent_pools=[parent_pool],
            num_states=num_states,
            mode=mode,
            seq_length=None,  # Variable due to marker tags
            name=name,
            iter_order=iter_order,
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
    
    def _get_valid_marker_positions(self, seq: str) -> list[int]:
        """Get valid marker insertion positions, excluding marker interiors."""
        # Get positions not inside existing markers
        valid_raw_positions = get_positions_without_markers(seq)
        
        # For region markers, ensure there's room for marker_length bases
        if self._marker_length > 0:
            # Filter to positions where we have enough room
            all_valid = []
            for i, raw_pos in enumerate(valid_raw_positions):
                # Check if there are marker_length consecutive valid positions
                if i + self._marker_length <= len(valid_raw_positions):
                    all_valid.append(raw_pos)
        else:
            # For zero-length markers, all positions are valid plus end
            all_valid = valid_raw_positions + [len(seq)]
        
        # Apply user position filter
        if self._positions is not None:
            indices = _validate_positions(
                self._positions,
                max_position=len(all_valid) - 1,
                min_position=0,
            )
            return [all_valid[i] for i in indices]
        
        return all_valid
    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with insertion position and strand."""
        seq = parent_seqs[0]
        
        valid_positions = self._get_valid_marker_positions(seq)
        if len(valid_positions) == 0:
            raise ValueError("No valid positions for marker insertion")
        
        # Determine strand for this state
        if self._strand == 'both':
            # For sequential mode, alternate strands
            if self.mode == 'sequential':
                state = self.counter.state
                state = 0 if state is None else state
                num_pos = len(valid_positions)
                position_index = (state // 2) % num_pos
                strand = '+' if (state % 2) == 0 else '-'
            else:
                # Random mode: randomly choose strand
                if rng is None:
                    raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")
                position_index = int(rng.integers(0, len(valid_positions)))
                strand = '+' if rng.random() < 0.5 else '-'
        else:
            strand = self._strand
            if self.mode in ('random', 'hybrid'):
                if rng is None:
                    raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")
                position_index = int(rng.integers(0, len(valid_positions)))
            else:
                state = self.counter.state
                state = 0 if state is None else state
                position_index = state % len(valid_positions)
        
        # Build marker tag
        raw_position = valid_positions[position_index]
        if self._marker_length > 0:
            content = seq[raw_position:raw_position + self._marker_length]
            marker_tag = build_marker_tag(self.marker_name, content, strand)
        else:
            marker_tag = build_marker_tag(self.marker_name, '', strand)
        
        return {
            'position': position_index,
            'strand': strand,
            'marker_tag': marker_tag,
        }
    
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Insert marker at position based on design card."""
        seq = parent_seqs[0]
        position_index = card['position']
        marker_tag = card['marker_tag']
        
        valid_positions = self._get_valid_marker_positions(seq)
        raw_position = valid_positions[position_index]
        
        if self._marker_length > 0:
            # Region marker: replace content with marker
            result_seq = (
                seq[:raw_position] +
                marker_tag +
                seq[raw_position + self._marker_length:]
            )
        else:
            # Zero-length marker: insert at position
            result_seq = seq[:raw_position] + marker_tag + seq[raw_position:]
        
        return {'seq_0': result_seq}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'marker_name': self.marker_name,
            'positions': self._positions,
            'strand': self._strand,
            'marker_length': self._marker_length,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }
