"""BreakpointScan operation - split sequences at breakpoint positions."""

from itertools import combinations
from math import comb

from ..types import Union, Optional, ModeType, Sequence, beartype
from ..operation import Operation
from ..pool import Pool, MultiPool
from .from_seqs_op import from_seqs


@beartype
class BreakpointScanOp(Operation):
    """Split a sequence at breakpoint positions into multiple segments.
    
    For num_breakpoints=n, produces n+1 output segments stored in seq_0, seq_1, etc.
    This is a multi-output operation (num_outputs = num_breakpoints + 1).
    
    In sequential mode, iterates over all C(num_positions, num_breakpoints) combinations.
    In random mode, randomly selects breakpoint positions for each sequence.
    """
    design_card_keys: Sequence[str] = ['seq', 'breakpoint_positions']
    
    def __init__(
        self,
        parent: Pool,
        num_breakpoints: int,
        positions: Optional[Sequence[int]],
        start: int,
        end: Optional[int],
        step_size: int,
        mode: ModeType,
        name: Optional[str],
        design_card_keys: Optional[Sequence[str]],
    ):
        """Initialize BreakpointScanOp.
        
        Args:
            parent: Parent Pool to extract breakpoints from
            num_breakpoints: Number of breakpoints to introduce
            positions: Explicit positions where breakpoints can occur (overrides start/end/step_size)
            start: Start of range for breakpoint positions (default: 0)
            end: End of range for breakpoint positions (default: seq_length)
            step_size: Step between candidate positions (default: 1)
            mode: 'random' or 'sequential'
            name: Optional name for this operation
            design_card_keys: Which design card keys to include
        """
        if num_breakpoints < 1:
            raise ValueError(f"num_breakpoints must be >= 1, got {num_breakpoints}")
        
        self.num_breakpoints = num_breakpoints
        
        # Determine sequence length
        seq_length = parent.seq_length
        if seq_length is None:
            raise ValueError("BreakpointScanOp requires parent with known seq_length")
        self._seq_length = seq_length
        
        # Compute valid breakpoint positions
        if positions is not None:
            # User provided explicit positions - filter to valid interior positions
            self._positions = sorted([p for p in positions if 0 < p < seq_length])
        else:
            # Compute positions from range parameters
            end_val = end if end is not None else seq_length
            all_positions = list(range(start, end_val, step_size))
            # Breakpoints must be interior (not at 0 or seq_length)
            self._positions = [p for p in all_positions if 0 < p < seq_length]
        
        if len(self._positions) < num_breakpoints:
            raise ValueError(
                f"Not enough valid positions ({len(self._positions)}) "
                f"for {num_breakpoints} breakpoints. "
                f"Valid positions must be in range (0, {seq_length})."
            )
        
        # Calculate number of states
        num_states = comb(len(self._positions), num_breakpoints)
        
        # Set num_outputs (override class attribute)
        self.num_outputs = num_breakpoints + 1
        
        # Build design card keys dynamically to include all segment columns
        base_keys = list(BreakpointScanOp.design_card_keys)
        for i in range(self.num_outputs):
            key = f'seq_{i}'
            if key not in base_keys:
                base_keys.append(key)
        
        # Build sequential cache if feasible
        if num_states <= Operation.max_sequential_states:
            self._sequential_cache = list(combinations(self._positions, num_breakpoints))
        else:
            self._sequential_cache = None
        
        super().__init__(
            parent_pools=[parent],
            num_states=num_states,
            mode=mode,
            seq_length=None,  # Variable (depends on segment)
            name=name or 'breakpoint_scan',
            design_card_keys=design_card_keys if design_card_keys is not None else base_keys,
        )
    
    def _validate_card_keys(self, keys: Sequence[str]) -> list[str]:
        """Override to allow dynamic seq_N keys."""
        # Build the full set of valid keys including dynamic ones
        valid_keys = set(BreakpointScanOp.design_card_keys)
        for i in range(self.num_outputs):
            valid_keys.add(f'seq_{i}')
        
        keys_set = set(keys)
        if not keys_set.issubset(valid_keys):
            invalid = keys_set - valid_keys
            raise ValueError(
                f"Invalid design_card_keys {invalid}. Valid: {valid_keys}"
            )
        return list(keys)
    
    def compute_results_row(
        self, 
        input_strings: Sequence[str], 
        sequential_state: int,
    ) -> dict:
        """Compute segments for one state."""
        seq = input_strings[0]
        
        # Get breakpoint positions for this state
        if self.mode == 'sequential':
            if self._sequential_cache is None:
                raise RuntimeError("Sequential mode requires feasible number of states")
            breakpoints = self._sequential_cache[sequential_state % self.num_states]
        else:  # random mode
            breakpoints = tuple(sorted(
                self.rng.choice(self._positions, size=self.num_breakpoints, replace=False)
            ))
        
        # Split sequence at breakpoints
        # breakpoints = (p1, p2, ...) → segments = [0:p1, p1:p2, ..., pn:end]
        boundaries = [0] + list(breakpoints) + [len(seq)]
        segments = [seq[boundaries[i]:boundaries[i+1]] for i in range(len(boundaries)-1)]
        
        # Build result dict
        result = {
            'seq': segments[0],  # Default seq = first segment
            'breakpoint_positions': breakpoints,
        }
        for i, seg in enumerate(segments):
            result[f'seq_{i}'] = seg
        
        return result


@beartype
def breakpoint_scan(
    parent: Union[Pool, str],
    num_breakpoints: int = 1,
    positions: Optional[Sequence[int]] = None,
    start: int = 0,
    end: Optional[int] = None,
    step_size: int = 1,
    mode: ModeType = 'random',
    name: str = 'breakpoint_scan',
    design_card_keys: Optional[Sequence[str]] = None,
) -> MultiPool:
    """Split a sequence at breakpoint positions.
    
    Returns a MultiPool with num_breakpoints + 1 output Pools, one per segment.
    
    In sequential mode, iterates over all C(n, k) breakpoint combinations where
    n = number of valid positions and k = num_breakpoints. Sequential mode is
    only allowed if C(n, k) <= Operation.max_sequential_states.
    
    In random mode, randomly selects num_breakpoints positions for each sequence.
    
    Args:
        parent: Input sequence (string or Pool)
        num_breakpoints: Number of breakpoints to introduce (default: 1)
        positions: Explicit positions where breakpoints can occur.
            If provided, overrides start/end/step_size.
            Positions must be interior (0 < position < seq_length).
        start: Start of range for breakpoint positions (default: 0)
        end: End of range for breakpoint positions (default: seq_length)
        step_size: Step between candidate positions (default: 1)
        mode: 'random' or 'sequential'
        name: Name for the operation
        design_card_keys: Which design card keys to include
    
    Returns:
        MultiPool: Container with num_breakpoints + 1 Pools, one per segment.
            Can be unpacked: left, middle, right = breakpoint_scan(seq, num_breakpoints=2)
    
    Example:
        >>> # Split sequence into 3 segments with 2 breakpoints
        >>> multi = breakpoint_scan("ACGTACGTACGT", num_breakpoints=2, mode='sequential')
        >>> left, middle, right = multi
        >>> print(left.seq, middle.seq, right.seq)
        
        >>> # Use specific positions only
        >>> multi = breakpoint_scan("ACGTACGT", positions=[2, 4, 6], num_breakpoints=1)
        >>> left, right = multi
        
        >>> # Segments can be used in further operations
        >>> left, right = breakpoint_scan(seq, num_breakpoints=1)
        >>> mutated_right = mutation_scan(right, num_mutations=1)
        >>> reconstructed = left + mutated_right
    
    Raises:
        ValueError: If not enough valid positions for the requested breakpoints
        ValueError: If mode='sequential' but too many states
    """
    if isinstance(parent, str):
        parent = from_seqs([parent], design_card_keys=[])
    
    op = BreakpointScanOp(
        parent=parent,
        num_breakpoints=num_breakpoints,
        positions=positions,
        start=start,
        end=end,
        step_size=step_size,
        mode=mode,
        name=name,
        design_card_keys=design_card_keys,
    )
    
    return MultiPool(op)

