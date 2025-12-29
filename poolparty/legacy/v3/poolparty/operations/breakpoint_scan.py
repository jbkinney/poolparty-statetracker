"""BreakpointScan operation - split sequences at breakpoint positions."""

from itertools import combinations
from math import comb
import numpy as np

from ..types import Union, Sequence, ModeType, beartype
from ..operation import Operation
from ..pool import Pool
from ..party import get_active_party


class BreakpointScanOp(Operation):
    """Split a sequence at breakpoint positions.
    
    For num_breakpoints=n, produces n+1 output segments.
    This is a multi-output operation.
    
    In sequential mode, iterates over all C(L-1, n) breakpoint combinations.
    In random mode, randomly selects breakpoint positions.
    """
    
    design_card_keys = ['breakpoints']
    
    @beartype
    def __init__(
        self,
        parent_pool: Pool,
        num_breakpoints: int = 1,
        positions: Sequence[int] | None = None,
        start: int | None = None,
        end: int | None = None,
        step_size: int = 1,
        min_spacing: int | None = None,
        max_spacing: int | None = None,
        mode: ModeType = 'sequential',
        name: str = 'breakpoint_scan',
    ) -> None:
        """Initialize BreakpointScanOp.
        
        Args:
            parent_pool: Input sequence pool
            num_breakpoints: Number of breakpoints (produces n+1 segments)
            positions: Valid breakpoint positions (overrides start/end/step_size)
            start: First valid breakpoint position (default: 1)
            end: Last valid breakpoint position (default: seq_length - 1)
            step_size: Step between valid positions (default: 1)
            min_spacing: Minimum spacing between consecutive breakpoints
            max_spacing: Maximum spacing between consecutive breakpoints
            mode: 'sequential' or 'random'
            name: Operation name
        """
        if num_breakpoints < 1:
            raise ValueError(f"num_breakpoints must be >= 1, got {num_breakpoints}")
        
        self.num_breakpoints = num_breakpoints
        self.positions = list(positions) if positions is not None else None
        self.start = start
        self.end = end
        self.step_size = step_size
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        
        # Set num_outputs for multi-output
        self.num_outputs = num_breakpoints + 1
        
        # Try to determine sequence length from parent
        self._seq_length: int | None = None
        self._valid_positions: list[int] | None = None
        self._sequential_cache: list | None = None
        
        if hasattr(parent_pool.operation, 'seqs'):
            self._seq_length = len(parent_pool.operation.seqs[0])
            num_states = self._build_caches(mode)
        else:
            num_states = 1  # Placeholder
        
        super().__init__(
            parent_pools=[parent_pool],
            num_states=num_states,
            mode=mode,
            name=name,
        )
        
        # Register with active party
        party = get_active_party()
        if party is not None:
            party._register_operation(self)
    
    @beartype
    def _is_valid_spacing(self, breakpoints: tuple[int, ...]) -> bool:
        """Check if breakpoints satisfy spacing constraints.
        
        Args:
            breakpoints: Tuple of breakpoint positions (sorted)
            
        Returns:
            True if all consecutive spacings are within [min_spacing, max_spacing]
        """
        if len(breakpoints) < 2:
            return True  # No spacing to check for single breakpoint
        for i in range(len(breakpoints) - 1):
            spacing = breakpoints[i + 1] - breakpoints[i]
            if self.min_spacing is not None and spacing < self.min_spacing:
                return False
            if self.max_spacing is not None and spacing > self.max_spacing:
                return False
        return True
    
    @beartype
    def _build_caches(self, mode: ModeType) -> int:
        """Build caches for sequential enumeration.
        
        Returns:
            Number of valid combinations after filtering
        """
        if self._seq_length is None:
            return 1
        
        # Determine valid breakpoint positions
        if self.positions is not None:
            self._valid_positions = [p for p in self.positions if 0 < p < self._seq_length]
        else:
            # Use start/end/step_size to generate positions
            start = self.start if self.start is not None else 1
            end = self.end if self.end is not None else self._seq_length - 1
            self._valid_positions = list(range(start, end + 1, self.step_size))
        
        if len(self._valid_positions) < self.num_breakpoints:
            raise ValueError(
                f"Not enough valid positions ({len(self._valid_positions)}) "
                f"for {self.num_breakpoints} breakpoints"
            )
        
        # 1. Validate unfiltered count first (ensures enumerable)
        num_combinations = comb(len(self._valid_positions), self.num_breakpoints)
        Operation.validate_num_states(num_combinations, mode)
        
        # 2. Generate and filter combinations by spacing constraints
        all_combos = list(combinations(self._valid_positions, self.num_breakpoints))
        self._sequential_cache = [c for c in all_combos if self._is_valid_spacing(c)]
        
        # Check if any valid combinations remain after filtering
        if len(self._sequential_cache) == 0:
            raise ValueError(
                f"No valid breakpoint combinations after applying spacing constraints "
                f"(min_spacing={self.min_spacing}, max_spacing={self.max_spacing})"
            )
        
        # 3. Return filtered count as actual num_states
        return len(self._sequential_cache)
    
    @beartype
    def compute(
        self,
        parent_seqs: list[str],
        state: int,
        rng: np.random.Generator | None,
    ) -> dict:
        """Split sequence at breakpoints."""
        seq = parent_seqs[0]
        seq_len = len(seq)
        
        # Build caches if needed
        if self._sequential_cache is None:
            self._seq_length = seq_len
            self.num_states = self._build_caches(self.mode)
        
        # Get breakpoint positions
        if self.mode == 'random':
            if rng is None:
                raise RuntimeError("Random mode requires RNG")
            # Sample from filtered cache to ensure spacing constraints are satisfied
            idx = rng.integers(len(self._sequential_cache))
            breakpoints = self._sequential_cache[idx]
        else:  # sequential
            breakpoints = self._sequential_cache[state % len(self._sequential_cache)]
        
        # Split sequence
        boundaries = [0] + list(breakpoints) + [seq_len]
        segments = [seq[boundaries[i]:boundaries[i+1]] for i in range(self.num_outputs)]
        
        # Build result with seq_0, seq_1, etc.
        result = {'breakpoints': breakpoints}
        for i, segment in enumerate(segments):
            result[f'seq_{i}'] = segment
        
        return result


@beartype
def breakpoint_scan(
    parent: Union[Pool, str],
    num_breakpoints: int = 1,
    positions: Sequence[int] | None = None,
    start: int | None = None,
    end: int | None = None,
    step_size: int = 1,
    min_spacing: int | None = None,
    max_spacing: int | None = None,
    mode: ModeType = 'sequential',
    name: str = 'breakpoint_scan',
) -> tuple[Pool, ...]:
    """Split a sequence at breakpoint positions.
    
    Returns a tuple of Pools, one per segment (num_breakpoints + 1 total).
    
    Args:
        parent: Input sequence or pool
        num_breakpoints: Number of breakpoints
        positions: Valid breakpoint positions (overrides start/end/step_size)
        start: First valid breakpoint position (default: 1)
        end: Last valid breakpoint position (default: seq_length - 1)
        step_size: Step between valid positions (default: 1)
        min_spacing: Minimum spacing between consecutive breakpoints
        max_spacing: Maximum spacing between consecutive breakpoints
        mode: 'sequential' or 'random'
        name: Operation name
    
    Returns:
        Tuple of Pools, one per segment
    
    Example:
        >>> left, right = breakpoint_scan('ACGTACGT', num_breakpoints=1)
        >>> left, middle, right = breakpoint_scan('ACGTACGT', num_breakpoints=2)
    """
    from .from_seqs import from_seqs
    
    if isinstance(parent, str):
        parent = from_seqs([parent], mode='sequential', name=f'{name}_input')
    
    op = BreakpointScanOp(parent, num_breakpoints=num_breakpoints, 
                          positions=positions, start=start, end=end,
                          step_size=step_size, min_spacing=min_spacing,
                          max_spacing=max_spacing, mode=mode, name=name)
    
    # Return tuple of pools, one per output
    return tuple(Pool(operation=op, output_index=i) for i in range(op.num_outputs))
