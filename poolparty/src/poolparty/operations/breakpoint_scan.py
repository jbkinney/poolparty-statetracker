"""BreakpointScan operation - split sequences at breakpoint positions."""
from itertools import combinations
from math import comb
from numbers import Real
from ..types import Pool_type, Union, Sequence, ModeType, Optional, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np


class BreakpointScanOp(Operation):
    """Split a sequence at breakpoint positions."""
    factory_name = "breakpoint_scan"
    design_card_keys = ['breakpoints']
    
    @beartype
    def __init__(
        self,
        parent_pool: Pool_type,
        num_breakpoints: int = 1,
        positions: Optional[Sequence[int]] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step_size: int = 1,
        min_spacing: Optional[int] = None,
        max_spacing: Optional[int] = None,
        mode: ModeType = 'sequential',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        op_iteration_order: Real = 0,
    ) -> None:
        """Initialize BreakpointScanOp."""
        if num_breakpoints < 1:
            raise ValueError(f"num_breakpoints must be >= 1, got {num_breakpoints}")
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")
        self.num_breakpoints = num_breakpoints
        self.positions = list(positions) if positions is not None else None
        self.start = start
        self.end = end
        self.step_size = step_size
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self._mode = mode
        self.num_outputs = num_breakpoints + 1
        self._seq_length = parent_pool.seq_length
        self._valid_positions = None
        self._sequential_cache = None
        if mode == 'sequential':
            if self._seq_length is not None:
                num_states = self._build_caches()
            else:
                num_states = 1
        elif mode == 'hybrid':
            num_states = num_hybrid_states
        else:
            num_states = 1
        super().__init__(
            parent_pools=[parent_pool],
            num_states=num_states,
            mode=mode,
            seq_length=None,  # Variable output lengths
            name=name,
            iter_order=op_iteration_order,
        )
    
    def _is_valid_spacing(self, breakpoints) -> bool:
        """Check if breakpoints satisfy spacing constraints."""
        if len(breakpoints) < 2:
            return True
        for i in range(len(breakpoints) - 1):
            spacing = int(breakpoints[i + 1]) - int(breakpoints[i])
            if self.min_spacing is not None and spacing < self.min_spacing:
                return False
            if self.max_spacing is not None and spacing > self.max_spacing:
                return False
        return True
    
    @beartype
    def _build_caches(self) -> int:
        """Build caches for sequential enumeration."""
        if self._seq_length is None:
            return 1
        if self.positions is not None:
            self._valid_positions = [p for p in self.positions if 0 < p < self._seq_length]
        else:
            start = self.start if self.start is not None else 1
            end = self.end if self.end is not None else self._seq_length - 1
            self._valid_positions = list(range(start, end + 1, self.step_size))
        if len(self._valid_positions) < self.num_breakpoints:
            raise ValueError(
                f"Not enough valid positions ({len(self._valid_positions)}) "
                f"for {self.num_breakpoints} breakpoints"
            )
        all_combos = list(combinations(self._valid_positions, self.num_breakpoints))
        self._sequential_cache = [c for c in all_combos if self._is_valid_spacing(c)]
        if len(self._sequential_cache) == 0:
            raise ValueError(
                f"No valid breakpoint combinations after applying spacing constraints "
                f"(min_spacing={self.min_spacing}, max_spacing={self.max_spacing})"
            )
        return len(self._sequential_cache)
    
    def _random_breakpoints(self, seq_len: int, rng: np.random.Generator) -> tuple:
        """Generate random breakpoint positions."""
        if self._valid_positions is None:
            if self.positions is not None:
                self._valid_positions = [p for p in self.positions if 0 < p < seq_len]
            else:
                start = self.start if self.start is not None else 1
                end = self.end if self.end is not None else seq_len - 1
                self._valid_positions = list(range(start, end + 1, self.step_size))
        if len(self._valid_positions) < self.num_breakpoints:
            raise ValueError(
                f"Not enough valid positions ({len(self._valid_positions)}) "
                f"for {self.num_breakpoints} breakpoints"
            )
        max_attempts = 1000
        for _ in range(max_attempts):
            positions = tuple(sorted(rng.choice(
                self._valid_positions, 
                size=self.num_breakpoints, 
                replace=False
            )))
            if self._is_valid_spacing(positions):
                return positions
        raise RuntimeError("Could not find valid breakpoint positions after 1000 attempts")
    
    @beartype
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with breakpoint positions."""
        seq = parent_seqs[0]
        seq_len = len(seq)
        if self.mode in ('random', 'hybrid'):
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)")
            breakpoints = self._random_breakpoints(seq_len, rng)
        else:
            if self._sequential_cache is None:
                self._seq_length = seq_len
                self._build_caches()
                self.counter._num_states = len(self._sequential_cache)
            # Use state 0 when inactive (state is None)
            state = self.counter.state
            state = 0 if state is None else state
            breakpoints = self._sequential_cache[state % len(self._sequential_cache)]
        return {'breakpoints': breakpoints}
    
    @beartype
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Split sequence at breakpoints based on design card."""
        seq = parent_seqs[0]
        seq_len = len(seq)
        breakpoints = card['breakpoints']
        boundaries = [0] + list(breakpoints) + [seq_len]
        segments = [seq[boundaries[i]:boundaries[i+1]] for i in range(self.num_outputs)]
        result = {}
        for i, segment in enumerate(segments):
            result[f'seq_{i}'] = segment
        return result
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'num_breakpoints': self.num_breakpoints,
            'positions': self.positions,
            'start': self.start,
            'end': self.end,
            'step_size': self.step_size,
            'min_spacing': self.min_spacing,
            'max_spacing': self.max_spacing,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'op_iteration_order': self.iter_order,
        }


@beartype
def breakpoint_scan(
    parent: Union[Pool_type, str],
    num_breakpoints: int = 1,
    positions: Optional[Sequence[int]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step_size: int = 1,
    min_spacing: Optional[int] = None,
    max_spacing: Optional[int] = None,
    mode: ModeType = 'sequential',
    num_hybrid_states: Optional[int] = None,
    pool_iteration_order: Real = 0,
    op_iteration_order: Real = 0,
    op_name: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    synchronize_pools: bool = True,
) -> tuple:
    """Split a sequence at breakpoint positions."""
    from .from_seqs import from_seqs
    if isinstance(parent, str):
        parent = from_seqs([parent], mode='fixed')
    op = BreakpointScanOp(parent, num_breakpoints=num_breakpoints, 
                          positions=positions, start=start, end=end,
                          step_size=step_size, min_spacing=min_spacing,
                          max_spacing=max_spacing, mode=mode, 
                          num_hybrid_states=num_hybrid_states, name=op_name,
                          op_iteration_order=op_iteration_order)
    shared_counter = op.build_pool_counter(op.parent_pools)
    if synchronize_pools:
        pools = tuple(Pool(operation=op, output_index=i, counter=shared_counter) 
                      for i in range(op.num_outputs))
    else:
        pools = tuple(Pool(operation=op, output_index=i, counter=shared_counter.deepcopy()) 
                      for i in range(op.num_outputs))
    # Set iteration_order on all output pools
    for pool in pools:
        pool.iter_order = pool_iteration_order
    if names is not None:
        if len(names) != len(pools):
            raise ValueError(
                f"names length ({len(names)}) must match "
                f"num_outputs ({len(pools)})"
            )
        for pool, name in zip(pools, names):
            pool.name = name
    return pools
