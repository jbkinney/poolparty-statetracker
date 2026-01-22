"""BreakpointScan operation - split sequences at breakpoint positions."""
from itertools import combinations
from ..types import Union, Sequence, ModeType, Optional, Integral, Real, PositionsType, beartype
from ..seq_utils import validate_positions
from ..operation import Operation
from ..pool import Pool
import numpy as np


@beartype
def breakpoint_scan(
    pool: Union[Pool, str],
    num_breakpoints: Integral,
    positions: PositionsType = None,
    min_spacing: Optional[Integral] = None,
    max_spacing: Optional[Integral] = None,
    seq_name_prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_states: Optional[Integral] = None,
    op_name: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None
) -> tuple[Pool, ...]:
    """
    Split a sequence at specified breakpoints and return each segment as a separate Pool.

    Parameters
    ----------
    pool : Union[Pool, str]
        Input Pool or sequence string to be split.
    num_breakpoints : Integral
        Number of breakpoints to introduce in the sequence.
    positions : PositionsType, default=None
        Specific breakpoint positions (indices) or None for all possible positions.
        Boundary positions are allowed by default.
    min_spacing : Optional[Integral], default=None
        Minimum allowed spacing between consecutive breakpoints.
    max_spacing : Optional[Integral], default=None
        Maximum allowed spacing between consecutive breakpoints.
    mode : ModeType, default='random'
        Breakpoint selection mode: 'random' or 'sequential'.
    num_states : Optional[Integral], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    op_name : Optional[str], default=None
        Name for the underlying BreakpointScan operation.
    names : Optional[Sequence[str]], default=None
        List of names for the resulting Pools (length=num_breakpoints+1).
    iter_order : Optional[Real], default=None
        Iteration order priority for each resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operation.

    Returns
    -------
    tuple[Pool, ...]
        Tuple of Pools, each yielding a segment of the original sequence.
    """
    from ..fixed_ops.from_seq import from_seq
    if names is None:
        names = [None] * (num_breakpoints + 1)
    elif len(names) != num_breakpoints + 1:
        raise ValueError(f"({len(names)=}) must match ({num_breakpoints + 1=}).")
    pool = from_seq(pool) if isinstance(pool, str) else pool
    op = BreakpointScanOp(
        pool,
        num_breakpoints=num_breakpoints,
        positions=positions,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        seq_name_prefix=seq_name_prefix,
        mode=mode,
        num_states=num_states,
        name=op_name,
        iter_order=op_iter_order,
    )
    shared_state = op.build_pool_counter(op.parent_pools)
    result_pools = tuple(
        Pool(
            operation=op,
            output_index=i,
            state=shared_state,
            iter_order=iter_order,
            name=names[i],
        )
        for i in range(op.num_outputs)
    )
    return result_pools


@beartype
class BreakpointScanOp(Operation):
    """Split a sequence at breakpoint positions."""
    factory_name = "breakpoint_scan"
    design_card_keys = ['breakpoints']
    
    def __init__(
        self,
        parent_pool: Pool,
        num_breakpoints: Integral = 1,
        positions: PositionsType = None,
        min_spacing: Optional[Integral] = None,
        max_spacing: Optional[Integral] = None,
        seq_name_prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[Integral] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        """Initialize BreakpointScanOp."""
        if num_breakpoints < 1:
            raise ValueError(f"num_breakpoints must be >= 1, got {num_breakpoints}")
        self.num_breakpoints = num_breakpoints
        self._positions = positions
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
        elif mode == 'random':
            num_states = num_states if num_states is not None else 1
        else:
            num_states = 1
        super().__init__(
            parent_pools=[parent_pool],
            num_values=num_states,
            mode=mode,
            seq_length=None,  # Variable output lengths
            name=name,
            iter_order=iter_order,
            seq_name_prefix=seq_name_prefix,
        )
    
    def _is_valid_spacing(self, breakpoints: Sequence[Integral]) -> bool:
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
    
    def _build_caches(self) -> int:
        """Build caches for sequential enumeration."""
        if self._seq_length is None:
            return 1
        self._valid_positions = validate_positions(
            self._positions,
            max_position=self._seq_length,
            min_position=0,
        )
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
            self._valid_positions = validate_positions(
                self._positions,
                max_position=seq_len,
                min_position=0,
            )
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
    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return design card with breakpoint positions (logical)."""
        seq = parent_seqs[0]
        seq_len = self._get_length_without_markers(seq)
        if self.mode == 'random':
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG - use Party.generate(seed=...)")
            breakpoints = self._random_breakpoints(seq_len, rng)
        else:
            if self._sequential_cache is None:
                self._seq_length = seq_len
                self._build_caches()
                self.state._num_values = len(self._sequential_cache)
            # Use state 0 when inactive (state is None)
            state = self.state.value
            state = 0 if state is None else state
            breakpoints = self._sequential_cache[state % len(self._sequential_cache)]
        return {'breakpoints': breakpoints}
    
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Split sequence at breakpoints based on design card."""
        seq = parent_seqs[0]
        raw_len = len(seq)
        logical_breakpoints = card['breakpoints']  # Logical positions (in marker-free seq)
        nonmarker_positions = self._get_nonmarker_positions(seq)
        seq_len = len(nonmarker_positions)
        
        # Translate logical breakpoints to raw positions
        # Logical position k means "split before the k-th char in marker-free sequence"
        raw_breakpoints = []
        for logical_pos in logical_breakpoints:
            if logical_pos >= seq_len:
                raw_breakpoints.append(raw_len)
            else:
                raw_breakpoints.append(nonmarker_positions[logical_pos])
        
        boundaries = [0] + raw_breakpoints + [raw_len]
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
            'positions': self._positions,
            'min_spacing': self.min_spacing,
            'max_spacing': self.max_spacing,
            'seq_name_prefix': self.name_prefix,
            'mode': self.mode,
            'num_states': self.num_values if self.mode == 'random' and self.num_values > 1 else None,
            'name': None,
            'iter_order': self.iter_order,
        }

