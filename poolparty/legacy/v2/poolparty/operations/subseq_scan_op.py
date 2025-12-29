"""SubseqScan operation - extract subsequences at specified positions."""

from ..types import Union, Optional, ModeType, Sequence, beartype
from ..operation import Operation
from ..pool import Pool
from .from_seqs_op import from_seqs
import numpy as np


@beartype
class SubseqScanOp(Operation):
    """Extract subsequences from a parent sequence at specified positions.
    
    Supports multiple subsequence lengths and two mutually exclusive position interfaces:
    - Range-based: uses start, end, step_size to compute positions automatically
    - Position-based: uses explicit positions list with optional position_probs
    """
    design_card_keys = ['seq', 'position', 'subseq_length']
    
    #########################################################
    # Constructor 
    #########################################################
    
    def __init__(
        self,
        parent: Pool,
        subseq_length: Union[int, Sequence[int]],
        start: Optional[int] = None,
        end: Optional[int] = None,
        step_size: Optional[int] = None,
        positions: Optional[Sequence[int]] = None,
        position_probs: Optional[Sequence[float]] = None,
        subseq_length_probs: Optional[Sequence[float]] = None,
        mode: ModeType = 'random',
        name: Optional[str] = None,
        design_card_keys: Optional[Sequence[str]] = None,
    ):
        """Initialize SubseqScanOp.
        
        Args:
            parent: Parent Pool to extract subsequences from
            subseq_length: Width/length of each subsequence window. Can be a single
                int or a list of ints for multiple lengths.
            
            Range-based interface (mutually exclusive with positions):
                start: Starting position for first subsequence (default: 0)
                end: Ending position exclusive (default: len(parent))
                step_size: Step between adjacent subsequence positions (default: 1)
            
            Position-based interface (mutually exclusive with range-based):
                positions: List of explicit positions to extract at
                position_probs: Probabilities for weighted position sampling in random mode
            
            subseq_length_probs: Probabilities for weighted length sampling in random mode.
                Only valid when subseq_length is a list and mode='random'.
            
            mode: 'random' or 'sequential' (default: 'random')
            name: Optional name for this operation
            design_card_keys: Which design card keys to include
        
        Raises:
            ValueError: If both range-based and position-based parameters are provided
            ValueError: If position_probs is provided without positions
            ValueError: If position_probs length doesn't match positions length
            ValueError: If position_probs are provided with mode='sequential'
            ValueError: If subseq_length_probs are provided with mode='sequential'
            ValueError: If positions list is empty
            ValueError: If any position is out of valid bounds for all lengths
            ValueError: If subseq_length contains invalid values
        """
        # Normalize subseq_length to a list
        if isinstance(subseq_length, int):
            self._lengths = [subseq_length]
        else:
            self._lengths = list(subseq_length)
        
        # Validate lengths
        if len(self._lengths) == 0:
            raise ValueError("subseq_length must not be empty")
        for L in self._lengths:
            if L <= 0:
                raise ValueError(f"All subseq_length values must be > 0, got {L}")
        
        # Get parent sequence length
        parent_seq_length = parent.seq_length
        if parent_seq_length is not None:
            for L in self._lengths:
                if L > parent_seq_length:
                    raise ValueError(
                        f"subseq_length ({L}) cannot be greater than "
                        f"parent sequence length ({parent_seq_length})"
                    )
        
        # Determine which interface is being used
        range_params_provided = any(p is not None for p in [start, end, step_size])
        position_params_provided = positions is not None
        
        if range_params_provided and position_params_provided:
            raise ValueError(
                "Cannot specify both range-based parameters (start/end/step_size) "
                "and position-based parameters (positions). Choose one interface."
            )
        
        if position_probs is not None and positions is None:
            raise ValueError("position_probs requires positions to be specified")
        
        if mode == 'sequential' and position_probs is not None:
            raise ValueError(
                "Cannot specify position_probs with mode='sequential'. "
                "Sequential mode iterates through all positions deterministically."
            )
        
        if mode == 'sequential' and subseq_length_probs is not None:
            raise ValueError(
                "Cannot specify subseq_length_probs with mode='sequential'. "
                "Sequential mode iterates through all lengths deterministically."
            )
        
        # Validate and normalize subseq_length_probs
        if subseq_length_probs is not None:
            if len(subseq_length_probs) != len(self._lengths):
                raise ValueError(
                    f"subseq_length_probs length ({len(subseq_length_probs)}) must match "
                    f"number of lengths ({len(self._lengths)})"
                )
            probs = np.array(subseq_length_probs, dtype=float)
            if np.any(probs < 0):
                raise ValueError("subseq_length_probs must be non-negative")
            if probs.sum() <= 0:
                raise ValueError("subseq_length_probs must sum to > 0")
            self._length_probs = probs / probs.sum()  # normalize
        else:
            # Uniform distribution over lengths
            self._length_probs = np.ones(len(self._lengths)) / len(self._lengths)
        
        # Compute positions for each length
        # _positions_per_length[i] = list of valid positions for self._lengths[i]
        # _position_probs_per_length[i] = probability array for positions at length i
        self._positions_per_length: list[list[int]] = []
        self._position_probs_per_length: list[np.ndarray] = []
        
        if position_params_provided:
            # Position-based interface
            if len(positions) == 0:
                raise ValueError("positions must be a non-empty list")
            
            user_positions = list(positions)
            
            # Check for duplicates in user positions
            if len(user_positions) != len(set(user_positions)):
                raise ValueError("positions must not contain duplicates")
            
            # Validate position_probs length matches user positions
            if position_probs is not None:
                if len(position_probs) != len(user_positions):
                    raise ValueError(
                        f"position_probs length ({len(position_probs)}) must match "
                        f"positions length ({len(user_positions)})"
                    )
                # Validate probs are non-negative
                for p in position_probs:
                    if p < 0:
                        raise ValueError("position_probs must be non-negative")
            
            # Compute valid positions for each length
            for L in self._lengths:
                valid_positions = []
                for pos in user_positions:
                    if parent_seq_length is None or (pos >= 0 and pos + L <= parent_seq_length):
                        valid_positions.append(pos)
                
                if len(valid_positions) == 0:
                    raise ValueError(
                        f"No valid positions for subseq_length={L}. "
                        f"All positions are out of bounds."
                    )
                
                self._positions_per_length.append(valid_positions)
                
                # Handle position_probs for this length
                if position_probs is not None:
                    # Filter probs to match valid positions
                    valid_probs = []
                    for i, pos in enumerate(user_positions):
                        if pos in valid_positions:
                            valid_probs.append(position_probs[i])
                    probs = np.array(valid_probs, dtype=float)
                    if probs.sum() <= 0:
                        raise ValueError(f"position_probs for valid positions at length {L} must sum to > 0")
                    self._position_probs_per_length.append(probs / probs.sum())
                else:
                    # Uniform distribution
                    n = len(valid_positions)
                    self._position_probs_per_length.append(np.ones(n) / n)
        else:
            # Range-based interface
            start_val = start if start is not None else 0
            end_val = end if end is not None else parent_seq_length
            step_val = step_size if step_size is not None else 1
            
            if start_val < 0:
                raise ValueError(f"start must be >= 0, got {start_val}")
            if step_val <= 0:
                raise ValueError(f"step_size must be > 0, got {step_val}")
            
            if end_val is None:
                raise ValueError(
                    "end must be specified when parent has unknown sequence length"
                )
            
            # Clamp end to parent sequence length if known
            if parent_seq_length is not None:
                end_val = min(end_val, parent_seq_length)
            
            # Compute positions for each length
            for L in self._lengths:
                positions_for_length = list(range(start_val, end_val - L + 1, step_val))
                
                if len(positions_for_length) == 0:
                    raise ValueError(
                        f"Range [start={start_val}, end={end_val}) with subseq_length={L} "
                        f"and step_size={step_val} produces no valid positions"
                    )
                
                self._positions_per_length.append(positions_for_length)
                # Uniform probabilities for range-based
                n = len(positions_for_length)
                self._position_probs_per_length.append(np.ones(n) / n)
        
        # Compute cumulative states for sequential lookup
        # States are organized: all positions for length[0], then all positions for length[1], etc.
        self._states_per_length = [len(pos_list) for pos_list in self._positions_per_length]
        self._cumulative_states = []
        cumsum = 0
        for n in self._states_per_length:
            self._cumulative_states.append(cumsum)
            cumsum += n
        total_states = cumsum
        
        # Determine seq_length: fixed if single length, None if variable
        output_seq_length = self._lengths[0] if len(self._lengths) == 1 else None
        
        super().__init__(
            parent_pools=[parent],
            num_states=total_states,
            mode=mode,
            seq_length=output_seq_length,
            name=name,
            design_card_keys=design_card_keys,
        )
    
    #########################################################
    # Helper methods
    #########################################################
    
    def _find_length_for_state(self, global_state: int) -> tuple[int, int, int]:
        """Find which length bucket a global state belongs to.
        
        Returns:
            (length_idx, length, local_state) where local_state is the position index
        """
        for i, (cumulative, n_states) in enumerate(zip(self._cumulative_states, self._states_per_length)):
            if global_state < cumulative + n_states:
                return i, self._lengths[i], global_state - cumulative
        # Should not reach here if global_state < num_states
        raise ValueError(f"State {global_state} out of range")
    
    #########################################################
    # Results computation
    #########################################################
    
    def compute_results_row(self, input_strings: Sequence[str], sequential_state: int) -> dict:
        """Extract subsequence at position determined by mode/state.
        
        Args:
            input_strings: List containing the parent sequence
            sequential_state: Sequential state number (for sequential mode)
        
        Returns:
            Dict with 'seq', 'position', and 'subseq_length' keys
        """
        parent_seq = input_strings[0]
        
        if self.mode == 'random':
            # Sample length according to length_probs
            length_idx = int(self.rng.choice(len(self._lengths), p=self._length_probs))
            length = self._lengths[length_idx]
            
            # Sample position according to position_probs for this length
            positions = self._positions_per_length[length_idx]
            probs = self._position_probs_per_length[length_idx]
            pos_idx = int(self.rng.choice(len(positions), p=probs))
            position = positions[pos_idx]
        else:
            # Sequential mode: map global state to length and position
            state = sequential_state % self.num_states
            length_idx, length, pos_idx = self._find_length_for_state(state)
            position = self._positions_per_length[length_idx][pos_idx]
        
        # Extract subsequence
        seq = parent_seq[position:position + length]
        
        return {
            'seq': seq,
            'position': position,
            'subseq_length': length,
        }


#########################################################
# Public factory function
#########################################################

@beartype
def subseq_scan(
    parent: Union[Pool, str],
    subseq_length: Union[int, Sequence[int]],
    start: Optional[int] = None,
    end: Optional[int] = None,
    step_size: Optional[int] = None,
    positions: Optional[Sequence[int]] = None,
    position_probs: Optional[Sequence[float]] = None,
    subseq_length_probs: Optional[Sequence[float]] = None,
    mode: ModeType = 'random',
    name: str = 'subseq_scan',
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Create a Pool that extracts subsequences at specified positions.
    
    Supports multiple subsequence lengths and two mutually exclusive position interfaces:
    
    **Range-based interface:**
        Systematically extracts subsequences using start, end, and step_size.
        
        Given L=len(parent), W=subseq_length:
        - Extracts windows where [pos, pos+W) fits within [start, end)
        - For multiple lengths, computes valid positions for each length
        - Defaults: start=0, end=L, step_size=1
    
    **Position-based interface:**
        Directly specifies explicit extraction positions with optional probability weights.
        
        - Parameters: positions (required), position_probs (optional)
        - For multiple lengths, positions must be valid for at least one length
        - Invalid (position, length) pairs are skipped
    
    **Sequential mode iteration order:**
        All positions for length[0], then all positions for length[1], etc.
    
    Args:
        parent: Input sequence (string or Pool) to extract subsequences from
        subseq_length: Width of each subsequence window. Can be a single int or a list.
        start: Starting position for first subsequence (default: 0)
        end: Ending position exclusive (default: len(parent))
        step_size: Step between adjacent subsequences (default: 1)
        positions: List of explicit positions to extract at
        position_probs: Probabilities for weighted position sampling (random mode only)
        subseq_length_probs: Probabilities for weighted length sampling (random mode only)
        mode: 'random' or 'sequential' (default: 'random')
        name: Name for this operation
        design_card_keys: Which design card keys to include
    
    Returns:
        Pool: A pool that generates subsequences.
    
    Examples:
        >>> # Single length, range-based
        >>> pool = subseq_scan('ACGTACGTACGT', subseq_length=4, mode='sequential')
        >>> pool.operation.num_states
        9  # Positions 0-8
        
        >>> # Multiple lengths
        >>> pool = subseq_scan('ACGTACGT', subseq_length=[3, 5], mode='sequential')
        >>> pool.operation.num_states
        10  # 6 positions for len=3, 4 positions for len=5
        
        >>> # Weighted length sampling
        >>> pool = subseq_scan('ACGTACGT', subseq_length=[3, 5], 
        ...                    subseq_length_probs=[0.8, 0.2], mode='random')
    
    Raises:
        ValueError: If both range-based and position-based parameters are provided
        ValueError: If positions is empty
        ValueError: If all subseq_length values are greater than sequence length
        ValueError: If any position is out of bounds for all lengths
    """
    # Convert string to Pool if needed
    if isinstance(parent, str):
        parent = from_seqs([parent], design_card_keys=[])
    
    return Pool(operation=SubseqScanOp(
        parent=parent,
        subseq_length=subseq_length,
        start=start,
        end=end,
        step_size=step_size,
        positions=positions,
        position_probs=position_probs,
        subseq_length_probs=subseq_length_probs,
        mode=mode,
        name=name,
        design_card_keys=design_card_keys,
    ))
