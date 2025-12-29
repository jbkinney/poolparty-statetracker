"""Subseq operation - extract subsequences from a sequence."""

from __future__ import annotations
from typing import List, Optional, Union, TYPE_CHECKING

from ..operation import Operation

if TYPE_CHECKING:
    from ..pool import Pool


class SubseqOp(Operation):
    """Extract fixed-width subsequences from a parent sequence.
    
    This is a transformer operation - it has one parent pool.
    Extracts windows of a fixed width at specified positions.
    """
    op_name = 'subseq'
    
    def __init__(
        self,
        parent: 'Pool',
        width: int,
        positions: list[int],
        mode: str,
        name: Optional[str] = None,
    ):
        """Initialize SubseqOp.
        
        Args:
            parent: The input Pool to extract from
            width: Width of each subsequence window
            positions: List of positions to extract at
            mode: Either 'random' or 'sequential'
            name: Optional name for this operation
        """
        self.width = width
        self.positions = positions
        
        # Initialize base class attributes
        super().__init__(
            parent_pools=[parent],
            num_states=len(positions),
            mode=mode,
            seq_length=width,
            name=name,
        )
    
    def compute_seq(
        self, 
        input_strings: list[str], 
        state: int
    ) -> str:
        """Extract subsequence at the position determined by state.
        
        Args:
            input_strings: List containing the parent sequence
            state: Internal state number
        
        Returns:
            Subsequence
        """
        seq = input_strings[0]
        
        # Wrap state within valid range
        idx = state % len(self.positions)
        pos = self.positions[idx]
        
        # Extract subsequence
        return seq[pos:pos + self.width]


def subseq_op(
    seq: Union['Pool', str],
    width: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step_size: Optional[int] = None,
    positions: Optional[list[int]] = None,
    mode: str = 'random',
    name: Optional[str] = None,
) -> 'Pool':
    """Extract fixed-width subsequences from a sequence.
    
    Supports two mutually exclusive interfaces:
    
    **Range-based interface:**
        Systematically extracts subsequences using start, end, and step_size.
        
        Given L=len(seq), W=width:
        - Extracts windows where [pos, pos+W) fits within [start, end)
        - num_internal_states = len(range(start, min(end, L) - W + 1, step_size))
        - Defaults: start=0, end=L, step_size=1
    
    **Position-based interface:**
        Directly specifies explicit extraction positions.
        
        - Parameters: positions (required)
        - num_internal_states = len(positions)
    
    Args:
        seq: Input sequence (string or Pool) to extract subsequences from
        width: Width of each subsequence window
        start: Starting position for first subsequence (default: 0)
        end: Ending position (exclusive). Default: len(seq)
        step_size: Step between adjacent subsequences (default: 1)
        positions: List of explicit positions to extract at
        mode: Either 'random' or 'sequential' (default: 'random')
        name: Optional name for this pool
    
    Returns:
        A Pool that generates subsequences.
    
    Example:
        >>> pool = subseq('ACGTACGTACGT', width=4, mode='sequential')
        >>> pool.operation.num_states
        9  # Positions 0-8
        >>> seqs = pool.generate_library(num_complete_iterations=1)
        >>> seqs[0]
        'ACGT'
    
    Raises:
        ValueError: If both range-based and position-based parameters are provided
        ValueError: If positions is empty
        ValueError: If width is longer than sequence
        ValueError: If any position is out of bounds
    """
    # Import here to avoid circular imports
    from ..pool import Pool
    from .from_seqs_op import from_seqs_op
    
    # If seq is a string, wrap it in from_seqs first
    if isinstance(seq, str):
        parent = from_seqs_op([seq])
    else:
        parent = seq
    
    # Determine which interface is being used
    range_params_provided = any(p is not None for p in [start, end, step_size])
    position_params_provided = positions is not None
    
    if range_params_provided and position_params_provided:
        raise ValueError(
            "Cannot specify both range-based parameters (start/end/step_size) "
            "and position-based parameters (positions). Choose one interface."
        )
    
    # Get sequence length
    seq_len = parent.seq_length
    
    # Validate width
    if width <= 0:
        raise ValueError(f"width must be > 0, got {width}")
    if width > seq_len:
        raise ValueError(
            f"width ({width}) cannot be longer than sequence length ({seq_len})"
        )
    
    if position_params_provided:
        # Position-based interface
        if not positions or len(positions) == 0:
            raise ValueError("positions must be a non-empty list")
        
        # Validate positions
        for pos in positions:
            if pos < 0 or pos + width > seq_len:
                raise ValueError(
                    f"Position {pos} is invalid: subsequence window "
                    f"[{pos}, {pos + width}) must fit within "
                    f"sequence of length {seq_len}"
                )
        
        if len(positions) != len(set(positions)):
            raise ValueError("positions must not contain duplicates")
        
        computed_positions = list(positions)
    else:
        # Range-based interface
        start_val = start if start is not None else 0
        end_val = end if end is not None else seq_len
        step_val = step_size if step_size is not None else 1
        
        # Clamp end to sequence length
        end_val = min(end_val, seq_len)
        
        # Validate
        if start_val < 0:
            raise ValueError(f"start must be >= 0, got {start_val}")
        if step_val <= 0:
            raise ValueError(f"step_size must be > 0, got {step_val}")
        
        # Compute positions
        computed_positions = list(range(start_val, end_val - width + 1, step_val))
        
        if len(computed_positions) == 0:
            raise ValueError(
                f"Range [start={start_val}, end={end_val}) with width={width} "
                f"and step_size={step_val} produces no valid positions"
            )
    
    return Pool(
        operation=SubseqOp(parent, width, computed_positions, mode=mode, name=name),
    )
