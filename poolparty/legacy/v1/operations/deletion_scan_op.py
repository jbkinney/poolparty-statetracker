"""DeletionScan operation - scan deletions across a sequence."""

from __future__ import annotations
from typing import List, Optional, Union, TYPE_CHECKING

from ..operation import Operation

if TYPE_CHECKING:
    from ..pool import Pool


class DeletionScanOp(Operation):
    """Scan deletions across a parent sequence.
    
    This is a transformer operation - it has one parent pool.
    Systematically removes or marks segments from the sequence.
    """
    op_name = 'deletion_scan'
    
    def __init__(
        self,
        parent: 'Pool',
        deletion_size: int,
        positions: list[int],
        mark_changes: bool,
        deletion_character: str,
        mode: str,
        name: Optional[str] = None,
    ):
        """Initialize DeletionScanOp."""
        self.deletion_size = deletion_size
        self.positions = positions
        self.mark_changes = mark_changes
        self.deletion_character = deletion_character
        
        # Compute seq_length based on mark_changes and parent's seq_length
        if mark_changes:
            # Length unchanged when marking deletions
            seq_length = parent.seq_length
        elif parent.seq_length is not None:
            # Fixed length parent with actual deletion
            seq_length = parent.seq_length - deletion_size
        else:
            # Variable length parent
            seq_length = None
        
        # Initialize base class attributes
        super().__init__(
            parent_pools=[parent],
            num_states=len(positions),
            mode=mode,
            seq_length=seq_length,
            name=name,
        )
    
    def compute_seq(
        self, 
        input_strings: Sequence[str], 
        state: int
    ) -> str:
        """Compute sequence with deletion at the position determined by state.
        
        Args:
            input_strings: List containing the parent sequence
            state: Internal state number
        
        Returns:
            Sequence with deletion
        """
        base_seq = input_strings[0]
        
        # Get deletion position
        idx = state % len(self.positions)
        pos = self.positions[idx]
        
        # Apply deletion
        if self.mark_changes:
            return (
                base_seq[:pos] + 
                self.deletion_character * self.deletion_size + 
                base_seq[pos + self.deletion_size:]
            )
        else:
            return base_seq[:pos] + base_seq[pos + self.deletion_size:]


def deletion_scan_op(
    seq: Union['Pool', str],
    deletion_size: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step_size: Optional[int] = None,
    positions: Optional[list[int]] = None,
    mark_changes: bool = True,
    deletion_character: str = '-',
    mode: str = 'random',
    name: Optional[str] = None,
) -> 'Pool':
    """Scan deletions across a sequence.
    
    Performs deletion scanning mutagenesis by systematically removing or marking
    segments from the sequence.
    
    Args:
        seq: Input sequence (string or Pool) to delete from
        deletion_size: Number of positions to delete
        start: Starting position for first deletion (default: 0)
        end: Ending position (default: len(seq))
        step_size: Step between adjacent deletions (default: 1)
        positions: List of explicit positions to delete at
        mark_changes: If True, mark deletions with deletion_character; if False,
            actually remove them. Default: True
        deletion_character: Character to mark deletions. Default: '-'
        mode: Either 'random' or 'sequential'. Default: 'random'
        name: Optional name for this pool
    
    Returns:
        A Pool that generates deletion variants.
    
    Example:
        >>> pool = deletion_scan('ACGTACGT', deletion_size=2, mode='sequential')
        >>> pool.operation.num_states
        7  # Positions 0-6
        >>> seqs = pool.generate_library(num_complete_iterations=1)
        >>> seqs[0]
        '--GTACGT'  # Deletion at position 0
    
    Raises:
        ValueError: If deletion_size > sequence length
        ValueError: If both range-based and position-based parameters are provided
    """
    # Import here to avoid circular imports
    from ..pool import Pool
    from .from_seqs_op import from_seqs_op
    
    # If seq is a string, wrap it in from_seqs first
    if isinstance(seq, str):
        parent = from_seqs_op([seq])
        seq_len = len(seq)
    else:
        parent = seq
        seq_len = parent.seq_length
    
    # Validate deletion_size
    if deletion_size <= 0:
        raise ValueError(f"deletion_size must be > 0, got {deletion_size}")
    if deletion_size > seq_len:
        raise ValueError(
            f"deletion_size ({deletion_size}) cannot be larger than "
            f"sequence length ({seq_len})"
        )
    
    # Determine positions
    range_params_provided = any(p is not None for p in [start, end, step_size])
    position_params_provided = positions is not None
    
    if range_params_provided and position_params_provided:
        raise ValueError(
            "Cannot specify both range-based parameters (start/end/step_size) "
            "and position-based parameters (positions)."
        )
    
    if position_params_provided:
        if not positions:
            raise ValueError("positions must be a non-empty list")
        for pos in positions:
            if pos < 0 or pos + deletion_size > seq_len:
                raise ValueError(
                    f"Position {pos} is invalid: deletion window "
                    f"[{pos}, {pos + deletion_size}) must fit within sequence"
                )
        computed_positions = list(positions)
    else:
        start_val = start if start is not None else 0
        end_val = end if end is not None else seq_len
        step_val = step_size if step_size is not None else 1
        
        end_val = min(end_val, seq_len)
        computed_positions = list(range(start_val, end_val - deletion_size + 1, step_val))
        
        if len(computed_positions) == 0:
            raise ValueError("No valid deletion positions for given parameters")
    
    return Pool(
        operation=DeletionScanOp(
            parent=parent,
            deletion_size=deletion_size,
            positions=computed_positions,
            mark_changes=mark_changes,
            deletion_character=deletion_character,
            mode=mode,
            name=name,
        ),
    )
