"""InsertionScan operation - scan insertions across a sequence."""

from __future__ import annotations
from typing import List, Optional, Union, Literal, TYPE_CHECKING

from ..operation import Operation

if TYPE_CHECKING:
    from ..pool import Pool


class InsertionScanOp(Operation):
    """Scan insertions across a parent sequence.
    
    This is a transformer operation - it has parent pools (background and insert).
    Systematically inserts or overwrites with a given sequence.
    """
    op_name = 'insertion_scan'
    
    def __init__(
        self,
        background_parent: 'Pool',
        insert_parent: 'Pool',
        positions: list[int],
        insert_or_overwrite: str,
        mark_changes: bool,
        background_len: int,
        insert_len: int,
        mode: str,
        name: Optional[str] = None,
    ):
        """Initialize InsertionScanOp."""
        self.positions = positions
        self.insert_or_overwrite = insert_or_overwrite
        self.mark_changes = mark_changes
        self._background_len = background_len
        self._insert_len = insert_len
        
        # Compute seq_length based on mode
        if insert_or_overwrite == 'insert':
            seq_length = background_len + insert_len
        else:  # 'overwrite'
            seq_length = background_len
        
        # Initialize base class attributes
        super().__init__(
            parent_pools=[background_parent, insert_parent],
            num_states=len(positions),
            mode=mode,
            seq_length=seq_length,
            name=name,
        )
    
    def compute_seq(
        self, 
        input_strings: list[str], 
        state: int
    ) -> str:
        """Compute sequence with insertion at the position determined by state.
        
        Args:
            input_strings: List containing [background_seq, insert_seq]
            state: Internal state number
        
        Returns:
            Sequence with insertion
        """
        background = input_strings[0]
        insert = input_strings[1]
        
        # Get insertion position
        idx = state % len(self.positions)
        pos = self.positions[idx]
        
        # Apply mark_changes if requested
        if self.mark_changes:
            insert = insert.swapcase()
        
        # Apply insertion
        if self.insert_or_overwrite == 'insert':
            return background[:pos] + insert + background[pos:]
        else:  # 'overwrite'
            return background[:pos] + insert + background[pos + len(insert):]


def insertion_scan_op(
    background_seq: Union['Pool', str],
    insert_seq: Union['Pool', str],
    start: Optional[int] = None,
    end: Optional[int] = None,
    step_size: Optional[int] = None,
    positions: Optional[list[int]] = None,
    insert_or_overwrite: Literal['insert', 'overwrite'] = 'overwrite',
    mark_changes: bool = False,
    mode: str = 'random',
    name: Optional[str] = None,
) -> 'Pool':
    """Scan insertions across a sequence.
    
    Performs scanning mutagenesis by either overwriting positions in the background
    sequence or inserting between positions.
    
    Args:
        background_seq: Background sequence (string or Pool) to scan across
        insert_seq: Sequence to insert or overwrite with (string or Pool)
        start: Starting position for first insertion (default: 0)
        end: Ending position (default: len(background_seq))
        step_size: Step between adjacent insertions (default: 1)
        positions: List of explicit positions to scan
        insert_or_overwrite: 'overwrite' replaces characters, 'insert' adds between.
            Default: 'overwrite'
        mark_changes: If True, apply swapcase() to inserted sequence. Default: False
        mode: Either 'random' or 'sequential'. Default: 'random'
        name: Optional name for this pool
    
    Returns:
        A Pool that generates insertion variants.
    
    Example:
        >>> pool = insertion_scan('AAAA', 'XX', mode='sequential')
        >>> pool.operation.num_states
        3  # Positions 0, 1, 2
        >>> seqs = pool.generate_library(num_complete_iterations=1)
        >>> seqs[0]
        'XXAA'  # Overwrite at position 0
    
    Raises:
        ValueError: If insert_seq is longer than background in overwrite mode
        ValueError: If both range-based and position-based parameters are provided
    """
    # Import here to avoid circular imports
    from ..pool import Pool
    from .from_seqs_op import from_seqs_op
    
    # Wrap strings in from_seqs
    if isinstance(background_seq, str):
        background_parent = from_seqs_op([background_seq])
        background_len = len(background_seq)
    else:
        background_parent = background_seq
        background_len = background_seq.seq_length
    
    if isinstance(insert_seq, str):
        insert_parent = from_seqs_op([insert_seq])
        insert_len = len(insert_seq)
    else:
        insert_parent = insert_seq
        insert_len = insert_seq.seq_length
    
    # Validate insert_or_overwrite
    if insert_or_overwrite not in ('insert', 'overwrite'):
        raise ValueError(
            f"insert_or_overwrite must be 'insert' or 'overwrite', "
            f"got '{insert_or_overwrite}'"
        )
    
    # Validate lengths for overwrite mode
    if insert_or_overwrite == 'overwrite' and insert_len > background_len:
        raise ValueError(
            f"In overwrite mode, insert_seq length ({insert_len}) cannot exceed "
            f"background_seq length ({background_len})"
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
        # Validate positions based on mode
        for pos in positions:
            if insert_or_overwrite == 'overwrite':
                if pos < 0 or pos + insert_len > background_len:
                    raise ValueError(
                        f"Position {pos} is invalid for overwrite mode"
                    )
            else:  # 'insert'
                if pos < 0 or pos > background_len:
                    raise ValueError(
                        f"Position {pos} is invalid for insert mode"
                    )
        computed_positions = list(positions)
    else:
        start_val = start if start is not None else 0
        end_val = end if end is not None else background_len
        step_val = step_size if step_size is not None else 1
        
        if insert_or_overwrite == 'overwrite':
            end_val = min(end_val, background_len)
            computed_positions = list(range(start_val, end_val - insert_len + 1, step_val))
        else:  # 'insert'
            end_val = min(end_val, background_len)
            computed_positions = list(range(start_val, end_val + 1, step_val))
        
        if len(computed_positions) == 0:
            raise ValueError("No valid insertion positions for given parameters")
    
    return Pool(
        operation=InsertionScanOp(
            background_parent=background_parent,
            insert_parent=insert_parent,
            positions=computed_positions,
            insert_or_overwrite=insert_or_overwrite,
            mark_changes=mark_changes,
            background_len=background_len,
            insert_len=insert_len,
            mode=mode,
            name=name,
        ),
    )
