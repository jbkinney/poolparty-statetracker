"""MultiinsertionScan operation - enumerate spacing combinations for multiple inserts."""

from __future__ import annotations
import itertools
from typing import List, Optional, Union, Tuple, Literal, TYPE_CHECKING

from ..operation import Operation

if TYPE_CHECKING:
    from ..pool import Pool


class MultiinsertionScanOp(Operation):
    """Enumerate spacing combinations for multiple inserts relative to an anchor.
    
    This is a transformer operation - it has parent pools (background + inserts).
    Pre-computes all valid combinations for a finite state space.
    """
    op_name = 'multiinsertion_scan'
    
    def __init__(
        self,
        background_parent: 'Pool',
        insert_parents: list['Pool'],
        anchor_pos: int,
        valid_combinations: list[tuple[int, ...]],
        insert_lengths: list[int],
        insert_or_overwrite: str,
        mark_changes: bool,
        result_length: int,
        mode: str,
        name: Optional[str] = None,
    ):
        """Initialize MultiinsertionScanOp."""
        self.anchor_pos = anchor_pos
        self.valid_combinations = valid_combinations
        self.insert_lengths = insert_lengths
        self.insert_or_overwrite = insert_or_overwrite
        self.mark_changes = mark_changes
        
        # Initialize base class attributes
        super().__init__(
            parent_pools=[background_parent] + insert_parents,
            num_states=len(valid_combinations),
            mode=mode,
            seq_length=result_length,
            name=name,
        )
    
    def compute_seq(
        self, 
        input_strings: list[str], 
        state: int
    ) -> str:
        """Compute sequence with inserts at the positions determined by state.
        
        Args:
            input_strings: List containing [background, insert_0, insert_1, ...]
            state: Internal state number
        
        Returns:
            Sequence with inserts
        """
        background = input_strings[0]
        inserts = input_strings[1:]
        
        # Get positions for this state
        idx = state % len(self.valid_combinations)
        positions = self.valid_combinations[idx]
        
        # Apply mark_changes if requested
        if self.mark_changes:
            inserts = [ins.swapcase() for ins in inserts]
        
        # Apply insertions (must apply from right to left to preserve positions)
        result = background
        for i in reversed(range(len(inserts))):
            pos = positions[i]
            insert = inserts[i]
            
            if self.insert_or_overwrite == 'insert':
                result = result[:pos] + insert + result[pos:]
            else:  # 'overwrite'
                result = result[:pos] + insert + result[pos + len(insert):]
        
        return result


def multiinsertion_scan_op(
    background_seq: Union['Pool', str],
    insert_seqs: list[Union['Pool', str]],
    anchor_pos: int,
    insert_ranges: Optional[list[tuple[int, int]]] = None,
    insert_ranges_with_step: Optional[list[tuple[int, int, int]]] = None,
    insert_positions: Optional[list[list[int]]] = None,
    min_spacing: int = 0,
    enforce_order: bool = True,
    insert_or_overwrite: Literal['insert', 'overwrite'] = 'overwrite',
    mark_changes: bool = False,
    mode: str = 'sequential',
    name: Optional[str] = None,
) -> 'Pool':
    """Enumerate spacing combinations for multiple inserts relative to an anchor.
    
    Pre-computes all valid (non-overlapping, properly ordered) combinations at
    construction, giving a finite state space for sequential enumeration.
    
    Args:
        background_seq: Background sequence (string or Pool) to insert into
        insert_seqs: List of sequences (string or Pool) to insert
        anchor_pos: Reference position (0-indexed) for distance calculations
        insert_ranges: For each insert, (start, end) relative to anchor.
            The insert can be placed anywhere in [anchor+start, anchor+end).
        insert_ranges_with_step: For each insert, (start, end, step) relative to anchor.
            Like insert_ranges but with explicit step size.
        insert_positions: For each insert, explicit positions relative to anchor.
            [d1, d2, ...] means insert at anchor+d1, anchor+d2, etc.
        min_spacing: Minimum gap (bp) between adjacent inserts. Default: 0
        enforce_order: If True, inserts must appear in list order. Default: True
        insert_or_overwrite: 'overwrite' (default) or 'insert'
        mark_changes: If True, apply swapcase() to inserted regions. Default: False
        mode: 'sequential' (default) or 'random'
        name: Optional pool name
    
    Returns:
        A Pool that generates insertion combinations.
    
    Example:
        >>> pool = multiinsertion_scan(
        ...     'NNNNNNNNNN',
        ...     insert_seqs=['AA', 'BB'],
        ...     anchor_pos=5,
        ...     insert_ranges=[(-4, 0), (1, 5)],
        ...     mode='sequential'
        ... )
    
    Raises:
        ValueError: If no valid combinations exist
        ValueError: If insert_seqs is empty
    """
    # Import here to avoid circular imports
    from ..pool import Pool
    from .from_seqs_op import from_seqs_op
    
    # Validate inputs
    if not insert_seqs:
        raise ValueError("insert_seqs must be non-empty")
    
    n_inserts = len(insert_seqs)
    
    # Wrap strings in from_seqs
    if isinstance(background_seq, str):
        background_parent = from_seqs_op([background_seq])
        background_len = len(background_seq)
    else:
        background_parent = background_seq
        background_len = background_seq.seq_length
    
    insert_parents = []
    insert_lengths = []
    for seq in insert_seqs:
        if isinstance(seq, str):
            insert_parents.append(from_seqs_op([seq]))
            insert_lengths.append(len(seq))
        else:
            insert_parents.append(seq)
            insert_lengths.append(seq.seq_length)
    
    # Determine position options for each insert
    position_options = []
    
    for i in range(n_inserts):
        if insert_positions is not None and i < len(insert_positions) and insert_positions[i] is not None:
            # Explicit positions
            positions = [anchor_pos + d for d in insert_positions[i]]
        elif insert_ranges_with_step is not None and i < len(insert_ranges_with_step) and insert_ranges_with_step[i] is not None:
            # Range with step
            start, end, step = insert_ranges_with_step[i]
            positions = list(range(anchor_pos + start, anchor_pos + end, step))
        elif insert_ranges is not None and i < len(insert_ranges) and insert_ranges[i] is not None:
            # Range without step (step=1)
            start, end = insert_ranges[i]
            positions = list(range(anchor_pos + start, anchor_pos + end))
        else:
            raise ValueError(f"No position specification for insert {i}")
        
        # Filter to valid positions based on mode
        valid_positions = []
        for pos in positions:
            if insert_or_overwrite == 'overwrite':
                if 0 <= pos and pos + insert_lengths[i] <= background_len:
                    valid_positions.append(pos)
            else:  # 'insert'
                if 0 <= pos <= background_len:
                    valid_positions.append(pos)
        
        if not valid_positions:
            raise ValueError(f"No valid positions for insert {i}")
        
        position_options.append(valid_positions)
    
    # Generate all combinations and filter for validity
    all_combinations = list(itertools.product(*position_options))
    
    valid_combinations = []
    for combo in all_combinations:
        # Check order constraint
        if enforce_order:
            sorted_positions = sorted(enumerate(combo), key=lambda x: x[1])
            if [p[0] for p in sorted_positions] != list(range(n_inserts)):
                continue
        
        # Check non-overlap and min_spacing
        valid = True
        for i in range(n_inserts):
            for j in range(i + 1, n_inserts):
                pos_i, pos_j = combo[i], combo[j]
                len_i = insert_lengths[i]
                
                # Determine end of insert i
                if insert_or_overwrite == 'overwrite':
                    end_i = pos_i + len_i
                else:
                    end_i = pos_i + len_i
                
                # Check if they overlap or violate min_spacing
                if pos_i < pos_j:
                    if end_i + min_spacing > pos_j:
                        valid = False
                        break
                else:
                    end_j = pos_j + insert_lengths[j]
                    if end_j + min_spacing > pos_i:
                        valid = False
                        break
            
            if not valid:
                break
        
        if valid:
            valid_combinations.append(combo)
    
    if not valid_combinations:
        raise ValueError("No valid combinations for given parameters")
    
    # Calculate result length
    if insert_or_overwrite == 'insert':
        result_length = background_len + sum(insert_lengths)
    else:
        result_length = background_len
    
    return Pool(
        operation=MultiinsertionScanOp(
            background_parent=background_parent,
            insert_parents=insert_parents,
            anchor_pos=anchor_pos,
            valid_combinations=valid_combinations,
            insert_lengths=insert_lengths,
            insert_or_overwrite=insert_or_overwrite,
            mark_changes=mark_changes,
            result_length=result_length,
            mode=mode,
            name=name,
        ),
    )
