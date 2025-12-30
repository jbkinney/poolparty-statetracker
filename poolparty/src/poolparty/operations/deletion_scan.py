"""DeletionScan - delete segments from background at scanning positions."""
from numbers import Real
from ..types import Pool_type, Union, ModeType, Optional, beartype
from ..pool import Pool


@beartype
def deletion_scan(
    bg_pool: Union[Pool_type, str],
    deletion_length: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step_size: int = 1,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[int] = None,
    pool_iteration_order: Real = 0,
    op_iteration_order: Real = 0,
    deletion_marker: Optional[str] = '-',
    spacer_str: str = '',
    op_name: Optional[str] = None,
    name: Optional[str] = None,
) -> Pool_type:
    """Delete segments from background at scanning positions.
    
    This is a convenience wrapper around breakpoint_scan and join that deletes
    segments of the specified length from bg_pool at various positions.
    
    If deletion_marker is specified (default: '-'), the deleted segment is 
    replaced with the marker repeated deletion_length times. If deletion_marker
    is None, the segment is simply removed and the output length is reduced.
    
    Args:
        bg_pool: Background pool or sequence string.
        deletion_length: Length of segment to delete.
        start: Start position for deletion (default: 0).
        end: End position for deletion (default: bg_length - deletion_length).
        step_size: Step size for scanning positions (default: 1).
        mode: Iteration mode ('sequential', 'random', or 'hybrid').
        num_hybrid_states: Number of states for hybrid mode.
        pool_iteration_order: Sort key for the result pool (default 0).
        op_iteration_order: Sort key for the breakpoint_scan counter (default 0).
        deletion_marker: Character(s) to mark deletion (default: '-').
            If None, segment is simply removed without marker.
        spacer_str: String to insert between segments (default: '').
        op_name: Optional name for the join operation.
        name: Optional name for the result pool.
    
    Returns:
        A pool with deleted segments at scanning positions.
        If deletion_marker is specified: output length = bg_pool.seq_length.
        If deletion_marker is None: output length = bg_pool.seq_length - deletion_length.
    
    Raises:
        ValueError: If bg_pool doesn't have defined seq_length,
            if deletion_length >= bg_length, or if end exceeds maximum.
    """
    from .from_seqs import from_seqs
    from .seq_slice import seq_slice
    from .join import join
    from .breakpoint_scan import breakpoint_scan
    
    # Convert string input to pool if needed
    if isinstance(bg_pool, str):
        bg_pool = from_seqs([bg_pool], mode='fixed')
    
    # Validate that bg_pool has defined seq_length
    bg_length = bg_pool.seq_length
    if bg_length is None:
        raise ValueError("bg_pool must have a defined seq_length")
    
    # Validate deletion_length
    if deletion_length <= 0:
        raise ValueError(f"deletion_length must be > 0, got {deletion_length}")
    if deletion_length >= bg_length:
        raise ValueError(
            f"deletion_length ({deletion_length}) must be < bg_pool.seq_length ({bg_length})"
        )
    
    # Compute max_end and validate/default end parameter
    max_end = bg_length - deletion_length
    if end is None:
        end = max_end
    elif end > max_end:
        raise ValueError(
            f"end ({end}) exceeds maximum allowed value ({max_end}) "
            f"based on bg_pool.seq_length ({bg_length}) - deletion_length ({deletion_length})"
        )
    
    # Default start to 0
    if start is None:
        start = 0
    
    # Split background at breakpoint positions
    left, right = breakpoint_scan(
        pool=bg_pool,
        num_breakpoints=1,
        start=start,
        end=end,
        step_size=step_size,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        op_iter_order=op_iteration_order,
    )
    
    # Clip the right segment by removing the first deletion_length characters
    right_clipped = seq_slice(right, slice(deletion_length, None, None))
    
    if deletion_marker is not None:
        # Create marker sequence as replacement
        marker_seq = deletion_marker * deletion_length
        marker_pool = from_seqs([marker_seq], mode='fixed')
        # Join left, marker, and right_clipped
        result = join([left, marker_pool, right_clipped], spacer_str=spacer_str, op_name=op_name)
    else:
        # No marker - just join left and right_clipped
        result = join([left, right_clipped], spacer_str=spacer_str, op_name=op_name)
    
    result.iteration_order = pool_iteration_order
    if name is not None:
        result.name = name
    
    return result

