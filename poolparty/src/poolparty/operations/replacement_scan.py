"""ReplacementScan - replace a segment of background with insert sequences."""
from ..types import Pool_type, Union, ModeType, Optional, beartype
from ..pool import Pool


@beartype
def replacement_scan(
    bg_pool: Union[Pool_type, str],
    ins_pool: Union[Pool_type, str],
    start: Optional[int] = None,
    end: Optional[int] = None,
    step_size: int = 1,
    mode: ModeType = 'sequential',
    hybrid_mode_num_states: Optional[int] = None,
    iteration_order: int = 0,
    spacer_str: str = '',
    op_name: Optional[str] = None,
    pool_name: Optional[str] = None,
) -> Pool_type:
    """Replace a segment of background with insert sequences at scanning positions.
    
    This is a convenience wrapper around breakpoint_scan, seq_slice, and join
    that replaces a segment of bg_pool with sequences from ins_pool at various 
    positions. The output length equals bg_pool.seq_length (the insert replaces
    an equal-length segment of the background).
    
    Args:
        bg_pool: Background pool or sequence string.
        ins_pool: Insert pool or sequence string.
        start: Start position for replacement (default: 0).
        end: End position for replacement (default: bg_length - ins_length).
        step_size: Step size for scanning positions (default: 1).
        mode: Iteration mode ('sequential', 'random', or 'hybrid').
        hybrid_mode_num_states: Number of states for hybrid mode.
        iteration_order: Sort key for the breakpoint_scan counter (default 0).
        spacer_str: String to insert between segments (default: '').
        op_name: Optional name for the join operation.
        pool_name: Optional name for the result pool.
    
    Returns:
        A pool with replaced sequences at scanning positions.
        Output length = bg_pool.seq_length.
    
    Raises:
        ValueError: If pools don't have defined seq_length or if end exceeds maximum.
    """
    from .from_seqs import from_seqs
    from .seq_slice import seq_slice
    from .join import join
    from .breakpoint_scan import breakpoint_scan
    
    # Convert string inputs to pools if needed
    if isinstance(bg_pool, str):
        bg_pool = from_seqs([bg_pool], mode='fixed')
    if isinstance(ins_pool, str):
        ins_pool = from_seqs([ins_pool], mode='fixed')
    
    # Validate that both pools have defined seq_length
    bg_length = bg_pool.seq_length
    ins_length = ins_pool.seq_length
    if bg_length is None:
        raise ValueError("bg_pool must have a defined seq_length")
    if ins_length is None:
        raise ValueError("ins_pool must have a defined seq_length")
    
    # Compute max_end and validate/default end parameter
    max_end = bg_length - ins_length
    if end is None:
        end = max_end
    elif end > max_end:
        raise ValueError(
            f"end ({end}) exceeds maximum allowed value ({max_end}) "
            f"based on bg_pool.seq_length ({bg_length}) - ins_pool.seq_length ({ins_length})"
        )
    
    # Default start to 0
    if start is None:
        start = 0
    
    # Split background at breakpoint positions
    left, right = breakpoint_scan(
        parent=bg_pool,
        num_breakpoints=1,
        start=start,
        end=end,
        step_size=step_size,
        mode=mode,
        hybrid_mode_num_states=hybrid_mode_num_states,
        iteration_order=iteration_order,
    )
    
    # Clip the right segment by removing the first ins_length characters
    right_clipped = seq_slice(right, slice(ins_length, None, None))
    
    # Join left, insert, and right_clipped
    result = join([left, ins_pool, right_clipped], spacer_str=spacer_str, op_name=op_name)
    
    if pool_name is not None:
        result.name = pool_name
    
    return result

