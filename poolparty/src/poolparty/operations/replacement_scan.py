"""ReplacementScan - replace a segment of background with insert sequences."""
from numbers import Real
from ..types import Union, ModeType, Optional, Integral, beartype
from ..pool import Pool


@beartype
def replacement_scan(
    bg_pool: Union[Pool, str],
    ins_pool: Union[Pool, str],
    spacer_str: str = '',
    start: Optional[Integral] = None,
    end: Optional[Integral] = None,
    step_size: Integral = 1,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Real = 0,
    op_iter_order: Real = 0,
) -> Pool:
    """
    Replace a segment of a background sequence with an insert sequence at multiple positions.

    Parameters
    ----------
    bg_pool : Union[Pool, str]
        Background sequence or pool.
    ins_pool : Union[Pool, str]
        Insert sequence or pool.
    spacer_str : str, default=''
        String to insert between sequence segments when joining (optional).
    start : Optional[Integral], default=0
        Start position for replacement scanning (0-indexed, inclusive).
    end : Optional[Integral], default=None
        End position for replacement scanning (0-indexed, inclusive; defaults to
        bg_pool.seq_length - ins_pool.seq_length).
    step_size : Integral, default=1
        Step size for scanning replacement positions.
    mode : ModeType, default='random'
        Scanning mode: 'sequential', 'random', or 'hybrid'.
    num_hybrid_states : Optional[Integral], default=None
        Number of states for hybrid mode (ignored for other modes).
    name : Optional[str], default=None
        Name to assign to the resulting Pool.
    op_name : Optional[str], default=None
        Name to assign to underlying breakpoint/join operations.
    iter_order : Real, default=0
        Iteration order priority for the resulting Pool.
    op_iter_order : Real, default=0
        Iteration order priority for the underlying operations.

    Returns
    -------
    Pool
        A Pool containing sequences where an equal-length segment of the background
        is replaced by the insert, for each scan position.
        Output sequence length = bg_pool.seq_length.
    """
    from .from_seq import from_seq
    from .seq_slice import seq_slice
    from .join import join
    from .breakpoint_scan import breakpoint_scan
    
    # Convert string inputs to pools if needed
    bg_pool = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool
    ins_pool = from_seq(ins_pool) if isinstance(ins_pool, str) else ins_pool
    
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
    breakpoint_scan_op_name = op_name+'.breakpoint_scan' if op_name is not None else None
    left, right = breakpoint_scan(
        pool=bg_pool,
        num_breakpoints=1,
        start=start,
        end=end,
        step_size=step_size,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        op_iter_order=op_iter_order,
        op_name=op_name,
    )
    
    # Clip the right segment by removing the first ins_length characters
    right_clipped = seq_slice(right, slice(ins_length, None, None))
    
    # Join left, insert, and right_clipped
    result_pool = join([left, ins_pool, right_clipped], 
                       spacer_str=spacer_str, name=name, op_name=op_name,
                       iter_order=iter_order)
    return result_pool

