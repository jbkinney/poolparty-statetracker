"""DeletionScan - delete segments from background at scanning positions."""
from numbers import Real
from ..types import Union, ModeType, Optional, Integral, Sequence, beartype
from ..pool import Pool


@beartype
def deletion_scan(
    bg_pool: Union[Pool, str],
    deletion_length: Integral,
    deletion_marker: Optional[str] = '-',
    spacer_str: str = '',
    positions: Optional[Sequence[Integral]] = None,
    start: Optional[Integral] = None,
    end: Optional[Integral] = None,
    step_size: Integral = 1,
    min_spacing: Optional[Integral] = None,
    max_spacing: Optional[Integral] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    name: Optional[str] = None, 
    op_name: Optional[str] = None,
    iter_order: Real = 0,
    op_iter_order: Real = 0,
) -> Pool:
    """
    Delete a segment of specified length from the background sequence or pool at one or more scanning positions.

    Parameters
    ----------
    bg_pool : Union[Pool, str]
        The background sequence or Pool from which to delete segments.
    deletion_length : Integral
        The length of each segment to be deleted.
    deletion_marker : Optional[str], default='-'
        String to insert marking the deletion position (e.g., for gapped output); set to None for no marker.
    spacer_str : str, default=''
        Optional string inserted between sequence segments when joining.
    positions : Optional[Sequence[Integral]], default=None
        Explicit positions at which to perform deletions. If provided, overrides start/end/step_size.
    start : Optional[Integral], default=None
        Minimum starting index for deletion scan (inclusive). Defaults to 0.
    end : Optional[Integral], default=None
        Maximum end index where deletions can occur (inclusive). Defaults to `bg_pool.seq_length - deletion_length`.
    step_size : Integral, default=1
        Step size for scanning deletion positions.
    min_spacing : Optional[Integral], default=None
        Minimum number of bases between consecutive deletions (if relevant for multi-deletion modes).
    max_spacing : Optional[Integral], default=None
        Maximum number of bases between consecutive deletions (if relevant for multi-deletion modes).
    mode : ModeType, default='random'
        Deletion scan mode: 'sequential', 'random', or 'hybrid'.
    num_hybrid_states : Optional[Integral], default=None
        Number of pool states when using 'hybrid' mode (ignored for other modes).
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Real, default=0
        Iteration order priority for the resulting Pool.
    op_iter_order : Real, default=0
        Iteration order priority for the underlying Operation.

    Returns
    -------
    Pool
        A Pool containing all sequence variants where a segment of <deletion_length> is deleted at each scan position.
        If `deletion_marker` is provided, the marker fills the deleted region; otherwise, the region is omitted.
        Output sequence length is `bg_pool.seq_length` if markers are used, otherwise reduced by `deletion_length`.
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
        positions=positions,
        start=start,
        end=end,
        step_size=step_size,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        op_iter_order=op_iter_order,
    )

    # Clip the right segment by removing the first deletion_length characters
    right_clipped = seq_slice(right, slice(deletion_length, None, None))

    if deletion_marker is not None:
        marker_seq = deletion_marker * deletion_length
        marker_pool = from_seqs([marker_seq], mode='fixed')
        pools_list = [left, marker_pool, right_clipped]
    else:
        pools_list = [left, right_clipped]
    result = join(
        pools_list,
        spacer_str=spacer_str,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
    )
    return result

