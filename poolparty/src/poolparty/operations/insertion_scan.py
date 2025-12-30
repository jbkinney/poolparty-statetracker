"""InsertionScan - insert sequences into background at scanning positions."""
from numbers import Real
from ..types import Union, ModeType, Optional, Integral, Sequence, beartype
from ..pool import Pool


@beartype
def insertion_scan(
    bg_pool: Union[Pool, str],
    ins_pool: Union[Pool, str],
    positions: Optional[Sequence[Integral]] = None,
    start: Optional[Integral] = None,
    end: Optional[Integral] = None,
    step_size: Integral = 1,
    min_spacing: Optional[Integral] = None,
    max_spacing: Optional[Integral] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    spacer_str: str = '',
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Real = 0,
    op_iter_order: Real = 0,
) -> Pool:
    """
    Create a Pool containing all possible variants of the background sequence or pool
    with the insertion pool inserted at each scan position.

    Parameters
    ----------
    bg_pool : Union[Pool, str]
        The background sequence or Pool into which the insert will be introduced.
    ins_pool : Union[Pool, str]
        The sequence or Pool to be inserted at each scan position.
    positions : Optional[Sequence[Integral]], default=None
        Explicit positions at which to perform insertions. If provided, overrides start/end/step_size.
    start : Optional[Integral], default=None
        Minimum allowed position (inclusive) at which to insert. Defaults to 0.
    end : Optional[Integral], default=None
        Maximum allowed position (inclusive) at which to insert. Defaults to bg_pool.seq_length.
    step_size : Integral, default=1
        Step size for scanning insertion positions.
    min_spacing : Optional[Integral], default=None
        Minimum number of bases between consecutive insertions (relevant if scanning >1 position).
    max_spacing : Optional[Integral], default=None
        Maximum number of bases between consecutive insertions (relevant if scanning >1 position).
    mode : ModeType, default='random'
        Insertion scan mode: 'sequential', 'random', or 'hybrid'.
    num_hybrid_states : Optional[Integral], default=None
        Number of pool states when using 'hybrid' mode (ignored otherwise).
    spacer_str : str, default=''
        Optional string inserted between pool segments when joining.
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying join/breakpoint operation.
    iter_order : Real, default=0
        Iteration order priority for the resulting Pool.
    op_iter_order : Real, default=0
        Iteration order priority for the underlying Operation.

    Returns
    -------
    Pool
        A Pool containing all sequence variants where ins_pool is inserted at each scan position in the background.
        Output sequence length is bg_pool.seq_length + ins_pool.seq_length.
    """
    from .from_seq import from_seq
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

    # For insertion, can insert at any position from 0 to bg_length (inclusive)
    max_end = bg_length
    if end is None:
        end = max_end
    elif end > max_end:
        raise ValueError(
            f"end ({end}) exceeds maximum allowed value ({max_end}) "
            f"based on bg_pool.seq_length ({bg_length})"
        )
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

    # Join left, insert, and right (no clipping - insert is added)
    result = join(
        [left, ins_pool, right],
        spacer_str=spacer_str,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
    )
    return result
