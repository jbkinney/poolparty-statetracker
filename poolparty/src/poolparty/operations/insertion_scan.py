"""InsertionScan - insert sequences into background at scanning positions."""
from ..types import Union, ModeType, Optional, Integral, Real, PositionsType, beartype
from ..seq_utils import validate_positions
from ..pool import Pool


@beartype
def insertion_scan(
    bg_pool: Union[Pool, str],
    ins_pool: Union[Pool, str],
    positions: PositionsType = None,
    min_spacing: Optional[Integral] = None,
    max_spacing: Optional[Integral] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    spacer_str: str = '',
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Insert a sequence into a background sequence at specified scanning positions.

    Parameters
    ----------
    bg_pool : Pool or str
        The background Pool or sequence string in which to insert.
    ins_pool : Pool or str
        The insert Pool or sequence string to be inserted.
    positions : PositionsType, default=None
        Positions to consider for the start of the insertion (0-based, inclusive). If None, all valid positions are considered.
    min_spacing : Optional[Integral], default=None
        Minimum spacing required between breakpoints (not commonly used for single insertions).
    max_spacing : Optional[Integral], default=None
        Maximum spacing allowed between breakpoints (not commonly used for single insertions).
    mode : ModeType, default='random'
        Selection mode for insert positions: 'random', 'sequential', or 'hybrid'.
    num_hybrid_states : Optional[Integral], default=None
        Number of pool states when using 'hybrid' mode (ignored by other modes).
    spacer_str : str, default=''
        String to insert as a spacer between pool segments (optional, placed between left, insert, and right).
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operation.

    Returns
    -------
    Pool
        A Pool yielding sequences where the insert is placed at the selected position(s) in the background.
    """
    from .from_seq import from_seq
    from .join import join
    from ..markers import marker_scan, replace_marker_content

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

    # Validate positions (valid range: 0 to bg_length inclusive)
    validated_positions = validate_positions(positions, max_position=bg_length, min_position=0)

    # Note: min_spacing/max_spacing are not supported in marker-based approach
    # They were rarely used for single insertions anyway
    if min_spacing is not None or max_spacing is not None:
        raise ValueError(
            "min_spacing and max_spacing are not supported in the marker-based "
            "implementation of insertion_scan. Use breakpoint_scan directly if needed."
        )

    # 1. Insert zero-length marker at scanning positions
    marked = marker_scan(
        bg_pool,
        marker='_ins',
        marker_length=0,
        positions=validated_positions,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        op_iter_order=op_iter_order,
    )

    # 2. Wrap insert with spacers if needed
    if spacer_str:
        content = join([from_seq(spacer_str), ins_pool, from_seq(spacer_str)])
    else:
        content = ins_pool

    # 3. Replace marker with insert content
    result = replace_marker_content(
        marked,
        content,
        '_ins',
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
    )
    return result
