"""Scan operations - insert, replace, delete, or shuffle sequences at scanning positions."""
from numbers import Integral, Real
from typing import Literal

from ..types import Union, ModeType, Optional, PositionsType, beartype
from ..seq_utils import validate_positions
from ..party import get_active_party
from ..pool import Pool


# Type alias for scan action
ActionType = Literal['insert', 'replace', 'delete', 'shuffle']


@beartype
def scan(
    action: ActionType,
    bg_pool: Union[Pool, str],
    ins_pool: Optional[Union[Pool, str]] = None,
    del_length: Optional[Integral] = None,
    del_char: str = '-',
    shuffle_length: Optional[Integral] = None,
    positions: PositionsType = None,
    spacer_str: str = '',
    mark_changes: Optional[bool] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Unified scan operation for inserting, replacing, deleting, or shuffling at scanning positions.

    Parameters
    ----------
    action : Literal['insert', 'replace', 'delete', 'shuffle']
        The type of scan operation to perform:
        - 'insert': Insert a sequence at scanning positions (adds to background length)
        - 'replace': Replace a segment with insert sequence (preserves background length)
        - 'delete': Remove a segment from background (reduces length or fills with marker)
        - 'shuffle': Shuffle characters within a segment at scanning positions
    bg_pool : Pool or str
        The background Pool or sequence string.
    ins_pool : Pool or str, optional
        The insert Pool or sequence string. Required for 'insert' and 'replace' actions.
    del_length : Integral, optional
        Number of characters to delete. Required for 'delete' action.
    del_char : str, default='-'
        Character used to fill deletion gap when mark_changes=True. Only used for 'delete' action.
    shuffle_length : Integral, optional
        Length of region to shuffle. Required for 'shuffle' action.
    positions : PositionsType, default=None
        Positions to consider for the operation (0-based). If None, all valid positions are used.
    spacer_str : str, default=''
        String to insert as a spacer between segments.
    mark_changes : Optional[bool], default=None
        Flag controlling how changes are marked:
        - 'delete': If True, fill gap with del_char * del_length; if False, simply remove segment
        - 'replace'/'insert'/'shuffle': If True, apply swap_case() to the modified region
        If None, resolves from Party defaults.
    mode : ModeType, default='random'
        Selection mode: 'random', 'sequential', or 'hybrid'.
    num_hybrid_states : Optional[Integral], default=None
        Number of pool states when using 'hybrid' mode.
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
        A Pool yielding sequences with the scan operation applied.

    Examples
    --------
    >>> with pp.Party():
    ...     # Insert 'TTT' at all positions in background
    ...     result = pp.scan('insert', 'AAAAAAAAAA', ins_pool='TTT')
    ...
    ...     # Replace 3-bp segments with 'TTT'
    ...     result = pp.scan('replace', 'AAAAAAAAAA', ins_pool='TTT')
    ...
    ...     # Delete 3-bp segments (fill with '-')
    ...     result = pp.scan('delete', 'AAAAAAAAAA', del_length=3, mark_changes=True)
    ...
    ...     # Delete 3-bp segments (remove entirely)
    ...     result = pp.scan('delete', 'AAAAAAAAAA', del_length=3, mark_changes=False)
    ...
    ...     # Shuffle 3-bp segments at scanning positions
    ...     result = pp.scan('shuffle', 'AAACCCGGGTTT', shuffle_length=3)
    """
    from .from_seq import from_seq
    from .join import join
    from .swap_case import swap_case
    from .seq_shuffle import seq_shuffle
    from ..markers import marker_scan, replace_marker_content, apply_at_marker

    # Convert string inputs to pools if needed
    bg_pool = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool

    # Validate bg_pool has defined seq_length
    bg_length = bg_pool.seq_length
    if bg_length is None:
        raise ValueError("bg_pool must have a defined seq_length")

    # Resolve mark_changes from party defaults if not explicitly set
    party = get_active_party()
    if mark_changes is None:
        mark_changes = party.get_default('mark_changes', False) if party else False

    # Action-specific validation and setup
    if action == 'insert':
        # Insert action: ins_pool required, marker_length=0
        if ins_pool is None:
            raise ValueError("ins_pool is required for 'insert' action")
        ins_pool = from_seq(ins_pool) if isinstance(ins_pool, str) else ins_pool
        ins_length = ins_pool.seq_length
        if ins_length is None:
            raise ValueError("ins_pool must have a defined seq_length")

        marker_name = '_ins'
        marker_length = 0
        max_position = bg_length  # Can insert at any position including after last char

        # Apply swap_case if mark_changes
        if mark_changes:
            ins_pool = swap_case(ins_pool)

    elif action == 'replace':
        # Replace action: ins_pool required, marker_length=ins_length
        if ins_pool is None:
            raise ValueError("ins_pool is required for 'replace' action")
        ins_pool = from_seq(ins_pool) if isinstance(ins_pool, str) else ins_pool
        ins_length = ins_pool.seq_length
        if ins_length is None:
            raise ValueError("ins_pool must have a defined seq_length")

        marker_name = '_rep'
        marker_length = ins_length
        max_position = bg_length - ins_length

        # Apply swap_case if mark_changes
        if mark_changes:
            ins_pool = swap_case(ins_pool)

    elif action == 'delete':
        # Delete action: del_length required
        if del_length is None:
            raise ValueError("del_length is required for 'delete' action")
        if del_length <= 0:
            raise ValueError(f"del_length must be > 0, got {del_length}")
        if del_length >= bg_length:
            raise ValueError(
                f"del_length ({del_length}) must be < bg_pool.seq_length ({bg_length})"
            )

        marker_name = '_del'
        marker_length = int(del_length)
        max_position = bg_length - del_length

    elif action == 'shuffle':
        # Shuffle action: shuffle_length required
        if shuffle_length is None:
            raise ValueError("shuffle_length is required for 'shuffle' action")
        if shuffle_length <= 0:
            raise ValueError(f"shuffle_length must be > 0, got {shuffle_length}")
        if shuffle_length >= bg_length:
            raise ValueError(
                f"shuffle_length ({shuffle_length}) must be < bg_pool.seq_length ({bg_length})"
            )

        marker_name = '_shuf'
        marker_length = int(shuffle_length)
        max_position = bg_length - shuffle_length
    else:
        raise ValueError(
            f"Invalid action: {action}. Must be 'insert', 'replace', 'delete', or 'shuffle'."
        )

    # Validate positions
    validated_positions = validate_positions(positions, max_position, min_position=0)

    # 1. Insert marker at scanning positions
    marked = marker_scan(
        bg_pool,
        marker=marker_name,
        marker_length=marker_length,
        positions=validated_positions,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        op_name=op_name,
        op_iter_order=op_iter_order,
    )

    # 2. Build replacement content or apply transformation based on action
    if action == 'shuffle':
        # Shuffle action: use apply_at_marker with seq_shuffle
        # Note: seq_shuffle only supports 'random' mode for the actual shuffling.
        # The 'mode' parameter controls position selection via marker_scan above.
        def shuffle_transform(content_pool):
            shuffled = seq_shuffle(content_pool, mode='random')
            if mark_changes:
                shuffled = swap_case(shuffled)
            # Wrap with spacers if needed
            if spacer_str:
                shuffled = join([from_seq(spacer_str), shuffled, from_seq(spacer_str)])
            return shuffled

        result = apply_at_marker(
            marked,
            marker_name,
            shuffle_transform,
            name=name,
            iter_order=iter_order,
        )
    else:
        # insert, replace, delete: use replace_marker_content
        if action in ('insert', 'replace'):
            # Use ins_pool (already processed with swap_case if needed)
            content = ins_pool
            # Wrap with spacers if needed
            if spacer_str:
                content = join([from_seq(spacer_str), content, from_seq(spacer_str)])
        else:  # action == 'delete'
            if mark_changes:
                # Fill gap with del_char * del_length
                marker_str = del_char * marker_length
                content = from_seq(marker_str)
                # Wrap with spacers if needed
                if spacer_str:
                    content = join([from_seq(spacer_str), content, from_seq(spacer_str)])
            else:
                # Simply remove the segment - just use spacer_str once (or empty)
                content = from_seq(spacer_str)

        result = replace_marker_content(
            marked,
            content,
            marker_name,
            name=name,
            op_name=op_name,
            iter_order=iter_order,
            op_iter_order=op_iter_order,
        )
    return result


# =============================================================================
# Backward-compatible wrapper functions
# =============================================================================

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
        Positions to consider for the start of the insertion (0-based, inclusive).
        If None, all valid positions are considered.
    min_spacing : Optional[Integral], default=None
        Not supported. Raises ValueError if provided.
    max_spacing : Optional[Integral], default=None
        Not supported. Raises ValueError if provided.
    mode : ModeType, default='random'
        Selection mode for insert positions: 'random', 'sequential', or 'hybrid'.
    num_hybrid_states : Optional[Integral], default=None
        Number of pool states when using 'hybrid' mode (ignored by other modes).
    spacer_str : str, default=''
        String to insert as a spacer between pool segments.
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
        A Pool yielding sequences where the insert is placed at the selected position(s)
        in the background.
    """
    # min_spacing/max_spacing not supported in marker-based approach
    if min_spacing is not None or max_spacing is not None:
        raise ValueError(
            "min_spacing and max_spacing are not supported in the marker-based "
            "implementation of insertion_scan. Use breakpoint_scan directly if needed."
        )

    return scan(
        action='insert',
        bg_pool=bg_pool,
        ins_pool=ins_pool,
        positions=positions,
        spacer_str=spacer_str,
        mark_changes=False,  # insertion_scan doesn't use mark_changes
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
    )


@beartype
def replacement_scan(
    bg_pool: Union[Pool, str],
    ins_pool: Union[Pool, str],
    positions: PositionsType = None,
    spacer_str: str = '',
    mark_changes: Optional[bool] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Replace a segment of the background sequence with an insert at specified scanning positions.

    Parameters
    ----------
    bg_pool : Pool or str
        Background Pool or sequence string in which the replacement will occur.
    ins_pool : Pool or str
        Insert Pool or sequence string to replace the segment in the background.
    positions : PositionsType, default=None
        Positions at which to place the start of the replacement (0-based, inclusive).
        If None, all valid positions are considered.
    spacer_str : str, default=''
        String to insert as a spacer between segments when joining.
    mark_changes : Optional[bool], default=None
        If True, apply swapcase() to the insert sequence. If None, uses party default.
    mode : ModeType, default='random'
        Selection mode for replacement positions: 'sequential', 'random', or 'hybrid'.
    num_hybrid_states : Optional[Integral], default=None
        Number of pool states to use when mode is 'hybrid' (ignored for other modes).
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
        A Pool yielding sequences where a segment of the background is replaced by the
        insert sequence at the specified scanning positions.
    """
    return scan(
        action='replace',
        bg_pool=bg_pool,
        ins_pool=ins_pool,
        positions=positions,
        spacer_str=spacer_str,
        mark_changes=mark_changes,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
    )


@beartype
def deletion_scan(
    bg_pool: Union[Pool, str],
    deletion_length: Integral,
    deletion_marker: Optional[str] = '-',
    spacer_str: str = '',
    positions: PositionsType = None,
    min_spacing: Optional[Integral] = None,
    max_spacing: Optional[Integral] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Scan a pool for all possible single deletions of a fixed length.

    Parameters
    ----------
    bg_pool : Pool or str
        Source pool or sequence string to delete from.
    deletion_length : Integral
        Number of characters to delete at each valid position.
    deletion_marker : Optional[str], default='-'
        String to insert at the deletion site (i.e., a gap marker). If None, deleted
        segment is removed with no marker.
    spacer_str : str, default=''
        String to insert as a spacer between pool segments after deletion.
    positions : PositionsType, default=None
        Positions to consider for the start of the deletion (0-based).
        If None, all valid positions are used.
    min_spacing : Optional[Integral], default=None
        Not supported. Raises ValueError if provided.
    max_spacing : Optional[Integral], default=None
        Not supported. Raises ValueError if provided.
    mode : ModeType, default='random'
        Deletion mode: 'random', 'sequential', or 'hybrid'.
    num_hybrid_states : Optional[Integral], default=None
        Number of pool states when using 'hybrid' mode (ignored by other modes).
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
        A Pool yielding sequences where a segment of the specified length is removed
        from the source at each allowed position, optionally with a marker inserted.
    """
    # min_spacing/max_spacing not supported in marker-based approach
    if min_spacing is not None or max_spacing is not None:
        raise ValueError(
            "min_spacing and max_spacing are not supported in the marker-based "
            "implementation of deletion_scan. Use breakpoint_scan directly if needed."
        )

    # Map old API: deletion_marker='-' -> mark_changes=True, del_char='-'
    #              deletion_marker=None -> mark_changes=False
    mark_changes = deletion_marker is not None
    del_char = deletion_marker if deletion_marker else '-'

    return scan(
        action='delete',
        bg_pool=bg_pool,
        del_length=deletion_length,
        del_char=del_char,
        positions=positions,
        spacer_str=spacer_str,
        mark_changes=mark_changes,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
    )


@beartype
def shuffle_scan(
    bg_pool: Union[Pool, str],
    shuffle_length: Integral,
    positions: PositionsType = None,
    spacer_str: str = '',
    mark_changes: Optional[bool] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Shuffle characters within a window at specified scanning positions.

    Parameters
    ----------
    bg_pool : Pool or str
        Source pool or sequence string to shuffle regions of.
    shuffle_length : Integral
        Length of the region to shuffle at each position.
    positions : PositionsType, default=None
        Positions to consider for the start of the shuffle region (0-based).
        If None, all valid positions are used.
    spacer_str : str, default=''
        String to insert as a spacer around the shuffled region.
    mark_changes : Optional[bool], default=None
        If True, apply swap_case() to the shuffled region. If None, uses party default.
    mode : ModeType, default='random'
        Selection mode: 'random', 'sequential', or 'hybrid'.
    num_hybrid_states : Optional[Integral], default=None
        Number of pool states when using 'hybrid' mode (ignored by other modes).
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
        A Pool yielding sequences where a region of the specified length is shuffled
        at each allowed position.
    """
    return scan(
        action='shuffle',
        bg_pool=bg_pool,
        shuffle_length=shuffle_length,
        positions=positions,
        spacer_str=spacer_str,
        mark_changes=mark_changes,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
    )
