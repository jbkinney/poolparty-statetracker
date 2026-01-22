"""Deletion scan operation - delete a segment at scanning positions."""
from numbers import Integral, Real

from ..types import Union, ModeType, Optional, PositionsType, RegionType, beartype
from ..party import get_active_party
from ..pool import Pool


@beartype
def deletion_scan(
    pool: Union[Pool, str],
    deletion_length: Integral,
    deletion_marker: Optional[str] = '-',
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    mark_changes: Optional[bool] = None,
    spacer_str: str = '',
    positions: PositionsType = None,
    min_spacing: Optional[Integral] = None,
    max_spacing: Optional[Integral] = None,
    seq_name_prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Scan a pool for all possible single deletions of a fixed length.

    Parameters
    ----------
    pool : Pool or str
        Source pool or sequence string to delete from.
    deletion_length : Integral
        Number of characters to delete at each valid position.
    deletion_marker : Optional[str], default='-'
        Character to insert at the deletion site. If None, segment is removed.
    region : RegionType, default=None
        Region to constrain the scan to. Can be a marker name (str) or [start, stop].
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
    positions : PositionsType, default=None
        Positions to consider for the start of the deletion (0-based, relative to region).
    mode : ModeType, default='random'
        Deletion mode: 'random' or 'sequential'.

    Returns
    -------
    Pool
        A Pool yielding sequences where a segment of the specified length is removed
        from the source at each allowed position, optionally with a marker inserted.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..marker_ops import marker_scan

    # Validate min_spacing/max_spacing not supported
    if min_spacing is not None or max_spacing is not None:
        raise ValueError(
            "min_spacing and max_spacing are not supported. "
            "Use breakpoint_scan directly if needed."
        )

    # Convert string to pool
    pool = from_seq(pool) if isinstance(pool, str) else pool

    # Validate bg_pool has defined seq_length (only when no region specified)
    bg_length = pool.seq_length
    if bg_length is None and region is None:
        raise ValueError("pool must have a defined seq_length")

    # Validate deletion_length
    if deletion_length <= 0:
        raise ValueError(f"del_length must be > 0, got {deletion_length}")
    if bg_length is not None and deletion_length >= bg_length:
        raise ValueError(
            f"del_length ({deletion_length}) must be < pool.seq_length ({bg_length})"
        )

    # Resolve mark_changes from deletion_marker presence
    party = get_active_party()
    if mark_changes is None:
        mark_changes = deletion_marker is not None

    # Resolve remove_marker from party defaults (for user's outer region)
    if remove_marker is None:
        remove_marker = party.get_default('remove_marker', True) if party else True

    del_char = deletion_marker if deletion_marker else '-'
    marker_name = '_del'
    marker_length = int(deletion_length)

    # 1. Insert marker at scanning positions
    # positions are relative to region content (marker_scan handles this via Operation base)
    marked = marker_scan(
        pool,
        marker=marker_name,
        marker_length=marker_length,
        positions=positions,  # Let marker_scan validate positions relative to region
        region=region,
        remove_marker=False,  # Keep outer region marker for now
        seq_name_prefix=seq_name_prefix,
        mode=mode,
        num_states=num_states,
        op_name=op_name,
        op_iter_order=op_iter_order,
        _factory_name='deletion_scan(marker_scan)',
    )

    # 2. Build deletion content string
    content_str = del_char * marker_length if mark_changes else ''

    # 3. Replace _del marker with content
    # spacer_str is only applied when mark_changes is True (i.e., when there's content to wrap)
    # Always remove the internal _del marker (it's our implementation detail)
    return from_seq(
        content_str,
        pool=marked,
        region=marker_name,
        remove_marker=True,  # Always remove the internal _del marker
        spacer_str=spacer_str if mark_changes else '',
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
        _factory_name='deletion_scan(from_seq)',
    )
