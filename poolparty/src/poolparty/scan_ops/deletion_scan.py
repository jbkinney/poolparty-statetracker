"""Deletion scan operation - delete a segment at scanning positions."""
from numbers import Integral, Real

from ..types import Union, ModeType, Optional, PositionsType, RegionType, beartype
from ..seq_utils import validate_positions
from ..party import get_active_party
from ..pool import Pool


@beartype
def deletion_scan(
    bg_pool: Union[Pool, str],
    deletion_length: Integral,
    deletion_marker: Optional[str] = '-',
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    mark_changes: Optional[bool] = None,
    spacer_str: str = '',
    positions: PositionsType = None,
    min_spacing: Optional[Integral] = None,
    max_spacing: Optional[Integral] = None,
    name_prefix: Optional[str] = None,
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
        Character to insert at the deletion site when mark_changes is True.
        If None, deleted segment is removed with no marker.
    region : RegionType, default=None
        Region to constrain the scan to. Can be a marker name (str) or [start, stop].
        If specified, positions are relative to the region start.
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
        If None, uses Party default.
    mark_changes : Optional[bool], default=None
        If True, insert deletion_marker for each deleted molecular character.
        If None, uses party default. If False, segment is simply removed.
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
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.join import join
    from ..marker_ops import marker_scan, replace_marker_content, apply_at_marker, insert_marker

    # Validate min_spacing/max_spacing not supported
    if min_spacing is not None or max_spacing is not None:
        raise ValueError(
            "min_spacing and max_spacing are not supported in the marker-based "
            "implementation of deletion_scan. Use breakpoint_scan directly if needed."
        )

    # Convert string inputs to pools if needed
    bg_pool = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool

    # Resolve mark_changes: if not explicitly set, use deletion_marker presence as the trigger
    # (deletion_marker='-' means insert markers, deletion_marker=None means no markers)
    # This maintains backward compatibility with the original API
    party = get_active_party()
    if mark_changes is None:
        # If deletion_marker is explicitly None, don't mark changes
        # If deletion_marker has a value (including default '-'), mark changes
        mark_changes = deletion_marker is not None

    # Resolve remove_marker from party defaults if not explicitly set
    if remove_marker is None:
        remove_marker = party.get_default('remove_marker', True) if party else True

    # Default deletion_marker character
    del_char = deletion_marker if deletion_marker else '-'

    # If region is specified, apply scan within that region
    if region is not None:
        # Define transform function that applies deletion_scan to region content
        def do_deletion_scan(region_content_pool):
            return _deletion_scan_impl(
                bg_pool=region_content_pool,
                deletion_length=deletion_length,
                del_char=del_char,
                mark_changes=mark_changes,
                spacer_str=spacer_str,
                positions=positions,
                name_prefix=name_prefix,
                mode=mode,
                num_hybrid_states=num_hybrid_states,
                op_name=op_name,
                op_iter_order=op_iter_order,
            )

        if isinstance(region, str):
            # Region is a marker name
            return apply_at_marker(
                bg_pool,
                marker_name=region,
                transform_fn=do_deletion_scan,
                remove_marker=remove_marker,
                name=name,
                iter_order=iter_order,
            )
        else:
            # Region is [start, stop] - insert temporary marker
            temp_marker = '_deletion_scan_region'
            marked_pool = insert_marker(
                bg_pool,
                marker_name=temp_marker,
                start=int(region[0]),
                stop=int(region[1]),
            )
            return apply_at_marker(
                marked_pool,
                marker_name=temp_marker,
                transform_fn=do_deletion_scan,
                remove_marker=True,  # Always remove temp marker
                name=name,
                iter_order=iter_order,
            )

    # No region specified - apply to entire bg_pool
    return _deletion_scan_impl(
        bg_pool=bg_pool,
        deletion_length=deletion_length,
        del_char=del_char,
        mark_changes=mark_changes,
        spacer_str=spacer_str,
        positions=positions,
        name_prefix=name_prefix,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
    )


def _deletion_scan_impl(
    bg_pool: Pool,
    deletion_length: Integral,
    del_char: str,
    mark_changes: bool,
    spacer_str: str,
    positions: PositionsType,
    name_prefix: Optional[str],
    mode: ModeType,
    num_hybrid_states: Optional[Integral],
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """Core deletion scan implementation without region handling."""
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.join import join
    from ..marker_ops import marker_scan, replace_marker_content

    # Validate bg_pool has defined seq_length
    bg_length = bg_pool.seq_length
    if bg_length is None:
        raise ValueError("bg_pool must have a defined seq_length")

    # Validate deletion_length
    if deletion_length <= 0:
        raise ValueError(f"del_length must be > 0, got {deletion_length}")
    if deletion_length >= bg_length:
        raise ValueError(
            f"del_length ({deletion_length}) must be < bg_pool.seq_length ({bg_length})"
        )

    # For deletion: marker_length=del_length, max_position=bg_length - del_length
    marker_name = '_del'
    marker_length = int(deletion_length)
    max_position = bg_length - deletion_length

    # Validate positions
    validated_positions = validate_positions(positions, max_position, min_position=0)

    # 1. Insert marker at scanning positions
    marked = marker_scan(
        bg_pool,
        marker=marker_name,
        marker_length=marker_length,
        positions=validated_positions,
        name_prefix=name_prefix,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        op_name=op_name,
        op_iter_order=op_iter_order,
    )

    # 2. Build replacement content based on mark_changes
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

    # 3. Replace marker with content
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
