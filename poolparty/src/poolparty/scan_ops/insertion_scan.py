"""Insertion scan operation - insert a sequence at scanning positions."""
from numbers import Integral, Real

from ..types import Union, ModeType, Optional, PositionsType, RegionType, beartype
from ..seq_utils import validate_positions
from ..party import get_active_party
from ..pool import Pool


@beartype
def insertion_scan(
    bg_pool: Union[Pool, str],
    ins_pool: Union[Pool, str],
    positions: PositionsType = None,
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    mark_changes: Optional[bool] = None,
    min_spacing: Optional[Integral] = None,
    max_spacing: Optional[Integral] = None,
    name_prefix: Optional[str] = None,
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
    region : RegionType, default=None
        Region to constrain the scan to. Can be a marker name (str) or [start, stop].
        If specified, positions are relative to the region start.
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
        If None, uses Party default.
    mark_changes : Optional[bool], default=None
        If True, apply swapcase() to the inserted content. If None, uses party default.
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
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.join import join
    from ..fixed_ops.swapcase import swapcase
    from ..marker_ops import marker_scan, replace_marker_content, apply_at_marker, insert_marker

    # Validate min_spacing/max_spacing not supported
    if min_spacing is not None or max_spacing is not None:
        raise ValueError(
            "min_spacing and max_spacing are not supported in the marker-based "
            "implementation of insertion_scan. Use breakpoint_scan directly if needed."
        )

    # Convert string inputs to pools if needed
    bg_pool = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool
    ins_pool = from_seq(ins_pool) if isinstance(ins_pool, str) else ins_pool

    # Resolve mark_changes from party defaults if not explicitly set
    party = get_active_party()
    if mark_changes is None:
        mark_changes = party.get_default('mark_changes', False) if party else False

    # Resolve remove_marker from party defaults if not explicitly set
    if remove_marker is None:
        remove_marker = party.get_default('remove_marker', True) if party else True

    # Apply swapcase to insert if mark_changes
    if mark_changes:
        ins_pool = swapcase(ins_pool)

    # If region is specified, apply scan within that region
    if region is not None:
        # Define transform function that applies insertion_scan to region content
        def do_insertion_scan(region_content_pool):
            return _insertion_scan_impl(
                bg_pool=region_content_pool,
                ins_pool=ins_pool,
                positions=positions,
                name_prefix=name_prefix,
                mode=mode,
                num_hybrid_states=num_hybrid_states,
                spacer_str=spacer_str,
                op_name=op_name,
                op_iter_order=op_iter_order,
            )

        if isinstance(region, str):
            # Region is a marker name
            return apply_at_marker(
                bg_pool,
                marker_name=region,
                transform_fn=do_insertion_scan,
                remove_marker=remove_marker,
                name=name,
                iter_order=iter_order,
            )
        else:
            # Region is [start, stop] - insert temporary marker
            temp_marker = '_insertion_scan_region'
            marked_pool = insert_marker(
                bg_pool,
                marker_name=temp_marker,
                start=int(region[0]),
                stop=int(region[1]),
            )
            return apply_at_marker(
                marked_pool,
                marker_name=temp_marker,
                transform_fn=do_insertion_scan,
                remove_marker=True,  # Always remove temp marker
                name=name,
                iter_order=iter_order,
            )

    # No region specified - apply to entire bg_pool
    return _insertion_scan_impl(
        bg_pool=bg_pool,
        ins_pool=ins_pool,
        positions=positions,
        name_prefix=name_prefix,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        spacer_str=spacer_str,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
    )


def _insertion_scan_impl(
    bg_pool: Pool,
    ins_pool: Pool,
    positions: PositionsType,
    name_prefix: Optional[str],
    mode: ModeType,
    num_hybrid_states: Optional[Integral],
    spacer_str: str,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """Core insertion scan implementation without region handling."""
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.join import join
    from ..marker_ops import marker_scan, replace_marker_content

    # Validate bg_pool has defined seq_length
    bg_length = bg_pool.seq_length
    if bg_length is None:
        raise ValueError("bg_pool must have a defined seq_length")

    # Validate ins_pool has defined seq_length
    ins_length = ins_pool.seq_length
    if ins_length is None:
        raise ValueError("ins_pool must have a defined seq_length")

    # For insertion: marker_length=0, can insert at any position including after last char
    marker_name = '_ins'
    marker_length = 0
    max_position = bg_length

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

    # 2. Build replacement content (ins_pool with optional spacers)
    content = ins_pool
    if spacer_str:
        content = join([from_seq(spacer_str), content, from_seq(spacer_str)])

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
