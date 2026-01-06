"""Shuffle scan operation - shuffle characters within a window at scanning positions."""
from numbers import Integral, Real

from ..types import Union, ModeType, Optional, PositionsType, RegionType, beartype
from ..seq_utils import validate_positions
from ..party import get_active_party
from ..pool import Pool


@beartype
def shuffle_scan(
    bg_pool: Union[Pool, str],
    shuffle_length: Integral,
    positions: PositionsType = None,
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    shuffles_per_position: Integral = 1,    
    spacer_str: str = '',
    mark_changes: Optional[bool] = None,
    seq_name_prefix: Optional[str] = None,
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
    region : RegionType, default=None
        Region to constrain the scan to. Can be a marker name (str) or [start, stop].
        If specified, positions are relative to the region start.
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
        If None, uses Party default.
    shuffles_per_position : Integral, default=1
        Number of shuffles to perform at each position.
    spacer_str : str, default=''
        String to insert as a spacer around the shuffled region.
    mark_changes : Optional[bool], default=None
        If True, apply swapcase() to the shuffled region. If None, uses party default.
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
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.join import join
    from ..fixed_ops.swapcase import swapcase
    from ..base_ops.shuffle_seq import shuffle_seq
    from ..marker_ops import marker_scan, apply_at_marker, insert_marker

    # Convert string inputs to pools if needed
    bg_pool = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool

    # Resolve mark_changes from party defaults if not explicitly set
    party = get_active_party()
    if mark_changes is None:
        mark_changes = party.get_default('mark_changes', False) if party else False

    # Resolve remove_marker from party defaults if not explicitly set
    if remove_marker is None:
        remove_marker = party.get_default('remove_marker', True) if party else True

    # If region is specified, apply scan within that region
    if region is not None:
        # Define transform function that applies shuffle_scan to region content
        def do_shuffle_scan(region_content_pool):
            return _shuffle_scan_impl(
                bg_pool=region_content_pool,
                shuffle_length=shuffle_length,
                positions=positions,
                shuffles_per_position=shuffles_per_position,
                spacer_str=spacer_str,
                mark_changes=mark_changes,
                seq_name_prefix=seq_name_prefix,
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
                transform_fn=do_shuffle_scan,
                remove_marker=remove_marker,
                name=name,
                iter_order=iter_order,
            )
        else:
            # Region is [start, stop] - insert temporary marker
            temp_marker = '_shuffle_scan_region'
            marked_pool = insert_marker(
                bg_pool,
                marker_name=temp_marker,
                start=int(region[0]),
                stop=int(region[1]),
            )
            return apply_at_marker(
                marked_pool,
                marker_name=temp_marker,
                transform_fn=do_shuffle_scan,
                remove_marker=True,  # Always remove temp marker
                name=name,
                iter_order=iter_order,
            )

    # No region specified - apply to entire bg_pool
    return _shuffle_scan_impl(
        bg_pool=bg_pool,
        shuffle_length=shuffle_length,
        positions=positions,
        shuffles_per_position=shuffles_per_position,
        spacer_str=spacer_str,
        mark_changes=mark_changes,
        seq_name_prefix=seq_name_prefix,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
    )


def _shuffle_scan_impl(
    bg_pool: Pool,
    shuffle_length: Integral,
    positions: PositionsType,
    shuffles_per_position: Integral,
    spacer_str: str,
    mark_changes: bool,
    seq_name_prefix: Optional[str],
    mode: ModeType,
    num_hybrid_states: Optional[Integral],
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """Core shuffle scan implementation without region handling."""
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.join import join
    from ..fixed_ops.swapcase import swapcase
    from ..base_ops.shuffle_seq import shuffle_seq
    from ..marker_ops import marker_scan, apply_at_marker

    # Validate bg_pool has defined seq_length
    bg_length = bg_pool.seq_length
    if bg_length is None:
        raise ValueError("bg_pool must have a defined seq_length")

    # Validate shuffle_length
    if shuffle_length <= 0:
        raise ValueError(f"shuffle_length must be > 0, got {shuffle_length}")
    if shuffle_length >= bg_length:
        raise ValueError(
            f"shuffle_length ({shuffle_length}) must be < bg_pool.seq_length ({bg_length})"
        )

    # For shuffle: marker_length=shuffle_length, max_position=bg_length - shuffle_length
    marker_name = '_shuf'
    marker_length = int(shuffle_length)
    max_position = bg_length - shuffle_length

    # Validate positions
    validated_positions = validate_positions(positions, max_position, min_position=0)

    # 1. Insert marker at scanning positions
    marked = marker_scan(
        bg_pool,
        marker=marker_name,
        marker_length=marker_length,
        positions=validated_positions,
        seq_name_prefix=seq_name_prefix,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        op_name=op_name,
        op_iter_order=op_iter_order,
    )

    # 2. Apply shuffle transform at marker
    # Note: shuffle_seq only supports 'random' mode for the actual shuffling.
    # The 'mode' parameter controls position selection via marker_scan above.
    def shuffle_transform(content_pool):
        shuffled = shuffle_seq(content_pool, 
                               mode='hybrid', 
                               num_hybrid_states=shuffles_per_position,
                               op_iter_order=-1)
        if mark_changes:
            shuffled = swapcase(shuffled)
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
    return result
