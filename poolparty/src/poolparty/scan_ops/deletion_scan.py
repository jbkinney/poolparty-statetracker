"""Deletion scan operation - delete a segment at scanning positions."""

from numbers import Integral, Real

from ..pool import Pool
from ..types import CardsType, ModeType, Optional, PositionsType, RegionType, Union, beartype


@beartype
def deletion_scan(
    pool: Union[Pool, str],
    deletion_length: Integral,
    deletion_marker: Optional[str] = "-",
    positions: PositionsType = None,
    region: RegionType = None,
    prefix: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    style: Optional[str] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = "deletion_scan",
) -> Pool:
    """
    Scan a pool for all possible single deletions of a fixed length.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent pool or sequence string.
    deletion_length : Integral
        Number of characters to delete at each valid position.
    deletion_marker : Optional[str], default='-'
        Character to insert at the deletion site. If None, segment is removed.
    positions : PositionsType, default=None
        Positions to consider for the start of the deletion (0-based, relative to region).
    region : RegionType, default=None
        Region to constrain the scan to. Can be a marker name or [start, stop] interval.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : ModeType, default='random'
        Selection mode: 'random' or 'sequential'.
    num_states : Optional[Integral], default=None
        Number of states. In sequential mode, overrides the computed count
        (cycling if greater, clipping if less). In random mode, if None
        defaults to 1 (pure random sampling).
    style : Optional[str], default=None
        Style to apply to deletion gap characters (e.g., 'gray', 'red bold').
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    cards : CardsType, default=None
        Design card keys to include. Available keys: ``'position_index'``,
        ``'start'``, ``'end'``, ``'name'``, ``'region_seq'``.

    Returns
    -------
    Pool
        A Pool yielding sequences where a segment of the specified length is removed
        from the source at each allowed position, optionally with a marker inserted.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..region_ops import region_scan, replace_region

    # Convert string to pool
    pool = (
        from_seq(pool, _factory_name=f"{_factory_name}(from_seq)") if isinstance(pool, str) else pool
    )

    # Validate bg_pool has defined seq_length (only when no region specified)
    bg_length = pool.seq_length
    if bg_length is None and region is None:
        raise ValueError("pool must have a defined seq_length")

    # Validate deletion_length
    if deletion_length <= 0:
        raise ValueError(f"del_length must be > 0, got {deletion_length}")
    if bg_length is not None and deletion_length >= bg_length:
        raise ValueError(f"del_length ({deletion_length}) must be < pool.seq_length ({bg_length})")

    # Use composition pattern: region_scan + replace_region
    marker_name = "_del"

    # 1. Mark the regions to delete with tags
    marked = region_scan(
        pool,
        tag_name=marker_name,
        region_length=int(deletion_length),
        positions=positions,
        region=region,
        remove_tags=False,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        cards=cards,
        _factory_name=f"{_factory_name}(region_scan)",
    )

    # 2. Create replacement content (gap markers or empty string)
    if deletion_marker is not None:
        marker_content = deletion_marker * int(deletion_length)
        # Only apply style if there's content to style
        replacement_style = style
    else:
        marker_content = ""
        # No style when there's no marker content
        replacement_style = None

    marker_pool = from_seq(marker_content, _factory_name=f"{_factory_name}(from_seq)")

    # 3. Replace the marked region with the marker content
    return replace_region(
        marked,
        marker_pool,
        marker_name,
        iter_order=iter_order,
        _factory_name=f"{_factory_name}(replace_region)",
        _style=replacement_style,
    )
