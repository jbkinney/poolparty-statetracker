"""Deletion multiscan operation - delete segments at multiple positions simultaneously."""
from numbers import Integral, Real

from ..types import Union, Optional, PositionsType, beartype
from ..utils import validate_positions
from ..pool import Pool


@beartype
def deletion_multiscan(
    pool: Union[Pool, str],
    deletion_length: Integral,
    num_deletions: Integral,
    deletion_marker: Optional[str] = '-',
    positions: PositionsType = None,
    prefix: Optional[str] = None,
    mode: str = 'random',
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
) -> Pool:
    """
    Delete segments at multiple positions simultaneously.

    Uses region_multiscan() to insert tags at multiple positions, then
    replaces each region's content with deletion characters (or removes it).

    Parameters
    ----------
    pool : Pool or str
        Source pool or sequence string to delete from.
    deletion_length : Integral
        Number of characters to delete at each position.
    num_deletions : Integral
        Number of simultaneous deletions to make.
    deletion_marker : Optional[str], default='-'
        Character to insert at each deletion site. If None, deleted segments
        are removed with no marker.
    spacer_str : str, default=''
        String to insert as a spacer around deletion sites.
    positions : PositionsType, default=None
        Valid positions for deletion starts (0-based). If None, all valid
        positions are used.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : str, default='random'
        Position selection mode: 'random'.
    num_states : Optional[Integral], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool yielding sequences with multiple segments deleted simultaneously.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..region_ops import region_multiscan, replace_region

    # Validate mode
    if mode != 'random':
        raise ValueError(
            f"deletion_multiscan supports only mode='random', got '{mode}'"
        )

    # Validate num_deletions
    if num_deletions < 1:
        raise ValueError(f"num_deletions must be >= 1, got {num_deletions}")

    # Convert string inputs to pools if needed
    pool_obj = from_seq(pool) if isinstance(pool, str) else pool

    # Validate pool has defined seq_length
    bg_length = pool_obj.seq_length
    if bg_length is None:
        raise ValueError("pool must have a defined seq_length")

    # Validate deletion_length
    if deletion_length <= 0:
        raise ValueError(f"deletion_length must be > 0, got {deletion_length}")
    if deletion_length >= bg_length:
        raise ValueError(
            f"deletion_length ({deletion_length}) must be < pool.seq_length ({bg_length})"
        )

    # Check if there's room for num_deletions non-overlapping regions
    min_required_length = num_deletions * deletion_length
    if min_required_length > bg_length:
        raise ValueError(
            f"Cannot fit {num_deletions} non-overlapping deletions of length "
            f"{deletion_length} in sequence of length {bg_length}"
        )

    del_char = deletion_marker if deletion_marker else '-'

    # Generate auto-indexed marker names
    markers = [f'_del_{i}' for i in range(num_deletions)]
    marker_length = int(deletion_length)
    max_position = bg_length - deletion_length

    # Validate positions
    validated_positions = validate_positions(positions, max_position, min_position=0)

    # 1. Insert tags at multiple positions using region_multiscan
    marked = region_multiscan(
        pool_obj,
        regions=markers,
        num_insertions=int(num_deletions),
        positions=validated_positions,
        region_length=marker_length,
        insertion_mode='ordered',
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
    )

    # 2. Build replacement content based on deletion_marker
    if deletion_marker is not None:
        # Fill gap with del_char * deletion_length
        marker_str = del_char * marker_length
        content = from_seq(marker_str)
    else:
        # Simply remove the segment - use empty content
        content = from_seq('')

    # 3. Replace each region's content with deletion content
    result = marked
    for region_name in markers:
        result = replace_region(
            result,
            content,
            region_name,
            iter_order=iter_order,
        )

    return result
