"""Deletion multiscan operation - delete segments at multiple positions simultaneously."""
from numbers import Integral, Real

from ..types import Union, Optional, PositionsType, beartype
from ..seq_utils import validate_positions
from ..pool import Pool


@beartype
def deletion_multiscan(
    bg_pool: Union[Pool, str],
    deletion_length: Integral,
    num_deletions: Integral,
    deletion_marker: Optional[str] = '-',
    spacer_str: str = '',
    positions: PositionsType = None,
    seq_name_prefix: Optional[str] = None,
    mode: str = 'random',
    num_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Delete segments at multiple positions simultaneously.

    Uses marker_multiscan() to insert markers at multiple positions, then
    replaces each marker's content with deletion characters (or removes it).

    Parameters
    ----------
    bg_pool : Pool or str
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
    mode : str, default='random'
        Position selection mode: 'random'.
    num_states : Optional[Integral], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operations.
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operations.

    Returns
    -------
    Pool
        A Pool yielding sequences with multiple segments deleted simultaneously.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..marker_ops import marker_multiscan, replace_marker_content

    # Validate mode
    if mode != 'random':
        raise ValueError(
            f"deletion_multiscan supports only mode='random', got '{mode}'"
        )

    # Validate num_deletions
    if num_deletions < 1:
        raise ValueError(f"num_deletions must be >= 1, got {num_deletions}")

    # Convert string inputs to pools if needed
    bg_pool = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool

    # Validate bg_pool has defined seq_length
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

    # 1. Insert markers at multiple positions using marker_multiscan
    marked = marker_multiscan(
        bg_pool,
        markers=markers,
        num_insertions=int(num_deletions),
        positions=validated_positions,
        marker_length=marker_length,
        insertion_mode='ordered',
        seq_name_prefix=seq_name_prefix,
        mode=mode,
        num_states=num_states,
        op_name=op_name,
        op_iter_order=op_iter_order,
    )

    # 2. Build replacement content based on deletion_marker
    if deletion_marker is not None:
        # Fill gap with del_char * deletion_length
        marker_str = del_char * marker_length
        content = from_seq(marker_str)
        # Apply spacer_str via replace_marker_content
        effective_spacer = spacer_str
    else:
        # Simply remove the segment - use spacer_str as the content (once)
        content = from_seq(spacer_str)
        effective_spacer = ''  # Don't add additional spacers

    # 3. Replace each marker's content with deletion content
    result = marked
    for marker_name in markers:
        result = replace_marker_content(
            result,
            content,
            marker_name,
            spacer_str=effective_spacer,
            name=None,  # Only set name on final result
            op_name=op_name,
            iter_order=None,  # Only set iter_order on final result
            op_iter_order=op_iter_order,
        )

    # Set name and iter_order on final result if provided
    if name is not None:
        result._name = name
    if iter_order is not None:
        result._iter_order = iter_order

    return result
