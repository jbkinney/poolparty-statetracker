"""DeletionScan - delete segments from background at scanning positions."""
from ..types import Union, ModeType, Optional, Integral, Real, PositionsType, beartype
from ..seq_utils import validate_positions
from ..pool import Pool


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
    """Delete a segment from the background at scanning positions."""
    from .from_seqs import from_seqs
    from .seq_slice import seq_slice
    from .join import join
    from .breakpoint_scan import breakpoint_scan

    # Convert string input to pool if needed
    if isinstance(bg_pool, str):
        bg_pool = from_seqs([bg_pool], mode='fixed')

    # Validate that bg_pool has defined seq_length
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

    # Validate positions (valid range: 0 to bg_length - deletion_length)
    max_position = bg_length - deletion_length
    validated_positions = validate_positions(positions, max_position, min_position=0)

    # Split background at breakpoint positions
    left, right = breakpoint_scan(
        pool=bg_pool,
        num_breakpoints=1,
        positions=validated_positions,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        op_iter_order=op_iter_order,
    )

    # Clip the right segment by removing the first deletion_length characters
    right_clipped = seq_slice(right, slice(deletion_length, None, None))

    if deletion_marker is not None:
        marker_seq = deletion_marker * deletion_length
        marker_pool = from_seqs([marker_seq], mode='fixed')
        pools_list = [left, marker_pool, right_clipped]
    else:
        pools_list = [left, right_clipped]
    result = join(
        pools_list,
        spacer_str=spacer_str,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
    )
    return result

