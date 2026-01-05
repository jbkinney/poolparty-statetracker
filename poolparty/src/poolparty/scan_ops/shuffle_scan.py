"""Shuffle scan operation - shuffle characters within a window at scanning positions."""
from numbers import Integral, Real

from ..types import Union, ModeType, Optional, PositionsType, beartype
from ..seq_utils import validate_positions
from ..party import get_active_party
from ..pool import Pool


@beartype
def shuffle_scan(
    bg_pool: Union[Pool, str],
    shuffle_length: Integral,
    positions: PositionsType = None,
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
    from ..base_ops.seq_shuffle import seq_shuffle
    from ..marker_ops import marker_scan, apply_at_marker

    # Convert string inputs to pools if needed
    bg_pool = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool

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

    # Resolve mark_changes from party defaults if not explicitly set
    party = get_active_party()
    if mark_changes is None:
        mark_changes = party.get_default('mark_changes', False) if party else False

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
    # Note: seq_shuffle only supports 'random' mode for the actual shuffling.
    # The 'mode' parameter controls position selection via marker_scan above.
    def shuffle_transform(content_pool):
        shuffled = seq_shuffle(content_pool, 
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
