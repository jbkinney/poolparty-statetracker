"""Shuffle scan operation - shuffle characters within a window at scanning positions."""
from numbers import Integral, Real

from ..types import Union, ModeType, Optional, PositionsType, RegionType, beartype
from ..party import get_active_party
from ..pool import Pool


@beartype
def shuffle_scan(
    pool: Union[Pool, str],
    shuffle_length: Integral,
    positions: PositionsType = None,
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    shuffles_per_position: Integral = 1,
    spacer_str: str = '',
    mark_changes: Optional[bool] = None,
    seq_name_prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = 'shuffle_scan',
) -> Pool:
    """
    Shuffle characters within a window at specified scanning positions.

    Parameters
    ----------
    pool : Pool or str
        Source pool or sequence string to shuffle regions of.
    shuffle_length : Integral
        Length of the region to shuffle at each position.
    positions : PositionsType, default=None
        Positions to consider for the start of the shuffle region (0-based).
    region : RegionType, default=None
        Region to constrain the scan to. Can be a marker name (str) or [start, stop].
    shuffles_per_position : Integral, default=1
        Number of shuffles to perform at each position.
    mark_changes : Optional[bool], default=None
        If True, apply swapcase() to the shuffled region.
    mode : ModeType, default='random'
        Selection mode: 'random', 'sequential', or 'hybrid'.

    Returns
    -------
    Pool
        A Pool yielding sequences where a region of the specified length is shuffled
        at each allowed position.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..base_ops.shuffle_seq import shuffle_seq
    from ..marker_ops import marker_scan

    # Convert string inputs to pools
    pool = from_seq(pool, _factory_name=f'{_factory_name}(from_seq)') if isinstance(pool, str) else pool

    # Validate pool has defined seq_length (only when no region specified)
    bg_length = pool.seq_length
    if bg_length is None and region is None:
        raise ValueError("pool must have a defined seq_length")

    # Validate shuffle_length
    if shuffle_length <= 0:
        raise ValueError(f"shuffle_length must be > 0, got {shuffle_length}")
    if bg_length is not None and shuffle_length >= bg_length:
        raise ValueError(
            f"shuffle_length ({shuffle_length}) must be < pool.seq_length ({bg_length})"
        )

    # Resolve defaults from party
    party = get_active_party()
    if remove_marker is None:
        remove_marker = party.get_default('remove_marker', True) if party else True

    marker_name = '_shuf'
    marker_length = int(shuffle_length)

    # 1. Insert marker at scanning positions
    marked = marker_scan(
        pool,
        marker=marker_name,
        marker_length=marker_length,
        positions=positions,
        region=region,
        remove_marker=False,  # Keep outer region marker for now
        seq_name_prefix=seq_name_prefix,
        mode=mode,
        num_states=num_states,
        op_name=op_name,
        op_iter_order=op_iter_order,
        _factory_name=f'{_factory_name}(marker_scan)',
    )

    # 2. Shuffle the marked region directly using shuffle_seq with region='_shuf'
    # spacer_str is handled by Operation base class in wrapped_compute_seq_from_card
    return shuffle_seq(
        marked,
        region=marker_name,
        remove_marker=True,  # Always remove the internal _shuf marker
        spacer_str=spacer_str,
        mark_changes=mark_changes,
        mode='random',
        num_states=shuffles_per_position,
        name=name,
        iter_order=iter_order,
        op_iter_order=-1,
        _factory_name=f'{_factory_name}(shuffle_seq)',
    )
