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
    shuffles_per_position: Integral = 1,
    prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
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
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        _factory_name=f'{_factory_name}(marker_scan)',
    )

    # 2. Shuffle the marked region directly using shuffle_seq with region='_shuf'
    return shuffle_seq(
        marked,
        region=marker_name,
        mode='random',
        num_states=shuffles_per_position,
        iter_order=-1,
        _factory_name=f'{_factory_name}(shuffle_seq)',
    )
