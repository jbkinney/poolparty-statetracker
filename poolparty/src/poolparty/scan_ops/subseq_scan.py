"""Subsequence scan operation - extract subsequences at scanning positions."""
from numbers import Integral, Real

from ..types import Union, Literal, ModeType, Optional, PositionsType, RegionType, beartype
from ..utils import validate_positions
from ..party import get_active_party
from ..pool import Pool


@beartype
def subseq_scan(
    pool: Union[Pool, str],
    seq_length: Integral,
    positions: PositionsType = None,
    region: RegionType = None,
    strand: Literal['+', '-', 'both'] = '+',
    prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
) -> Pool:
    """
    Extract subsequences of a specified length at scanning positions.

    Scans a region across the pool and extracts the region content,
    returning subsequences at each valid position.

    Parameters
    ----------
    pool : Pool or str
        Source pool or sequence string to extract subsequences from.
    seq_length : Integral
        Length of subsequence to extract at each position.
    positions : PositionsType, default=None
        Positions to consider for the start of extraction (0-based).
        If None, all valid positions are used.
    region : RegionType, default=None
        Region to constrain the scan to. Can be a region name (str) or [start, stop].
        If specified, positions are relative to the region start.
    strand : Literal['+', '-', 'both'], default='+'
        Strand for extraction: '+', '-', or 'both'.
        If '-', content is reverse-complemented.
        If 'both', creates 2x states scanning both strands.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : ModeType, default='random'
        Position selection mode: 'random' or 'sequential'.
    num_states : Optional[Integral], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool yielding subsequences extracted at each allowed position.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..region_ops import region_scan, extract_region, insert_tags

    # Convert string input to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool

    # If region is specified, extract subsequences only from that region
    if region is not None:
        if isinstance(region, str):
            # Region is a region name - extract region content first
            region_content = extract_region(pool, region)
            # Apply subseq_scan to the region content
            return _subseq_scan_impl(
                pool=region_content,
                seq_length=seq_length,
                positions=positions,
                strand=strand,
                prefix=prefix,
                mode=mode,
                num_states=num_states,
                iter_order=iter_order,
            )
        else:
            # Region is [start, stop] - insert temporary tags, extract, then scan
            temp_region = '_subseq_scan_region'
            marked_pool = insert_tags(
                pool,
                region_name=temp_region,
                start=int(region[0]),
                stop=int(region[1]),
            )
            region_content = extract_region(marked_pool, temp_region)
            return _subseq_scan_impl(
                pool=region_content,
                seq_length=seq_length,
                positions=positions,
                strand=strand,
                prefix=prefix,
                mode=mode,
                num_states=num_states,
                iter_order=iter_order,
            )

    # No region specified - apply to entire pool
    return _subseq_scan_impl(
        pool=pool,
        seq_length=seq_length,
        positions=positions,
        strand=strand,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
    )


def _subseq_scan_impl(
    pool: Pool,
    seq_length: Integral,
    positions: PositionsType,
    strand: Literal['+', '-', 'both'],
    prefix: Optional[str],
    mode: ModeType,
    num_states: Optional[Integral],
    iter_order: Optional[Real] = None,
) -> Pool:
    """Core subseq scan implementation without region handling."""
    from ..region_ops import region_scan, extract_region

    # Validate pool has defined seq_length
    pool_length = pool.seq_length
    if pool_length is None:
        raise ValueError("pool must have a defined seq_length")

    # Validate seq_length
    if seq_length <= 0:
        raise ValueError(f"seq_length must be > 0, got {seq_length}")
    if seq_length > pool_length:
        raise ValueError(
            f"seq_length ({seq_length}) must be <= pool.seq_length ({pool_length})"
        )

    # Calculate max position for region tag placement
    region_name = '_subseq'
    region_length = int(seq_length)
    max_position = pool_length - seq_length

    # Validate positions
    validated_positions = validate_positions(positions, max_position, min_position=0)

    # 1. Scan region tags across pool at specified positions
    marked = region_scan(
        pool,
        region=region_name,
        region_length=region_length,
        positions=validated_positions,
        strand=strand,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
    )

    # 2. Extract region content as the result
    result = extract_region(
        marked,
        region_name,
        iter_order=iter_order,
    )

    return result
