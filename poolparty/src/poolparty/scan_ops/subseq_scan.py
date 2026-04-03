"""Subsequence scan operation - extract subsequences at scanning positions."""

from numbers import Integral, Real

from ..pool import Pool
from ..types import CardsType, ModeType, Optional, PositionsType, RegionType, Union, beartype
from ..utils import validate_positions


@beartype
def subseq_scan(
    pool: Union[Pool, str],
    subseq_length: Integral,
    positions: PositionsType = None,
    region: RegionType = None,
    prefix: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = "subseq_scan",
) -> Pool:
    """
    Extract subsequences of a specified length at scanning positions.

    Scans a region across the pool and extracts the region content,
    returning subsequences at each valid position.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent pool or sequence string.
    subseq_length : Integral
        Length of subsequence to extract at each position.
    positions : PositionsType, default=None
        Positions to consider for the start of extraction (0-based).
        If None, all valid positions are used.
    region : RegionType, default=None
        Region to constrain the scan to. Can be a marker name or [start, stop] interval.
        If specified, positions are relative to the region start.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : ModeType, default='random'
        Selection mode: 'random' or 'sequential'.
    num_states : Optional[Integral], default=None
        Number of states. In sequential mode, overrides the computed count
        (cycling if greater, clipping if less). In random mode, if None
        defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    cards : CardsType, default=None
        Design card keys to include. Available keys: ``'position_index'``,
        ``'start'``, ``'end'``, ``'name'``, ``'region_seq'``.

    Returns
    -------
    Pool
        A Pool yielding subsequences extracted at each allowed position.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..region_ops import extract_region, insert_tags

    # Convert string input to pool if needed
    pool = from_seq(pool, _factory_name=f"{_factory_name}(from_seq)") if isinstance(pool, str) else pool

    # If region is specified, extract subsequences only from that region
    if region is not None:
        if isinstance(region, str):
            # Region is a region name - extract region content first
            region_content = extract_region(pool, region)
            # Apply subseq_scan to the region content
            return _subseq_scan_impl(
                pool=region_content,
                subseq_length=subseq_length,
                positions=positions,
                prefix=prefix,
                mode=mode,
                num_states=num_states,
                iter_order=iter_order,
                cards=cards,
                _factory_name=_factory_name,
            )
        else:
            # Region is [start, stop] - insert temporary tags, extract, then scan
            temp_region = "_subseq_scan_region"
            marked_pool = insert_tags(
                pool,
                region_name=temp_region,
                start=int(region[0]),
                stop=int(region[1]),
            )
            region_content = extract_region(marked_pool, temp_region)
            return _subseq_scan_impl(
                pool=region_content,
                subseq_length=subseq_length,
                positions=positions,
                prefix=prefix,
                mode=mode,
                num_states=num_states,
                iter_order=iter_order,
                cards=cards,
                _factory_name=_factory_name,
            )

    # No region specified - apply to entire pool
    return _subseq_scan_impl(
        pool=pool,
        subseq_length=subseq_length,
        positions=positions,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        cards=cards,
        _factory_name=_factory_name,
    )


def _subseq_scan_impl(
    pool: Pool,
    subseq_length: Integral,
    positions: PositionsType,
    prefix: Optional[str],
    mode: ModeType,
    num_states: Optional[Integral],
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = "subseq_scan",
) -> Pool:
    """Core subseq scan implementation without region handling."""
    from ..region_ops import extract_region, region_scan

    # Validate pool has defined seq_length
    pool_length = pool.seq_length
    if pool_length is None:
        raise ValueError("pool must have a defined seq_length")

    # Validate subseq_length
    if subseq_length <= 0:
        raise ValueError(f"subseq_length must be > 0, got {subseq_length}")
    if subseq_length > pool_length:
        raise ValueError(f"subseq_length ({subseq_length}) must be <= pool.seq_length ({pool_length})")

    # Calculate max position for region tag placement
    region_name = "_subseq"
    region_length = int(subseq_length)
    max_position = pool_length - subseq_length

    # Validate positions
    validated_positions = validate_positions(positions, max_position, min_position=0)

    # 1. Scan region tags across pool at specified positions
    marked = region_scan(
        pool,
        tag_name=region_name,
        region_length=region_length,
        positions=validated_positions,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        cards=cards,
        _factory_name=f"{_factory_name}(region_scan)",
    )

    # 2. Extract region content as the result
    result = extract_region(
        marked,
        region_name,
        iter_order=iter_order,
    )

    return result
