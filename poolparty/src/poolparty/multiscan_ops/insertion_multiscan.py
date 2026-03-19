"""Insertion multiscan operation - insert sequences at multiple positions simultaneously."""

from numbers import Integral, Real

from ..pool import Pool
from ..types import Literal, MultiPositionsType, Optional, RegionType, Sequence, Union, beartype
from ..region_ops.region_multiscan import _is_per_insert_positions
from ..utils import validate_positions


@beartype
def insertion_multiscan(
    pool: Union[Pool, str],
    num_insertions: Integral,
    insertion_pools: Union[Pool, Sequence[Pool]],
    positions: MultiPositionsType = None,
    region: RegionType = None,
    names: Optional[Sequence[str]] = None,
    insertion_mode: Literal["ordered", "unordered"] = "ordered",
    min_spacing: Optional[Integral] = None,
    max_spacing: Optional[Integral] = None,
    prefix: Optional[str] = None,
    mode: str = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
) -> Pool:
    """
    Insert sequences at multiple positions simultaneously.

    Parameters
    ----------
    pool : Pool or str
        Source pool or sequence string to insert into.
    num_insertions : Integral
        Number of simultaneous insertions to make.
    insertion_pools : Pool or Sequence[Pool]
        Pool(s) providing insertion content. If a single Pool is provided,
        it will be deepcopied num_insertions-1 times. If a Sequence of Pools
        is provided, its length must equal num_insertions.
    positions : PositionsType, default=None
        Valid positions for insertions (0-based).
    region : RegionType, default=None
        Region to constrain the scan to.
    names : Optional[Sequence[str]], default=None
        Custom names for the insertion regions. If None, auto-generated
        (_ins_0, _ins_1, ...).
    insertion_mode : Literal['ordered', 'unordered'], default='ordered'
        How to assign insertion pools to positions.
    min_spacing : Optional[Integral], default=None
        Minimum gap between adjacent insertion positions.
    max_spacing : Optional[Integral], default=None
        Maximum gap between adjacent insertion positions. None = unbounded.
    mode : str, default='random'
        Position selection mode: 'random' or 'sequential'.
    num_states : Optional[Integral], default=None
        Number of states. If None, auto-determined for sequential mode.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool yielding sequences with multiple insertions made simultaneously.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..region_ops import region_multiscan, replace_region

    if num_insertions < 1:
        raise ValueError(f"num_insertions must be >= 1, got {num_insertions}")

    pool_obj = from_seq(pool) if isinstance(pool, str) else pool

    bg_length = pool_obj.seq_length
    if bg_length is None and region is None:
        raise ValueError("pool must have a defined seq_length")

    if isinstance(insertion_pools, Pool):
        pools_list = [insertion_pools]
        for i in range(num_insertions - 1):
            pools_list.append(insertion_pools.deepcopy(name=f"_ins_pool_{i + 1}"))
    else:
        pools_list = list(insertion_pools)
        if len(pools_list) != num_insertions:
            raise ValueError(
                f"insertion_pools length ({len(pools_list)}) must equal "
                f"num_insertions ({num_insertions})"
            )

    for i, pool in enumerate(pools_list):
        if pool.seq_length is None:
            raise ValueError(f"insertion_pools[{i}] must have a defined seq_length")

    markers = list(names) if names is not None else [f"_ins_{i}" for i in range(num_insertions)]
    if len(markers) != num_insertions:
        raise ValueError(f"len(names) ({len(markers)}) must equal num_insertions ({num_insertions})")
    marker_length = 0

    if _is_per_insert_positions(positions) or region is not None:
        validated_positions = positions
    elif bg_length is not None:
        max_position = bg_length
        validated_positions = validate_positions(positions, max_position, min_position=0)
    else:
        validated_positions = positions

    marked = region_multiscan(
        pool_obj,
        regions=markers,
        num_insertions=int(num_insertions),
        positions=validated_positions,
        region=region,
        region_length=marker_length,
        insertion_mode=insertion_mode,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
    )

    result = marked
    for region_name, ins_pool in zip(markers, pools_list):
        result = replace_region(
            result,
            ins_pool,
            region_name,
            iter_order=iter_order,
        )

    return result
