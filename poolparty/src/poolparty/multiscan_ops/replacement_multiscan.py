"""Replacement multiscan operation - replace segments at multiple positions simultaneously."""

from numbers import Integral, Real

from ..pool import Pool
from ..types import Literal, MultiPositionsType, Optional, RegionType, Sequence, Union, beartype
from ..region_ops.region_multiscan import _is_per_insert_positions
from ..utils import validate_positions


@beartype
def replacement_multiscan(
    pool: Union[Pool, str],
    num_replacements: Integral,
    replacement_pools: Union[Pool, Sequence[Pool]],
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
    Replace segments at multiple positions simultaneously.

    Parameters
    ----------
    pool : Pool or str
        Source pool or sequence string to replace segments in.
    num_replacements : Integral
        Number of simultaneous replacements to make.
    replacement_pools : Pool or Sequence[Pool]
        Pool(s) providing replacement content. If a single Pool is provided,
        it will be deepcopied num_replacements-1 times. If a Sequence of Pools
        is provided, its length must equal num_replacements.
    positions : PositionsType, default=None
        Valid positions for replacement starts (0-based).
    region : RegionType, default=None
        Region to constrain the scan to.
    names : Optional[Sequence[str]], default=None
        Custom names for the replacement regions. If None, auto-generated
        (_rep_0, _rep_1, ...).
    insertion_mode : Literal['ordered', 'unordered'], default='ordered'
        How to assign replacement pools to positions.
    min_spacing : Optional[Integral], default=None
        Minimum gap between end of one replacement and start of next.
    max_spacing : Optional[Integral], default=None
        Maximum gap between adjacent replacements. None = unbounded.
    mode : str, default='random'
        Position selection mode: 'random' or 'sequential'.
    num_states : Optional[Integral], default=None
        Number of states. If None, auto-determined for sequential mode.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool yielding sequences with multiple segments replaced simultaneously.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..region_ops import region_multiscan, replace_region

    if num_replacements < 1:
        raise ValueError(f"num_replacements must be >= 1, got {num_replacements}")

    pool_obj = from_seq(pool) if isinstance(pool, str) else pool

    bg_length = pool_obj.seq_length
    if bg_length is None and region is None:
        raise ValueError("pool must have a defined seq_length")

    if isinstance(replacement_pools, Pool):
        pools_list = [replacement_pools]
        for i in range(num_replacements - 1):
            pools_list.append(replacement_pools.deepcopy(name=f"_rep_pool_{i + 1}"))
    else:
        pools_list = list(replacement_pools)
        if len(pools_list) != num_replacements:
            raise ValueError(
                f"replacement_pools length ({len(pools_list)}) must equal "
                f"num_replacements ({num_replacements})"
            )

    replacement_lengths = []
    for i, pool in enumerate(pools_list):
        if pool.seq_length is None:
            raise ValueError(f"replacement_pools[{i}] must have a defined seq_length")
        replacement_lengths.append(pool.seq_length)

    if bg_length is not None:
        min_required_length = sum(replacement_lengths)
        if min_required_length > bg_length:
            raise ValueError(
                f"Cannot fit {num_replacements} non-overlapping replacements of lengths "
                f"{replacement_lengths} in sequence of length {bg_length}"
            )

    markers = list(names) if names is not None else [f"_rep_{i}" for i in range(num_replacements)]
    if len(markers) != num_replacements:
        raise ValueError(
            f"len(names) ({len(markers)}) must equal num_replacements ({num_replacements})"
        )

    # Per-region lengths for region_multiscan
    marker_lengths = replacement_lengths if len(set(replacement_lengths)) > 1 else replacement_lengths[0]

    if _is_per_insert_positions(positions) or region is not None:
        validated_positions = positions
    elif bg_length is not None:
        max_rl = max(replacement_lengths)
        max_position = bg_length - max_rl
        validated_positions = validate_positions(positions, max_position, min_position=0)
    else:
        validated_positions = positions

    marked = region_multiscan(
        pool_obj,
        regions=markers,
        num_insertions=int(num_replacements),
        positions=validated_positions,
        region=region,
        region_length=marker_lengths,
        insertion_mode=insertion_mode,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
    )

    result = marked
    for region_name, rep_pool in zip(markers, pools_list):
        result = replace_region(
            result,
            rep_pool,
            region_name,
            iter_order=iter_order,
        )

    return result
