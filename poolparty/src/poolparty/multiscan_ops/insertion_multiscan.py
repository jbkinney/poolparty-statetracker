"""Insertion/replacement multiscan operation - insert or replace at multiple positions."""

from numbers import Integral, Real

from ..pool import Pool
from ..types import CardsType, Literal, ModeType, MultiPositionsType, Optional, RegionType, Sequence, Union, beartype
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
    replace: bool = False,
    style: Optional[str] = None,
    insertion_mode: Literal["ordered", "unordered"] = "ordered",
    min_spacing: Optional[Integral] = None,
    max_spacing: Optional[Integral] = None,
    prefix: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = None,
) -> Pool:
    """
    Insert or replace sequences at multiple positions simultaneously.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent pool or sequence string.
    num_insertions : Integral
        Number of simultaneous insertions/replacements to make.
    insertion_pools : Union[Pool, Sequence[Pool]]
        Pool(s) providing content. If a single Pool is provided,
        it will be deepcopied ``num_insertions - 1`` times. If a Sequence of Pools
        is provided, its length must equal ``num_insertions``.
    positions : MultiPositionsType, default=None
        Valid positions (0-based). Can be a flat list (shared across all
        insertions) or a list of per-insertion sublists. If None, all valid
        positions are used.
    region : RegionType, default=None
        Region to constrain the scan to. Can be a marker name or [start, stop] interval.
    names : Optional[Sequence[str]], default=None
        Custom names for the insertion regions. If None, auto-generated
        (``_ins_0``, ``_ins_1``, ...).
    replace : bool, default=False
        If True, replace existing content at each position (region_length = pool seq_length).
        If False, insert at zero-width positions.
    style : Optional[str], default=None
        Style to apply to inserted/replaced content (e.g., 'red', 'blue bold').
    insertion_mode : Literal['ordered', 'unordered'], default='ordered'
        How to assign pools to positions. ``'ordered'`` preserves position
        order; ``'unordered'`` uses all permutations.
    min_spacing : Optional[Integral], default=None
        Minimum gap between adjacent positions.
    max_spacing : Optional[Integral], default=None
        Maximum gap between adjacent positions. None = unbounded.
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
        Design card keys to include. Available keys: ``'combination_index'``,
        ``'starts'``, ``'ends'``, ``'names'``, ``'region_seqs'``.

    Returns
    -------
    Pool
        A Pool yielding sequences with multiple insertions or replacements.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..region_ops import region_multiscan, replace_region

    if _factory_name is None:
        _factory_name = "replacement_multiscan" if replace else "insertion_multiscan"

    count_label = "num_replacements" if replace else "num_insertions"
    pools_label = "replacement_pools" if replace else "insertion_pools"

    if num_insertions < 1:
        raise ValueError(f"{count_label} must be >= 1, got {num_insertions}")

    pool_obj = from_seq(pool) if isinstance(pool, str) else pool

    bg_length = pool_obj.seq_length
    if bg_length is None and region is None:
        raise ValueError("pool must have a defined seq_length")

    default_prefix = "_rep" if replace else "_ins"

    if isinstance(insertion_pools, Pool):
        pools_list = [insertion_pools]
        for i in range(num_insertions - 1):
            pools_list.append(insertion_pools.deepcopy(name=f"{default_prefix}_pool_{i + 1}"))
    else:
        pools_list = list(insertion_pools)
        if len(pools_list) != num_insertions:
            raise ValueError(
                f"{pools_label} length ({len(pools_list)}) must equal "
                f"{count_label} ({num_insertions})"
            )

    pool_lengths = []
    for i, p in enumerate(pools_list):
        if p.seq_length is None:
            raise ValueError(f"{pools_label}[{i}] must have a defined seq_length")
        pool_lengths.append(p.seq_length)

    if replace:
        if bg_length is not None:
            min_required_length = sum(pool_lengths)
            if min_required_length > bg_length:
                raise ValueError(
                    f"Cannot fit {num_insertions} non-overlapping replacements of lengths "
                    f"{pool_lengths} in sequence of length {bg_length}"
                )

        marker_length = pool_lengths if len(set(pool_lengths)) > 1 else pool_lengths[0]
    else:
        marker_length = 0

    markers = (
        list(names) if names is not None
        else [f"{default_prefix}_{i}" for i in range(num_insertions)]
    )
    if len(markers) != num_insertions:
        raise ValueError(f"len(names) ({len(markers)}) must equal {count_label} ({num_insertions})")

    if _is_per_insert_positions(positions) or region is not None:
        validated_positions = positions
    elif bg_length is not None:
        if replace:
            max_rl = max(pool_lengths)
            max_position = bg_length - max_rl
        else:
            max_position = bg_length
        validated_positions = validate_positions(positions, max_position, min_position=0)
    else:
        validated_positions = positions

    marked = region_multiscan(
        pool_obj,
        tag_names=markers,
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
        cards=cards,
        _factory_name=f"{_factory_name}(region_multiscan)",
    )

    result = marked
    for region_name, ins_pool in zip(markers, pools_list):
        result = replace_region(
            result,
            ins_pool,
            region_name,
            iter_order=iter_order,
            _factory_name=f"{_factory_name}(replace_region)",
            _style=style,
        )

    return result


@beartype
def replacement_multiscan(
    pool: Union[Pool, str],
    num_replacements: Integral,
    replacement_pools: Union[Pool, Sequence[Pool]],
    positions: MultiPositionsType = None,
    region: RegionType = None,
    names: Optional[Sequence[str]] = None,
    style: Optional[str] = None,
    insertion_mode: Literal["ordered", "unordered"] = "ordered",
    min_spacing: Optional[Integral] = None,
    max_spacing: Optional[Integral] = None,
    prefix: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = "replacement_multiscan",
) -> Pool:
    """Replace segments at multiple positions simultaneously.

    Equivalent to ``insertion_multiscan(..., replace=True)``.
    See :func:`insertion_multiscan` for full parameter documentation.
    """
    return insertion_multiscan(
        pool=pool,
        num_insertions=num_replacements,
        insertion_pools=replacement_pools,
        positions=positions,
        region=region,
        names=names,
        replace=True,
        style=style,
        insertion_mode=insertion_mode,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        cards=cards,
        _factory_name=_factory_name,
    )
