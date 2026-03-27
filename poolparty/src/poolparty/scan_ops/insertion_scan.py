"""Insertion scan operation - insert a sequence at scanning positions."""

from numbers import Integral, Real

from ..pool import Pool
from ..types import CardsType, ModeType, Optional, PositionsType, RegionType, Union, beartype


@beartype
def insertion_scan(
    pool: Union[Pool, str],
    ins_pool: Union[Pool, str],
    positions: PositionsType = None,
    region: RegionType = None,
    replace: bool = False,
    style: Optional[str] = None,
    prefix: Optional[str] = None,
    prefix_position: Optional[str] = None,
    prefix_insert: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = "insertion_scan",
) -> Pool:
    """
    Insert or replace a sequence at specified scanning positions.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent pool or sequence string.
    ins_pool : Union[Pool, str]
        The insert Pool or sequence string to be inserted.
    positions : PositionsType, default=None
        Positions for insertion/replacement (0-based). If None, all valid positions.
    region : RegionType, default=None
        Region to constrain the scan to. Can be a marker name or [start, stop] interval.
    replace : bool, default=False
        If False, insert at position (output length = bg + ins).
        If True, replace content at position (output length = bg).
    style : Optional[str], default=None
        Style to apply to inserted content (e.g., 'red', 'blue bold').
    prefix : Optional[str], default=None
        Prefix for cartesian product index (e.g., 'ins_' produces 'ins_0', 'ins_1', ...).
    prefix_position : Optional[str], default=None
        Prefix for position index (e.g., 'pos_' produces 'pos_0', 'pos_1', ...).
    prefix_insert : Optional[str], default=None
        Prefix for insert index (e.g., 'ins_' produces 'ins_0', 'ins_1', ...).
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
        A Pool yielding sequences with the insert placed at selected position(s).
    """
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.passthrough import passthrough
    from ..region_ops import region_scan, replace_region

    # Convert string inputs to pools
    pool = (
        from_seq(pool, _factory_name=f"{_factory_name}(from_seq)")
        if isinstance(pool, str)
        else pool
    )
    ins_pool = (
        from_seq(ins_pool, _factory_name=f"{_factory_name}(from_seq)")
        if isinstance(ins_pool, str)
        else ins_pool
    )

    # Validate ins_pool has defined seq_length
    ins_length = ins_pool.seq_length
    if ins_length is None:
        raise ValueError("ins_pool must have a defined seq_length")

    # Validate bg_pool has defined seq_length (only when no region specified)
    bg_length = pool.seq_length
    if bg_length is None and region is None:
        raise ValueError("pool must have a defined seq_length")

    # Capture state references for naming
    ins_pool_state = ins_pool.state
    ins_pool_num_states = ins_pool.num_states

    # Determine marker configuration based on replace mode
    # replace=False: marker_length=0 (insert without removing background)
    # replace=True: marker_length=ins_length (replace background content)
    # Use different marker names to avoid conflicts when both are used in same Party
    marker_name = "_rep" if replace else "_ins"
    marker_length = ins_length if replace else 0

    # 1. Insert tags at scanning positions
    marked = region_scan(
        pool,
        tag_name=marker_name,
        region_length=marker_length,
        positions=positions,
        region=region,
        remove_tags=False,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        cards=cards,
        _factory_name=f"{_factory_name}(region_scan)",
    )
    marked = marked.named(f"{marked.name}:{_factory_name}(intermediate)")

    # Capture position state
    pos_state = marked.operation.state

    # 2. Replace marker with content
    result = replace_region(
        marked,
        ins_pool,
        marker_name,
        iter_order=iter_order,
        _factory_name=f"{_factory_name}(replace_region)",
        _style=style,
    )

    # 3. Add PassthroughOp for custom naming if any prefix is set
    if any([prefix, prefix_position, prefix_insert]):
        num_sites = ins_pool_num_states or 1

        def compute_names():
            # Check if this branch is active
            if not pos_state.is_active:
                return []
            if ins_pool_state is not None and not ins_pool_state.is_active:
                return []

            pos_idx = pos_state.value
            site_idx = ins_pool_state.value if ins_pool_state else 0

            contributions = []
            if prefix:  # Cartesian product index
                W = pos_idx * num_sites + site_idx
                contributions.append(f"{prefix}_{W}")
            if prefix_position:
                contributions.append(f"{prefix_position}_{pos_idx}")
            if prefix_insert:
                contributions.append(f"{prefix_insert}_{site_idx}")
            return contributions

        result = passthrough(
            result,
            _name_fn=compute_names,
            iter_order=iter_order,
            _factory_name=f"{_factory_name}(naming)",
        )

    return result


@beartype
def replacement_scan(
    pool: Union[Pool, str],
    ins_pool: Union[Pool, str],
    positions: PositionsType = None,
    region: RegionType = None,
    style: Optional[str] = None,
    prefix: Optional[str] = None,
    prefix_position: Optional[str] = None,
    prefix_insert: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = "replacement_scan",
) -> Pool:
    """Replace a segment with insert at specified scanning positions.

    Equivalent to ``insertion_scan(..., replace=True)``.
    See :func:`insertion_scan` for full parameter documentation.
    """
    return insertion_scan(
        pool=pool,
        ins_pool=ins_pool,
        positions=positions,
        region=region,
        replace=True,
        style=style,
        prefix=prefix,
        prefix_position=prefix_position,
        prefix_insert=prefix_insert,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        cards=cards,
        _factory_name=_factory_name if _factory_name is not None else "replacement_scan",
    )
