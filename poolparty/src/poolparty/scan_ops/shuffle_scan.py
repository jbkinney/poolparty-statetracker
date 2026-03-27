"""Shuffle scan operation - shuffle characters within a window at scanning positions."""

from numbers import Integral, Real

from ..pool import Pool
from ..types import CardsType, Literal, ModeType, Optional, PositionsType, RegionType, Union, beartype


@beartype
def shuffle_scan(
    pool: Union[Pool, str],
    shuffle_length: Integral,
    positions: PositionsType = None,
    region: RegionType = None,
    shuffle_type: Literal["mono", "dinuc"] = "mono",
    shuffles_per_position: Integral = 1,
    prefix: Optional[str] = None,
    prefix_position: Optional[str] = None,
    prefix_shuffle: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    style: Optional[str] = None,
    iter_order: Optional[Real] = None,
    cards: Optional[tuple[CardsType, CardsType]] = None,
    _factory_name: Optional[str] = "shuffle_scan",
) -> Pool:
    """
    Shuffle characters within a window at specified scanning positions.

    Parameters
    ----------
    pool : Union[Pool, str]
        Parent pool or sequence string.
    shuffle_length : Integral
        Length of the region to shuffle at each position.
    positions : PositionsType, default=None
        Positions to consider for the start of the shuffle region (0-based).
    region : RegionType, default=None
        Region to constrain the scan to. Can be a marker name or [start, stop] interval.
    shuffle_type : Literal["mono", "dinuc"], default="mono"
        Type of shuffle to perform:
        - ``"mono"``: random permutation preserving mononucleotide composition.
        - ``"dinuc"``: Euler-path shuffle preserving dinucleotide frequencies.
          The first and last characters of each window are fixed.
    shuffles_per_position : Integral, default=1
        Number of shuffles to perform at each position.
    prefix : Optional[str], default=None
        Prefix for cartesian product index (e.g., 'shuf' produces 'shuf_0', 'shuf_1', ...).
    prefix_position : Optional[str], default=None
        Prefix for position index (e.g., 'pos' produces 'pos_0', 'pos_1', ...).
    prefix_shuffle : Optional[str], default=None
        Prefix for shuffle variant index (e.g., 'var' produces 'var_0', 'var_1', ...).
    mode : ModeType, default='random'
        Selection mode: 'random' or 'sequential'.
    num_states : Optional[Integral], default=None
        Number of states. In sequential mode, overrides the computed count
        (cycling if greater, clipping if less). In random mode, if None
        defaults to 1 (pure random sampling).
    style : Optional[str], default=None
        Style to apply to shuffled characters (e.g., 'purple', 'red bold').
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    cards : Optional[tuple[CardsType, CardsType]], default=None
        Design card keys as a 2-tuple ``(scan_cards, shuffle_cards)``.
        Scan keys: ``'position_index'``, ``'start'``, ``'end'``, ``'name'``,
        ``'region_seq'``. Shuffle keys: ``'permutation'``.

    Returns
    -------
    Pool
        A Pool yielding sequences where a region of the specified length is shuffled
        at each allowed position.
    """
    from ..base_ops.shuffle_seq import shuffle_seq
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.passthrough import passthrough
    from ..region_ops import region_scan

    # Convert string inputs to pools
    pool = (
        from_seq(pool, _factory_name=f"{_factory_name}(from_seq)")
        if isinstance(pool, str)
        else pool
    )

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

    # Resolve cards
    cards_scan, cards_shuf = cards if cards else (None, None)

    region_name = "_shuf"
    region_length = int(shuffle_length)

    # 1. Insert tags at scanning positions
    marked = region_scan(
        pool,
        tag_name=region_name,
        region_length=region_length,
        positions=positions,
        region=region,
        remove_tags=False,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
        cards=cards_scan,
        _factory_name=f"{_factory_name}(region_scan)",
    )

    # Capture position state
    pos_state = marked.operation.state

    # 2. Shuffle the marked region directly using shuffle_seq with region='_shuf'
    result = shuffle_seq(
        marked,
        region=region_name,
        shuffle_type=shuffle_type,
        _remove_tags=True,  # Remove _shuf tags
        style=style,
        mode="random",
        num_states=shuffles_per_position,
        iter_order=-1,
        cards=cards_shuf,
        _factory_name=f"{_factory_name}(shuffle_seq)",
    )

    # Capture shuffle state
    shuffle_state = result.operation.state

    # 3. Add PassthroughOp for custom naming if any prefix is set
    if any([prefix, prefix_position, prefix_shuffle]):
        num_shuffles = int(shuffles_per_position) if shuffles_per_position else 1

        def compute_names():
            # Check if this branch is active
            if not pos_state.is_active:
                return []
            if shuffle_state is not None and not shuffle_state.is_active:
                return []

            pos_idx = pos_state.value
            shuffle_idx = shuffle_state.value if shuffle_state else 0

            contributions = []
            if prefix:  # Cartesian product index
                W = pos_idx * num_shuffles + shuffle_idx
                contributions.append(f"{prefix}_{W}")
            if prefix_position:
                contributions.append(f"{prefix_position}_{pos_idx}")
            if prefix_shuffle:
                contributions.append(f"{prefix_shuffle}_{shuffle_idx}")
            return contributions

        result = passthrough(
            result,
            _name_fn=compute_names,
            iter_order=iter_order,
            _factory_name=f"{_factory_name}(naming)",
        )

    return result
