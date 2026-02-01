"""Mutagenize scan operation - apply mutagenesis within a window at scanning positions."""

from numbers import Integral, Real

from ..pool import Pool
from ..types import ModeType, Optional, PositionsType, RegionType, Sequence, Union, beartype


@beartype
def mutagenize_scan(
    pool: Union[Pool, str],
    mutagenize_length: Integral,
    num_mutations: Optional[Integral] = None,
    mutation_rate: Optional[Real] = None,
    positions: PositionsType = None,
    region: RegionType = None,
    prefix: Optional[Union[str, Sequence[str]]] = None,
    mode: Union[ModeType, tuple[ModeType, ModeType]] = "random",
    num_states: Optional[Union[Integral, Sequence[Integral]]] = None,
    iter_order: Optional[Union[Real, Sequence[Real]]] = None,
    _factory_name: Optional[str] = "mutagenize_scan",
) -> Pool:
    """
    Apply mutagenesis within a window at specified scanning positions.

    Parameters
    ----------
    pool : Pool or str
        Source pool or sequence string to mutagenize regions of.
    mutagenize_length : Integral
        Length of the region to mutagenize at each position.
    num_mutations : Optional[Integral], default=None
        Fixed number of mutations to apply (mutually exclusive with mutation_rate).
    mutation_rate : Optional[Real], default=None
        Probability of mutation at each position (mutually exclusive with num_mutations).
    positions : PositionsType, default=None
        Positions to consider for the start of the mutagenize region (0-based).
        If None, all valid positions are used.
    region : RegionType, default=None
        Region to constrain the scan to. Can be a marker name (str) or [start, stop].
        If specified, positions are relative to the region start.
    prefix : Optional[Union[str, Sequence[str]]], default=None
        Prefix for sequence names.
        If sequence, first element is for scanning positions, second element is for mutagenization.
    mode : Union[ModeType, Sequence[ModeType]], default='random'
        Selection mode for scanning positions: 'random' or 'sequential'.
        If sequence, first element is for scanning positions, second element is for mutagenization.
    num_states : Optional[Union[Integral, Sequence[Integral]]], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
        If sequence, first element is for scanning positions, second element is for mutagenization.
    iter_order : Optional[Union[Real, Sequence[Real]]], default=None
        Iteration order priority for the Operation.
        If sequence, first element is for scanning positions, second element is for mutagenization.
    _factory_name: Optional[str], default=None
        Sets default name of the resulting operation

    Returns
    -------
    Pool
        A Pool yielding sequences where a region of the specified length is mutagenized
        at each allowed position.
    """
    from ..base_ops.mutagenize import mutagenize
    from ..fixed_ops.from_seq import from_seq
    from ..region_ops import region_scan

    # Convert string inputs to pools if needed
    pool = (
        from_seq(pool, _factory_name=f"{_factory_name}(from_seq)")
        if isinstance(pool, str)
        else pool
    )

    # Validate num_mutations/mutation_rate
    if num_mutations is None and mutation_rate is None:
        raise ValueError("Either num_mutations or mutation_rate must be provided")
    if num_mutations is not None and mutation_rate is not None:
        raise ValueError("Only one of num_mutations or mutation_rate can be provided, not both")

    # Determine marker configuration
    marker_name = "_mut"
    marker_length = mutagenize_length

    # Resolve mode - expand single value to tuple of two
    # Note: str is a Sequence, so check for str first
    if mode is None or isinstance(mode, str):
        mode = (mode, mode)
    elif isinstance(mode, Sequence) and len(mode) != 2:
        raise ValueError("mode must be a sequence of length 2")
    mode_scan, mode_mut = mode[0], mode[1]

    # Resolve num_states - expand single value to tuple of two
    if num_states is None or isinstance(num_states, Integral):
        num_states = (num_states, num_states)
    elif isinstance(num_states, Sequence) and len(num_states) != 2:
        raise ValueError("num_states must be a sequence of length 2")
    num_states_scan, num_states_mut = num_states[0], num_states[1]

    # Resolve prefix - expand single value to tuple of two
    # Note: str is a Sequence, so check for str first
    if prefix is None or isinstance(prefix, str):
        prefix = (prefix, prefix)
    elif isinstance(prefix, Sequence) and len(prefix) != 2:
        raise ValueError("prefix must be a sequence of length 2")
    prefix_scan, prefix_mut = prefix[0], prefix[1]

    # Resolve iter_order - expand single value to tuple of two
    if iter_order is None or isinstance(iter_order, Real):
        iter_order = (iter_order, iter_order)
    elif isinstance(iter_order, Sequence) and len(iter_order) != 2:
        raise ValueError("iter_order must be a sequence of length 2")
    iter_order_scan, iter_order_mut = iter_order[0], iter_order[1]

    # 1. Insert tags at scanning positions
    marked = region_scan(
        pool,
        region=marker_name,
        region_length=marker_length,
        positions=positions,
        region_constraint=region,
        remove_tags=False,  # Keep outer tags for now
        prefix=prefix_scan,
        mode=mode_scan,
        num_states=num_states_scan,
        iter_order=iter_order_scan,
        _factory_name=f"{_factory_name}(region_scan)",
    )

    # 2. Mutagenize marker with content
    result = mutagenize(
        pool=marked,
        num_mutations=num_mutations,
        mutation_rate=mutation_rate,
        region="_mut",
        prefix=prefix_mut,
        mode=mode_mut,
        num_states=num_states_mut,
        iter_order=iter_order_mut,
        _factory_name=f"{_factory_name}(mutagenize)",
    )

    return result
