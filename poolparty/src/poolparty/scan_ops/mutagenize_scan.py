"""Mutagenize scan operation - apply mutagenesis within a window at scanning positions."""
from numbers import Integral, Real

from ..types import Union, ModeType, Optional, PositionsType, RegionType, Tuple, Sequence, beartype
from ..party import get_active_party
from ..pool import Pool


@beartype
def mutagenize_scan(
    pool: Union[Pool, str],
    mutagenize_length: Integral,
    num_mutations: Optional[Integral] = None,
    mutation_rate: Optional[Real] = None,
    positions: PositionsType = None,
    region: RegionType = None,
    remove_marker: Optional[bool] = None,
    spacer_str: str = '',
    mark_changes: Optional[bool] = None,
    seq_name_prefix: Optional[Union[str, Sequence[str]]] = None,
    mode: Union[ModeType, Tuple[ModeType, ModeType]] = 'random',
    num_states: Optional[Union[Integral, Sequence[Integral]]] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Union[Real, Sequence[Real]]] = None,
    _factory_name: Optional[str] = 'mutagenize_scan',
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
    remove_marker : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from output.
        If None, uses Party default.
    spacer_str : str, default=''
        String to insert as a spacer around the mutagenized region.
    mark_changes : Optional[bool], default=None
        If True, apply swapcase() to the mutated bases. If None, uses party default.
    seq_name_prefix : Optional[Union[str, Sequence[str]]], default=None
        Prefix for sequence names. 
        If sequence, first element is for scanning positions, second element is for mutagenization.
    mode : Union[ModeType, Sequence[ModeType]], default='random'
        Selection mode for scanning positions: 'random' or 'sequential'.
        If sequence, first element is for scanning positions, second element is for mutagenization.
    num_states : Optional[Union[Integral, Sequence[Integral]]], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
        If sequence, first element is for scanning positions, second element is for mutagenization.
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Union[Real, Sequence[Real]]], default=None
        Iteration order priority for the underlying Operation.
        If sequence, first element is for scanning positions, second element is for mutagenization.
    _factory_name: Optional[str], default=None
        Sets default name of the resulting operation

    Returns
    -------
    Pool
        A Pool yielding sequences where a region of the specified length is mutagenized
        at each allowed position.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..base_ops.mutagenize import mutagenize
    from ..marker_ops import marker_scan

    # Convert string inputs to pools if needed
    pool = from_seq(pool, _factory_name=f'{_factory_name}(from_seq)') if isinstance(pool, str) else pool

    # Validate num_mutations/mutation_rate
    if num_mutations is None and mutation_rate is None:
        raise ValueError("Either num_mutations or mutation_rate must be provided")
    if num_mutations is not None and mutation_rate is not None:
        raise ValueError("Only one of num_mutations or mutation_rate can be provided, not both")

    # Resolve mark_changes from party defaults if not explicitly set
    party = get_active_party()
    if mark_changes is None:
        mark_changes = party.get_default('mark_changes', False) if party else False

    # Resolve remove_marker from party defaults if not explicitly set
    if remove_marker is None:
        remove_marker = party.get_default('remove_marker', True) if party else True

    # Determine marker configuration based on replace mode
    # replace=False: marker_length=0 (insert without removing background)
    # replace=True: marker_length=ins_length (replace background content)
    # Use different marker names to avoid conflicts when both are used in same Party
    marker_name = '_mut'
    marker_length = mutagenize_length

    # Resolve mode - expand single value to tuple of two
    # Note: str is a Sequence, so check for str first
    if mode is None or isinstance(mode, str):
        mode = (mode, mode)
    elif isinstance(mode, Sequence) and len(mode) != 2:
        raise ValueError("mode must be a sequence of length 2")
    mode_scan, mode_mut = mode[0], mode[1]

    # Resolve num_hybrid_states - expand single value to tuple of two
    if num_hybrid_states is None or isinstance(num_hybrid_states, Integral):
        num_hybrid_states = (num_hybrid_states, num_hybrid_states)
    elif isinstance(num_hybrid_states, Sequence) and len(num_hybrid_states) != 2:
        raise ValueError("num_hybrid_states must be a sequence of length 2")
    num_hybrid_states_scan, num_hybrid_states_mut = num_hybrid_states[0], num_hybrid_states[1]

    # Resolve seq_name_prefix - expand single value to tuple of two
    # Note: str is a Sequence, so check for str first
    if seq_name_prefix is None or isinstance(seq_name_prefix, str):
        seq_name_prefix = (seq_name_prefix, seq_name_prefix)
    elif isinstance(seq_name_prefix, Sequence) and len(seq_name_prefix) != 2:
        raise ValueError("seq_name_prefix must be a sequence of length 2")
    seq_name_prefix_scan, seq_name_prefix_mut = seq_name_prefix[0], seq_name_prefix[1]

    # Resolve op_iter_order - expand single value to tuple of two
    if op_iter_order is None or isinstance(op_iter_order, Real):
        op_iter_order = (op_iter_order, op_iter_order)
    elif isinstance(op_iter_order, Sequence) and len(op_iter_order) != 2:
        raise ValueError("op_iter_order must be a sequence of length 2")
    op_iter_order_scan, op_iter_order_mut = op_iter_order[0], op_iter_order[1]


    # 1. Insert marker at scanning positions
    marked = marker_scan(
        pool,
        marker=marker_name,
        marker_length=marker_length,
        positions=positions,
        region=region,
        remove_marker=False,  # Keep outer region marker for now
        seq_name_prefix=seq_name_prefix_scan,
        mode=mode_scan,
        num_hybrid_states=num_hybrid_states_scan,
        op_name=op_name,
        op_iter_order=op_iter_order_scan,
        _factory_name=f'{_factory_name}(marker_scan)',
    )

    # 2. Mutagenize marker with content 
    result = mutagenize(
        pool=marked,
        num_mutations=num_mutations,
        mutation_rate=mutation_rate,
        region='_mut',
        remove_marker=True,
        spacer_str=spacer_str,
        mark_changes=mark_changes,
        swapcase=False,
        seq_name_prefix=seq_name_prefix_mut,
        mode=mode_mut,
        num_states=num_states_mut,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order_mut,
        _factory_name=f'{_factory_name}(mutagenize)',
    )
    
    return result
