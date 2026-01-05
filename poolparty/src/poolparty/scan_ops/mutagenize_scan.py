"""Mutagenize scan operation - apply mutagenesis within a window at scanning positions."""
from numbers import Integral, Real

from ..types import Union, ModeType, Optional, PositionsType, beartype
from ..seq_utils import validate_positions
from ..party import get_active_party
from ..pool import Pool


@beartype
def mutagenize_scan(
    bg_pool: Union[Pool, str],
    mutagenize_length: Integral,
    num_mutations: Optional[Integral] = None,
    mutation_rate: Optional[Real] = None,
    positions: PositionsType = None,
    spacer_str: str = '',
    mark_changes: Optional[bool] = None,
    seq_name_prefix: Optional[str] = None,
    mode: ModeType = 'random',
    num_hybrid_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Apply mutagenesis within a window at specified scanning positions.

    Parameters
    ----------
    bg_pool : Pool or str
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
    spacer_str : str, default=''
        String to insert as a spacer around the mutagenized region.
    mark_changes : Optional[bool], default=None
        If True, apply swapcase() to the mutagenized region. If None, uses party default.
    mode : ModeType, default='random'
        Selection mode for scanning positions: 'random', 'sequential', or 'hybrid'.
        Note: The underlying MutagenizeOp always uses 'random' mode.
    num_hybrid_states : Optional[Integral], default=None
        Number of pool states when using 'hybrid' mode (ignored by other modes).
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operation.

    Returns
    -------
    Pool
        A Pool yielding sequences where a region of the specified length is mutagenized
        at each allowed position.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.join import join
    from ..fixed_ops.swapcase import swapcase
    from ..base_ops.mutagenize import mutagenize
    from ..marker_ops import marker_scan, apply_at_marker

    # Convert string inputs to pools if needed
    bg_pool = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool

    # Validate bg_pool has defined seq_length
    bg_length = bg_pool.seq_length
    if bg_length is None:
        raise ValueError("bg_pool must have a defined seq_length")

    # Validate mutagenize_length
    if mutagenize_length <= 0:
        raise ValueError(f"mutagenize_length must be > 0, got {mutagenize_length}")
    if mutagenize_length >= bg_length:
        raise ValueError(
            f"mutagenize_length ({mutagenize_length}) must be < bg_pool.seq_length ({bg_length})"
        )

    # Validate num_mutations/mutation_rate
    if num_mutations is None and mutation_rate is None:
        raise ValueError("Either num_mutations or mutation_rate must be provided")
    if num_mutations is not None and mutation_rate is not None:
        raise ValueError("Only one of num_mutations or mutation_rate can be provided, not both")

    # Resolve mark_changes from party defaults if not explicitly set
    party = get_active_party()
    if mark_changes is None:
        mark_changes = party.get_default('mark_changes', False) if party else False

    # For mutagenize: marker_length=mutagenize_length, max_position=bg_length - mutagenize_length
    marker_name = '_mut'
    marker_length = int(mutagenize_length)
    max_position = bg_length - mutagenize_length

    # Validate positions
    validated_positions = validate_positions(positions, max_position, min_position=0)

    # 1. Insert marker at scanning positions
    marked = marker_scan(
        bg_pool,
        marker=marker_name,
        marker_length=marker_length,
        positions=validated_positions,
        seq_name_prefix=seq_name_prefix,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        op_name=op_name,
        op_iter_order=op_iter_order,
    )

    # 2. Apply mutagenize transform at marker
    # Note: MutagenizeOp always uses 'random' mode for the actual mutagenesis.
    # The 'mode' parameter controls position selection via marker_scan above.
    def mutagenize_transform(content_pool):
        mutagenized = mutagenize(
            content_pool,
            num_mutations=num_mutations,
            mutation_rate=mutation_rate,
            mark_changes=mark_changes,
            mode='random',  # Always random for mutagenesis
        )
        # Wrap with spacers if needed
        if spacer_str:
            mutagenized = join([from_seq(spacer_str), mutagenized, from_seq(spacer_str)])
        return mutagenized

    result = apply_at_marker(
        marked,
        marker_name,
        mutagenize_transform,
        name=name,
        iter_order=iter_order,
    )
    return result
