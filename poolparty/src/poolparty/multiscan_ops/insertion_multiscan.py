"""Insertion multiscan operation - insert sequences at multiple positions simultaneously."""
from numbers import Integral, Real

from ..types import Union, Optional, Sequence, Literal, PositionsType, beartype
from ..seq_utils import validate_positions
from ..pool import Pool


@beartype
def insertion_multiscan(
    bg_pool: Union[Pool, str],
    num_insertions: Integral,
    insertion_pools: Union[Pool, Sequence[Pool]],
    positions: PositionsType = None,
    spacer_str: str = '',
    mark_changes: Optional[bool] = None,
    insertion_mode: Literal['ordered', 'unordered'] = 'ordered',
    seq_name_prefix: Optional[str] = None,
    mode: str = 'random',
    num_hybrid_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Insert sequences at multiple positions simultaneously.

    Uses marker_multiscan() to insert zero-length markers at multiple positions,
    then replaces each marker's content with sequences from insertion pools.

    Parameters
    ----------
    bg_pool : Pool or str
        Source pool or sequence string to insert into.
    num_insertions : Integral
        Number of simultaneous insertions to make.
    insertion_pools : Pool or Sequence[Pool]
        Pool(s) providing insertion content. If a single Pool is provided,
        it will be deepcopied num_insertions-1 times. If a Sequence of Pools
        is provided, its length must equal num_insertions.
    positions : PositionsType, default=None
        Valid positions for insertions (0-based). If None, all valid
        positions are used (0 to bg_length inclusive).
    spacer_str : str, default=''
        String to insert as a spacer around insertion sites.
    mark_changes : Optional[bool], default=None
        If True, apply swapcase() to insertion sequences. If None, uses
        party default.
    insertion_mode : Literal['ordered', 'unordered'], default='ordered'
        How to assign insertion pools to positions:
        - 'ordered': pools[i] goes to the i-th selected position (left to right)
        - 'unordered': randomly assign pools to positions
    mode : str, default='random'
        Position selection mode: 'random' or 'hybrid'.
    num_hybrid_states : Optional[Integral], default=None
        Number of pool states when using 'hybrid' mode.
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operations.
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operations.

    Returns
    -------
    Pool
        A Pool yielding sequences with multiple insertions made simultaneously.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.join import join
    from ..fixed_ops.swapcase import swapcase
    from ..marker_ops import marker_multiscan, replace_marker_content
    from ..party import get_active_party

    # Validate mode
    if mode not in ('random', 'hybrid'):
        raise ValueError(
            f"insertion_multiscan supports only mode='random' or 'hybrid', got '{mode}'"
        )

    # Validate num_insertions
    if num_insertions < 1:
        raise ValueError(f"num_insertions must be >= 1, got {num_insertions}")

    # Convert string inputs to pools if needed
    bg_pool = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool

    # Validate bg_pool has defined seq_length
    bg_length = bg_pool.seq_length
    if bg_length is None:
        raise ValueError("bg_pool must have a defined seq_length")

    # Handle insertion_pools: single Pool vs Sequence of Pools
    if isinstance(insertion_pools, Pool):
        # Single pool: create deepcopies
        pools_list = [insertion_pools]
        for i in range(num_insertions - 1):
            pools_list.append(insertion_pools.deepcopy(name=f'_ins_pool_{i+1}'))
    else:
        # Sequence of pools: validate length
        pools_list = list(insertion_pools)
        if len(pools_list) != num_insertions:
            raise ValueError(
                f"insertion_pools length ({len(pools_list)}) must equal "
                f"num_insertions ({num_insertions})"
            )

    # Validate all insertion pools have defined seq_length
    for i, pool in enumerate(pools_list):
        if pool.seq_length is None:
            raise ValueError(
                f"insertion_pools[{i}] must have a defined seq_length"
            )

    # Resolve mark_changes from party defaults if not explicitly set
    party = get_active_party()
    if mark_changes is None:
        mark_changes = party.get_default('mark_changes', False) if party else False

    # Generate auto-indexed marker names
    markers = [f'_ins_{i}' for i in range(num_insertions)]
    # Zero-length markers for insertions
    marker_length = 0
    # Can insert at any position from 0 to bg_length (inclusive)
    max_position = bg_length

    # Validate positions
    validated_positions = validate_positions(positions, max_position, min_position=0)

    # 1. Insert zero-length markers at multiple positions using marker_multiscan
    marked = marker_multiscan(
        bg_pool,
        markers=markers,
        num_insertions=int(num_insertions),
        positions=validated_positions,
        marker_length=marker_length,
        insertion_mode=insertion_mode,
        seq_name_prefix=seq_name_prefix,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        op_name=op_name,
        op_iter_order=op_iter_order,
    )

    # 2. Build insertion content for each pool
    result = marked
    for marker_name, ins_pool in zip(markers, pools_list):
        # Apply swapcase if mark_changes
        content = swapcase(ins_pool) if mark_changes else ins_pool

        # Wrap with spacers if needed
        if spacer_str:
            content = join([from_seq(spacer_str), content, from_seq(spacer_str)])

        # Replace marker with content
        result = replace_marker_content(
            result,
            content,
            marker_name,
            name=None,  # Only set name on final result
            op_name=op_name,
            iter_order=None,  # Only set iter_order on final result
            op_iter_order=op_iter_order,
        )

    # Set name and iter_order on final result if provided
    if name is not None:
        result._name = name
    if iter_order is not None:
        result._iter_order = iter_order

    return result
