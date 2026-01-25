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
    insertion_mode: Literal['ordered', 'unordered'] = 'ordered',
    prefix: Optional[str] = None,
    mode: str = 'random',
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
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
    insertion_mode : Literal['ordered', 'unordered'], default='ordered'
        How to assign insertion pools to positions:
        - 'ordered': pools[i] goes to the i-th selected position (left to right)
        - 'unordered': randomly assign pools to positions
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : str, default='random'
        Position selection mode: 'random'.
    num_states : Optional[Integral], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool yielding sequences with multiple insertions made simultaneously.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..marker_ops import marker_multiscan, replace_marker_content

    # Validate mode
    if mode != 'random':
        raise ValueError(
            f"insertion_multiscan supports only mode='random', got '{mode}'"
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
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        iter_order=iter_order,
    )

    # 2. Build insertion content for each pool
    result = marked
    for marker_name, ins_pool in zip(markers, pools_list):
        content = ins_pool

        # Replace marker with content
        result = replace_marker_content(
            result,
            content,
            marker_name,
            iter_order=iter_order,
        )

    return result
