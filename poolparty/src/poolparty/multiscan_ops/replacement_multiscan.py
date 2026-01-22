"""Replacement multiscan operation - replace segments at multiple positions simultaneously."""
from numbers import Integral, Real

from ..types import Union, Optional, Sequence, Literal, PositionsType, beartype
from ..seq_utils import validate_positions
from ..pool import Pool


@beartype
def replacement_multiscan(
    bg_pool: Union[Pool, str],
    num_replacements: Integral,
    replacement_pools: Union[Pool, Sequence[Pool]],
    positions: PositionsType = None,
    spacer_str: str = '',
    mark_changes: Optional[bool] = None,
    insertion_mode: Literal['ordered', 'unordered'] = 'ordered',
    seq_name_prefix: Optional[str] = None,
    mode: str = 'random',
    num_states: Optional[Integral] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
) -> Pool:
    """
    Replace segments at multiple positions simultaneously.

    Uses marker_multiscan() to insert markers at multiple positions, then
    replaces each marker's content with sequences from replacement pools.

    Parameters
    ----------
    bg_pool : Pool or str
        Source pool or sequence string to replace segments in.
    num_replacements : Integral
        Number of simultaneous replacements to make.
    replacement_pools : Pool or Sequence[Pool]
        Pool(s) providing replacement content. If a single Pool is provided,
        it will be deepcopied num_replacements-1 times. If a Sequence of Pools
        is provided, its length must equal num_replacements.
    positions : PositionsType, default=None
        Valid positions for replacement starts (0-based). If None, all valid
        positions are used.
    spacer_str : str, default=''
        String to insert as a spacer around replacement sites.
    mark_changes : Optional[bool], default=None
        If True, apply swapcase() to replacement sequences. If None, uses
        party default.
    insertion_mode : Literal['ordered', 'unordered'], default='ordered'
        How to assign replacement pools to positions:
        - 'ordered': pools[i] goes to the i-th selected position (left to right)
        - 'unordered': randomly assign pools to positions
    mode : str, default='random'
        Position selection mode: 'random'.
    num_states : Optional[Integral], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
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
        A Pool yielding sequences with multiple segments replaced simultaneously.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.swapcase import swapcase
    from ..marker_ops import marker_multiscan, replace_marker_content
    from ..party import get_active_party

    # Validate mode
    if mode != 'random':
        raise ValueError(
            f"replacement_multiscan supports only mode='random', got '{mode}'"
        )

    # Validate num_replacements
    if num_replacements < 1:
        raise ValueError(f"num_replacements must be >= 1, got {num_replacements}")

    # Convert string inputs to pools if needed
    bg_pool = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool

    # Validate bg_pool has defined seq_length
    bg_length = bg_pool.seq_length
    if bg_length is None:
        raise ValueError("bg_pool must have a defined seq_length")

    # Handle replacement_pools: single Pool vs Sequence of Pools
    if isinstance(replacement_pools, Pool):
        # Single pool: create deepcopies
        pools_list = [replacement_pools]
        for i in range(num_replacements - 1):
            pools_list.append(replacement_pools.deepcopy(name=f'_rep_pool_{i+1}'))
    else:
        # Sequence of pools: validate length
        pools_list = list(replacement_pools)
        if len(pools_list) != num_replacements:
            raise ValueError(
                f"replacement_pools length ({len(pools_list)}) must equal "
                f"num_replacements ({num_replacements})"
            )

    # Validate all replacement pools have defined seq_length
    replacement_lengths = []
    for i, pool in enumerate(pools_list):
        if pool.seq_length is None:
            raise ValueError(
                f"replacement_pools[{i}] must have a defined seq_length"
            )
        replacement_lengths.append(pool.seq_length)

    # All replacement pools must have the same seq_length
    replacement_length = replacement_lengths[0]
    if not all(length == replacement_length for length in replacement_lengths):
        raise ValueError(
            f"All replacement pools must have the same seq_length, got {replacement_lengths}"
        )

    # Check if there's room for num_replacements non-overlapping regions
    min_required_length = num_replacements * replacement_length
    if min_required_length > bg_length:
        raise ValueError(
            f"Cannot fit {num_replacements} non-overlapping replacements of length "
            f"{replacement_length} in sequence of length {bg_length}"
        )

    # Resolve mark_changes from party defaults if not explicitly set
    party = get_active_party()
    if mark_changes is None:
        mark_changes = party.get_default('mark_changes', False) if party else False

    # Generate auto-indexed marker names
    markers = [f'_rep_{i}' for i in range(num_replacements)]
    marker_length = int(replacement_length)
    max_position = bg_length - replacement_length

    # Validate positions
    validated_positions = validate_positions(positions, max_position, min_position=0)

    # 1. Insert markers at multiple positions using marker_multiscan
    marked = marker_multiscan(
        bg_pool,
        markers=markers,
        num_insertions=int(num_replacements),
        positions=validated_positions,
        marker_length=marker_length,
        insertion_mode=insertion_mode,
        seq_name_prefix=seq_name_prefix,
        mode=mode,
        num_states=num_states,
        op_name=op_name,
        op_iter_order=op_iter_order,
    )

    # 2. Build replacement content for each pool
    # spacer_str is handled by replace_marker_content
    result = marked
    for marker_name, rep_pool in zip(markers, pools_list):
        # Apply swapcase if mark_changes
        content = swapcase(rep_pool) if mark_changes else rep_pool

        # Replace marker with content
        result = replace_marker_content(
            result,
            content,
            marker_name,
            spacer_str=spacer_str,
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
