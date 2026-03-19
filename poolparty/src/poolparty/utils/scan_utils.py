"""Scan operation utilities."""

from itertools import product

from ..types import Optional, PositionsType, Sequence
from .seq_utils import validate_positions


def build_scan_cache(
    seq_length: Optional[int],
    item_length: int,
    positions: PositionsType,
    error_context: str = "scan",
) -> int:
    """Build cache and return number of states for a scan operation.

    Parameters
    ----------
    seq_length
        Length of the sequence to scan. If None, uses positions directly.
    item_length
        Length of the item being scanned (e.g., deletion length, region length).
    positions
        User-specified positions to scan, or None for all positions.
    error_context
        Context string for error messages (e.g., "deletion", "region tag insertion").

    Returns
    -------
    int
        Number of valid positions (states) for the scan operation.

    Raises
    ------
    ValueError
        If no valid positions exist for the scan.
    """
    if seq_length is None:
        if positions is not None:
            positions_list = validate_positions(
                positions,
                max_position=1000000,
                min_position=0,
            )
            return max(1, len(positions_list))
        return 1

    # Calculate maximum valid starting position
    max_start = seq_length - item_length
    if max_start < 0:
        max_start = 0

    num_all_positions = max_start + 1
    if positions is not None:
        indices = validate_positions(
            positions,
            max_position=num_all_positions - 1,
            min_position=0,
        )
        num_states = len(indices)
    else:
        num_states = num_all_positions

    if num_states == 0:
        raise ValueError(f"No valid positions for {error_context}")
    return num_states


def _normalize_region_lengths(
    region_length: int | Sequence[int],
    num_insertions: int,
) -> list[int]:
    """Normalize region_length to a list of per-region lengths."""
    if isinstance(region_length, int):
        return [region_length] * num_insertions
    lengths = list(region_length)
    if len(lengths) != num_insertions:
        raise ValueError(
            f"region_length sequence length ({len(lengths)}) "
            f"must equal num_insertions ({num_insertions})"
        )
    return lengths


def _is_valid_combo(
    combo: tuple[int, ...],
    region_lengths: list[int],
    min_spacing: int,
    max_spacing: Optional[int],
) -> bool:
    """Check if an assignment combo is valid (non-overlapping, spacing OK).

    ``combo[i]`` is the position for insert ``i`` with length ``region_lengths[i]``.
    Builds sorted intervals and checks gaps.
    """
    n = len(combo)
    if n == 1:
        return True

    # Build (position, length) pairs sorted by position
    intervals = sorted(zip(combo, region_lengths))

    for i in range(n - 1):
        gap = intervals[i + 1][0] - (intervals[i][0] + intervals[i][1])
        if gap < min_spacing:
            return False
        if max_spacing is not None and gap > max_spacing:
            return False
    return True


def enumerate_multiscan_combinations(
    valid_positions: list[int] | list[list[int]],
    num_insertions: int,
    region_length: int | Sequence[int],
    insertion_mode: str = "ordered",
    min_spacing: int = 0,
    max_spacing: Optional[int] = None,
    max_combinations: int = 10_000_000,
) -> list[tuple[int, ...]]:
    """Enumerate valid K-position assignment tuples for multiscan sequential mode.

    Each returned tuple is an **assignment**: ``combo[i]`` is the position for
    insert ``i``. In ordered mode, combos are spatially sorted (insert 0 is
    leftmost). In unordered mode, different orderings of the same position set
    produce distinct combos.

    Parameters
    ----------
    valid_positions
        Either a flat list of ints (shared positions, replicated for all inserts)
        or a list of lists (per-insert positions).
    num_insertions
        Number of simultaneous insertions (K).
    region_length
        Length of each inserted region. Single int for uniform, or a sequence
        of ints for per-region lengths.
    insertion_mode
        'ordered' (insert i maps to i-th leftmost position) or 'unordered'
        (all valid assignments of inserts to positions are enumerated).
    min_spacing
        Minimum gap between end of one region and start of next.
    max_spacing
        Maximum gap (None = unbounded).
    max_combinations
        Safety cap on Cartesian product size.

    Returns
    -------
    list[tuple[int, ...]]
        List of valid assignment tuples.
    """
    if num_insertions < 1:
        raise ValueError(f"num_insertions must be >= 1, got {num_insertions}")

    region_lengths = _normalize_region_lengths(region_length, num_insertions)

    is_per_insert = (
        len(valid_positions) > 0
        and isinstance(valid_positions[0], (list, tuple))
    )

    if is_per_insert:
        per_insert: list[list[int]] = [list(p) for p in valid_positions]
        if len(per_insert) != num_insertions:
            raise ValueError(
                f"per-insert positions length ({len(per_insert)}) "
                f"must equal num_insertions ({num_insertions})"
            )
    else:
        per_insert = [list(valid_positions)] * num_insertions

    # Safety cap
    raw_size = 1
    for p in per_insert:
        raw_size *= len(p)
        if raw_size > max_combinations:
            raise ValueError(
                f"Cartesian product size ({raw_size}) exceeds "
                f"max_combinations ({max_combinations})"
            )

    seen: set[frozenset[tuple[int, int]]] = set()
    result: list[tuple[int, ...]] = []

    for combo in product(*per_insert):
        # All positions must be distinct
        if len(set(combo)) < num_insertions:
            continue

        if insertion_mode == "ordered":
            # Ordered: insert 0 must be leftmost, insert 1 next, etc.
            if combo != tuple(sorted(combo)):
                continue
            # Deduplicate (shared positions can produce identical sorted combos)
            dedup_key = frozenset(enumerate(combo))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
        else:
            # Unordered: deduplicate by the set of (position, insert_idx) pairs.
            # Two combos that assign the same inserts to the same positions
            # (just listed in different iteration order) are duplicates.
            dedup_key = frozenset(enumerate(combo))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

        if _is_valid_combo(combo, region_lengths, min_spacing, max_spacing):
            result.append(combo)

    if not result:
        raise ValueError(
            f"No valid {num_insertions}-position combinations with "
            f"min_spacing={min_spacing}, max_spacing={max_spacing}"
        )

    return sorted(result)
