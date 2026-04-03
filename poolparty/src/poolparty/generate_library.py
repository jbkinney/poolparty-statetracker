"""Library generation functions for poolparty."""

import logging

logger = logging.getLogger(__name__)

import warnings

import numpy as np
import pandas as pd

from .types import Integral, Optional, Pool_type, Seq, Union, beartype, is_null_seq


@beartype
def generate_library(
    pool: Pool_type,
    num_cycles: Integral = 1,
    num_seqs: Optional[Integral] = None,
    seed: Optional[Integral] = None,
    init_state: Optional[int] = None,
    seqs_only: bool = False,
    _include_inline_styles: bool = False,
    discard_null_seqs: bool = False,
    max_iterations: Optional[int] = None,
    min_acceptance_rate: Optional[float] = None,
    attempts_per_rate_assessment: int = 100,
) -> Union[pd.DataFrame, list[str | None]]:
    """Generate sequences from a pool.

    Args:
        pool: The pool to generate sequences from.
        num_cycles: Number of complete iterations through all states.
        num_seqs: Number of sequences to generate.
        seed: Random seed for reproducibility.
        init_state: Initial state to start generation from.
        seqs_only: If True, return list of sequences instead of DataFrame.
        discard_null_seqs: If True, discard sequences that fail filters (null
            sequences). With num_seqs, keeps sampling until N valid sequences are
            collected. With num_cycles, enumerates all states and returns only the
            valid ones (output may have fewer than num_cycles * num_states rows).
        max_iterations: Maximum iterations before stopping. Default: state space
            size for sequential mode, or num_seqs * 100 for random mode.
        min_acceptance_rate: Minimum fraction of sequences that must pass filters.
            If actual rate falls below this, generation stops with a warning.
        attempts_per_rate_assessment: Iterations between acceptance rate checks.

    Returns:
        DataFrame with columns: name, seq, plus any requested design card columns.
        Or list of sequences if seqs_only=True. Entries are None for null
        rows when discard_null_seqs=False.

    Note:
        Design card columns are opt-in via the `cards` parameter on individual
        operations. Default output contains only 'name' and 'seq' columns.
    """
    # Initialize state tracking on pool if not present
    if not hasattr(pool, "_current_state"):
        pool._current_state = 0
    if not hasattr(pool, "_master_seed"):
        pool._master_seed = None

    # Coerce Integral types to native int
    num_cycles = int(num_cycles)
    if num_seqs is not None:
        num_seqs = int(num_seqs)
    if seed is not None:
        seed = int(seed)

    # Validate arguments
    if num_cycles <= 0:
        raise ValueError(f"num_cycles must be positive, got {num_cycles}")
    if attempts_per_rate_assessment <= 0:
        raise ValueError(
            f"attempts_per_rate_assessment must be positive, got {attempts_per_rate_assessment}"
        )
    if num_seqs is not None and num_cycles != 1:
        warnings.warn(
            "Both num_seqs and num_cycles provided; num_seqs takes precedence.",
            stacklevel=2,
        )
    if num_seqs is None:
        num_seqs = num_cycles * pool.state.num_values
    elif num_seqs <= 0:
        raise ValueError(f"num_seqs must be positive, got {num_seqs}")
    if init_state is not None:
        pool._current_state = init_state
    if seed is not None:
        pool._master_seed = seed
    if pool._master_seed is None:
        pool._master_seed = 0

    # Set default max_iterations
    if max_iterations is None:
        if pool.state.num_values > 1:
            max_iterations = pool.state.num_values
        else:
            max_iterations = num_seqs * 100

    logger.info(
        "Starting library generation: pool=%s num_seqs=%s seed=%s", pool.name, num_seqs, seed
    )

    # Topologically sort operations reachable from pool
    sorted_ops = _topo_sort_operations(pool)
    _seed_random_operations(sorted_ops, pool._master_seed)

    # Generate rows
    rows = []
    state = pool._current_state
    iterations = 0
    valid_count = 0
    max_global_state = state + num_seqs - 1  # For zero-padding in names

    while len(rows) < num_seqs:
        global_state = state
        row = _compute_one(
            pool,
            sorted_ops,
            global_state,
            max_global_state,
            _include_inline_styles,
        )

        # Check if this row has a null sequence
        seq_value = row.get("seq")
        is_null = seq_value is None

        if discard_null_seqs:
            if not is_null:
                rows.append(row)
                valid_count += 1
        else:
            # Include all rows (null sequences show as None in output)
            if is_null:
                row["seq"] = None
                row["name"] = None
            rows.append(row)
            if not is_null:
                valid_count += 1

        state += 1
        iterations += 1

        # Check acceptance rate periodically
        if (
            discard_null_seqs
            and min_acceptance_rate is not None
            and iterations > 0
            and iterations % attempts_per_rate_assessment == 0
        ):
            actual_rate = valid_count / iterations
            if actual_rate < min_acceptance_rate:
                warnings.warn(
                    f"Acceptance rate ({actual_rate:.1%}) below minimum "
                    f"({min_acceptance_rate:.1%}) after {iterations} iterations. "
                    f"Generated {valid_count} valid sequences. Stopping early.",
                    stacklevel=2,
                )
                break

        # Check max iterations (only relevant when filtering)
        if discard_null_seqs and iterations >= max_iterations:
            if len(rows) < num_seqs:
                warnings.warn(
                    f"Reached max_iterations ({max_iterations}) with only "
                    f"{len(rows)} valid sequences (requested {num_seqs}). "
                    f"Acceptance rate: {valid_count / iterations:.1%}",
                    stacklevel=2,
                )
            break

        # Check state space exhaustion (only for filtering in sequential mode)
        # When not filtering, allow cycling through states multiple times
        if discard_null_seqs and pool.state.num_values > 1:
            if state >= pool._current_state + pool.state.num_values:
                if len(rows) < num_seqs:
                    warnings.warn(
                        f"State space exhausted: only {len(rows)} valid sequences "
                        f"exist (requested {num_seqs}). "
                        f"Acceptance rate: {valid_count / iterations:.1%}",
                        stacklevel=2,
                    )
                break

    pool._current_state = state

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Handle empty DataFrame case
    if len(df) == 0:
        if seqs_only:
            return []
        return pd.DataFrame(columns=["name", "seq"])

    # Ensure name and seq are first columns, in that order
    cols = ["name", "seq"] + [c for c in df.columns if c not in ("name", "seq", "_inline_styles")]
    if "_inline_styles" in df.columns:
        cols.append("_inline_styles")
    df = df[cols]

    logger.info("Completed library generation: %d sequences", len(df))
    if seqs_only:
        return list(df["seq"])
    return df


def _topo_sort_operations(pool: Pool_type) -> list:
    """Topologically sort operations reachable from pool."""
    from .operation import Operation

    visited: set[int] = set()
    result: list[Operation] = []

    def visit(p: Pool_type) -> None:
        op = p.operation
        if op.id in visited:
            return
        for parent in op.parent_pools:
            visit(parent)
        visited.add(op.id)
        result.append(op)

    visit(pool)
    return result


def _seed_random_operations(sorted_ops: list, master_seed: int) -> None:
    """Clear RNG on all operations (RNG is created per-call in _compute_one)."""
    for op in sorted_ops:
        op.rng = None


def _compute_one(
    pool: Pool_type,
    sorted_ops: list,
    global_state: int,
    max_global_state: int,
    include_inline_styles: bool = False,
) -> dict:
    """Compute one row of output for the given global state.

    Returns a dict with:
    - 'name': the sequence name
    - 'seq': the final sequence string
    - Any requested design card columns from operations with cards specified
    """
    seq_cache: dict[int, Seq] = {}
    row: dict = {}

    # Sets the value of the pool state and, in doing so, propagates values
    # to all parent pool and operation states in the DAG.
    pool.state.value = global_state % pool.state.num_values

    # Collect all name contributions from operations in topological order
    all_contributions: list[str] = []

    # Iterates over the operations in topological order (sources to final).
    for op in sorted_ops:
        # Get parent Seq objects (already cached because of topological sort)
        parents = [seq_cache[p.operation.id] for p in op.parent_pools]

        # Determine RNG for this operation
        if op.mode == "random":
            if op.action_uniquely_determined_by_state:
                # Explicit num_states > 1: use state value
                state_val = op.state.value if op.state.value is not None else 0
            else:
                # Stateless random (num_states=None or 1): use global_state (row number)
                state_val = global_state
            seed_seq = np.random.SeedSequence([pool._master_seed, op.id, state_val])
            op_rng = np.random.default_rng(seed_seq)
        else:
            op_rng = op.rng

        # Compute output Seq and raw design card
        output_seq, raw_card = op.compute(parents, op_rng)

        # Store seq in cache for downstream operations
        seq_cache[op.id] = output_seq

        # Collect name contributions from this operation
        all_contributions.extend(op.compute_name_contributions(global_state, max_global_state))

        # Process design cards if this operation has cards requested
        if op.has_cards:
            # Get seq and state values for universal keys
            seq_value = output_seq.string if not is_null_seq(output_seq) else None
            state_value = op.state.value

            # Filter card based on _cards spec (handles universal keys too)
            filtered_card = op._filter_design_card(raw_card, seq_value, state_value)

            # Add to row with appropriate column naming
            for key, value in filtered_card.items():
                # If operation uses custom column names (dict), use key directly
                # Otherwise prefix with op.name
                if op.uses_custom_column_names:
                    col_name = key
                else:
                    col_name = f"{op.name}.{key}"

                # Return None if state is inactive (on an inactive branch)
                if not op.state.is_active:
                    row[col_name] = None
                else:
                    row[col_name] = value

    # Get the final sequence
    final_seq = seq_cache[pool.operation.id]
    if is_null_seq(final_seq):
        row["seq"] = None
    else:
        row["seq"] = final_seq.string

    # Compute final name from contributions (already in topological order)
    final_name = ".".join(all_contributions) if all_contributions else None
    row["name"] = final_name

    # Get inline styles from final Seq object (only if requested)
    if include_inline_styles:
        row["_inline_styles"] = final_seq.style

    return row
