"""Library generation functions for poolparty."""

import logging

logger = logging.getLogger(__name__)

import warnings

import numpy as np
import pandas as pd

import statetracker as st

from .types import Literal, Optional, Pool_type, Seq, Sequence, Union, beartype, is_null_seq
from .utils.df_utils import counter_col_name, finalize_generate_df, organize_columns
from .utils.utils import clean_df_int_columns


@beartype
def generate_library(
    pool: Pool_type,
    num_cycles: int = 1,
    num_seqs: Optional[int] = None,
    seed: Optional[int] = None,
    init_state: Optional[int] = None,
    seqs_only: bool = False,
    report_design_cards: bool = False,
    aux_pools: Sequence[Pool_type] = (),
    pools_to_report: Union[str, Sequence[Pool_type]] = "all",
    organize_columns_by: Literal["pool", "type"] = "type",
    _include_inline_styles: bool = False,
    discard_null_seqs: bool = False,
    max_iterations: Optional[int] = None,
    min_acceptance_rate: Optional[float] = None,
    attempts_per_rate_assessment: int = 100,
) -> Union[pd.DataFrame, list[str]]:
    """Generate sequences from a pool.

    Args:
        pool: The pool to generate sequences from.
        num_cycles: Number of complete iterations through all states.
        num_seqs: Number of sequences to generate.
        seed: Random seed for reproducibility.
        init_state: Initial state to start generation from.
        seqs_only: If True, return list of sequences instead of DataFrame.
        report_design_cards: If True, include detailed design card info in output.
            Column visibility controlled by party config (use pp.load_config()).
            When False (default), returns minimal DataFrame with "name" and "seq".
        aux_pools: Additional pools to include in output.
        pools_to_report: Which pools to report ('all', 'self', or list of pools).
        organize_columns_by: Column organization ('pool' or 'type').
        discard_null_seqs: If True, keep iterating until num_seqs valid (non-null)
            sequences are generated. Requires num_seqs to be specified.
        max_iterations: Maximum iterations before stopping. Default: state space
            size for sequential mode, or num_seqs * 100 for random mode.
        min_acceptance_rate: Minimum fraction of sequences that must pass filters.
            If actual rate falls below this, generation stops with a warning.
        attempts_per_rate_assessment: Iterations between acceptance rate checks.

    Returns:
        DataFrame with generated sequences, or list of sequences if seqs_only=True.
    """
    # Initialize state tracking on pool if not present
    if not hasattr(pool, "_current_state"):
        pool._current_state = 0
    if not hasattr(pool, "_master_seed"):
        pool._master_seed = None

    # Validate arguments
    if discard_null_seqs and num_seqs is None:
        raise ValueError(
            "num_seqs must be specified when discard_null_seqs=True. "
            "Cannot use num_cycles with filtering."
        )
    if num_seqs is None:
        num_seqs = num_cycles * pool.state.num_values
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

    # Get config from pool's party (not active party, which may be different)
    config = pool._party._config if hasattr(pool, "_party") and pool._party else None
    suppress_cards = config.suppress_cards if config else False

    logger.info(
        "Starting library generation: pool=%s num_seqs=%s seed=%s", pool.name, num_seqs, seed
    )

    # Build outputs dict
    outputs: dict[str, Pool_type] = {f"{pool.name}.seq": pool}
    if report_design_cards and not suppress_cards:
        for aux_pool in aux_pools:
            outputs[f"{aux_pool.name}.seq"] = aux_pool

    sorted_ops = _topo_sort_operations(outputs)
    _seed_random_operations(sorted_ops, pool._master_seed)

    # Determine which pools to report (only used when report_design_cards=True and cards not suppressed)
    if report_design_cards and not suppress_cards:
        if pools_to_report == "all":
            pools_filter = _collect_all_pools(outputs)
        elif pools_to_report == "self":
            pools_filter = {pool}
        else:
            pools_filter = set(pools_to_report)

        ops_to_report = {p.operation.id for p in pools_filter}

        # Add filtered pools to outputs if not already present
        for p in pools_filter:
            key = f"{p.name}.seq"
            if key not in outputs:
                outputs[key] = p

        # Get column visibility from config (defaults to True)
        report_pool_states = config.show_pool_states if config else True
        report_op_states = config.show_op_states if config else True

        states = _collect_counters(pools_filter, report_pool_states, report_op_states)
    else:
        pools_filter = {pool}
        ops_to_report = set()
        states = []

    # Generate rows
    rows = []
    state = pool._current_state
    iterations = 0
    valid_count = 0
    seq_col = f"{pool.name}.seq"
    max_global_state = state + num_seqs - 1  # For zero-padding in names

    while len(rows) < num_seqs:
        global_state = state
        row = _compute_one(
            pool,
            sorted_ops,
            outputs,
            global_state,
            max_global_state,
            states,
            report_design_cards and not suppress_cards,
            ops_to_report,
            pools_filter,
            _include_inline_styles,
        )

        # Check if this row has a null sequence
        seq_value = row.get(seq_col)
        is_null = seq_value is None or seq_value == ""

        if discard_null_seqs:
            if not is_null:
                rows.append(row)
                valid_count += 1
        else:
            # Include all rows (null sequences show as None in output)
            if is_null:
                row[seq_col] = None
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

    # Build and format DataFrame
    df = pd.DataFrame(rows)

    # Handle empty DataFrame case
    if len(df) == 0:
        if seqs_only:
            return []
        return pd.DataFrame(columns=["name", "seq"])

    if not report_design_cards:
        # Minimal output: just "name" and "seq" columns
        df = df[["name", f"{pool.name}.seq"]].rename(columns={f"{pool.name}.seq": "seq"})
        if seqs_only:
            return list(df["seq"])
        return df

    # Full design card output
    df = clean_df_int_columns(df)
    df = organize_columns(df, pools_filter, organize_columns_by)

    # Get column visibility from config
    report_seq = config.show_seq if config else True
    report_pool_seqs = config.show_pool_seqs if config else True
    show_name = config.show_name if config else True

    df = finalize_generate_df(df, pool.name, report_seq, report_pool_seqs, pools_filter, show_name)

    # If cards are suppressed, remove the pool-specific seq column (keep only 'seq')
    if suppress_cards and f"{pool.name}.seq" in df.columns:
        df = df.drop(columns=[f"{pool.name}.seq"])

    logger.info("Completed library generation: %d sequences", len(df))
    if seqs_only:
        return list(df["seq"])
    return df


## THIS IS THE TOPOLOGICAL SORTING FUNCTION THAT IS USED TO DETERMINE THE ORDER OF OPERATIONS.
def _topo_sort_operations(outputs: dict) -> list:
    """Topologically sort operations reachable from outputs."""
    from .operation import Operation

    visited: set[int] = set()
    result: list[Operation] = []

    def visit(pool: Pool_type) -> None:
        op = pool.operation
        if op.id in visited:
            return
        for parent in op.parent_pools:
            visit(parent)
        visited.add(op.id)
        result.append(op)

    for pool in outputs.values():
        visit(pool)
    return result


def _seed_random_operations(sorted_ops: list, master_seed: int) -> None:
    """Clear RNG on all operations (RNG is created per-call in _compute_one)."""
    for op in sorted_ops:
        op.rng = None


def _collect_all_pools(outputs: dict) -> set:
    """Collect all pools reachable from the outputs."""
    visited: set[int] = set()
    result: set = set()

    def visit(pool: Pool_type) -> None:
        pool_id = id(pool)
        if pool_id in visited:
            return
        visited.add(pool_id)
        result.add(pool)
        for parent in pool.operation.parent_pools:
            visit(parent)

    for pool in outputs.values():
        visit(pool)
    return result


def _collect_counters(
    pools_filter: set,
    include_pool_states: bool = True,
    include_op_states: bool = True,
) -> list[st.State]:
    """Collect counters from the specified pools."""
    visited: set[int] = set()
    result: list[st.State] = []
    for pool in pools_filter:
        if include_pool_states:
            counter_id = id(pool.state)
            if counter_id not in visited:
                visited.add(counter_id)
                result.append(pool.state)
        if include_op_states:
            op_counter = pool.operation.state
            op_counter_id = id(op_counter)
            if op_counter_id not in visited:
                visited.add(op_counter_id)
                result.append(op_counter)
    return result


## THIS IS THE FUNCTION THAT COMPUTES ONE ROW OF OUTPUT FOR THE GIVEN GLOBAL STATE.
## CALLS EACH OPERATION IN THE TOPOLOGICAL SORT ORDER AND CACHES THE RESULTS.
def _compute_one(
    pool: Pool_type,
    sorted_ops: list,
    outputs: dict,
    global_state: int,
    max_global_state: int,
    states: list[st.State] = (),
    report_design_cards: bool = True,
    ops_to_report: set = None,
    pools_filter: set = None,
    include_inline_styles: bool = False,
) -> dict:
    """Compute one row of output for the given global state."""
    seq_cache: dict[int, Seq] = {}
    card_cache: dict[int, dict] = {}
    row: dict = {}

    # Sets the value of the pool state and, in doing so, propagates values
    # to all parent pool and operation states in the DAG.
    pool.state.value = global_state % pool.state.num_values

    # Collect all name contributions from operations in topological order
    all_contributions: list[str] = []

    # Iterates over the operations in topological order (sources to final).
    # This is the code that effectively implements the DAG.
    for op in sorted_ops:
        # Get parent Seq objects (already cached because of topological sort)
        parents = [seq_cache[p.operation.id] for p in op.parent_pools]

        # Determine RNG for this operation
        if op.mode == "random":
            if op.action_uniquely_determined_by_state:
                # Explicit num_states > 1: use state value
                state = op.state.value if op.state.value is not None else 0
            else:
                # Stateless random (num_states=None or 1): use global_state (row number)
                state = global_state
            seed_seq = np.random.SeedSequence([pool._master_seed, op.id, state])
            op_rng = np.random.default_rng(seed_seq)
        else:
            op_rng = op.rng

        # Compute output Seq and design card (handles region wrapping automatically)
        output_seq, card = op.compute(parents, op_rng)

        # Store in caches for downstream operations
        seq_cache[op.id] = output_seq
        card_cache[op.id] = card

        # Collect name contributions from this operation
        all_contributions.extend(op.compute_name_contributions(global_state, max_global_state))

        # Design cards are already filtered in Operation.compute()
        if report_design_cards and (ops_to_report is None or op.id in ops_to_report):
            for key, value in card.items():
                # Return None if state is inactive (on an inactive branch)
                if not op.state.is_active:
                    row[f"{op.name}.key.{key}"] = None
                else:
                    row[f"{op.name}.key.{key}"] = value

    # Read state values AFTER design card computation
    # (allows operations like StackOp to set state value during compute_design_card)
    for i, state in enumerate(states):
        col_name = counter_col_name(state, i)
        row[col_name] = state.value

    for output_name, output_pool in outputs.items():
        # Return None if pool state is inactive (on an inactive branch)
        if not output_pool.state.is_active:
            row[output_name] = None
        else:
            seq_obj = seq_cache[output_pool.operation.id]
            # Handle NullSeq - convert to empty string for DataFrame output
            if is_null_seq(seq_obj):
                row[output_name] = ""
            else:
                row[output_name] = seq_obj.string

    # Compute final name from contributions (already in topological order)
    final_name = ".".join(all_contributions) if all_contributions else None
    row["name"] = final_name

    # Get inline styles from final Seq object (only if requested)
    if include_inline_styles:
        final_seq = seq_cache[pool.operation.id]
        row["_inline_styles"] = final_seq.style

    return row
