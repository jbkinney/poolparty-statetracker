"""Sequence generation functions for poolparty."""
import statecounter as sc
from .types import Pool_type, Union, Sequence, Literal, Optional, beartype
from .utils import clean_df_int_columns
from .df_format import counter_col_name, organize_columns, finalize_generate_df
import numpy as np
import pandas as pd


@beartype
def generate_seqs(
    pool: Pool_type,
    num_seqs: Optional[int] = None,
    num_cycles: Optional[int] = None,
    seed: Optional[int] = None,
    init_state: Optional[int] = None,
    aux_pools: Sequence[Pool_type] = (),
    report_seq: bool = True,
    report_pool_seqs: bool = True,
    report_pool_states: bool = True,
    report_op_states: bool = True,
    report_op_keys: bool = True,
    pools_to_report: Union[str, Sequence[Pool_type]] = 'all',
    organize_columns_by: Literal['pool', 'type'] = 'type',
) -> pd.DataFrame:
    """Generate sequences from a pool.
    
    Args:
        pool: The pool to generate sequences from.
        num_seqs: Number of sequences to generate.
        num_cycles: Number of complete iterations through all states.
        seed: Random seed for reproducibility.
        init_state: Initial state to start generation from.
        aux_pools: Additional pools to include in output.
        report_seq: Whether to include the main sequence column.
        report_pool_seqs: Whether to include per-pool sequence columns.
        report_pool_states: Whether to include pool state columns.
        report_op_states: Whether to include operation state columns.
        report_op_keys: Whether to include operation key columns.
        pools_to_report: Which pools to report ('all', 'self', or a list).
        organize_columns_by: How to organize columns ('pool' or 'type').
    
    Returns:
        DataFrame with generated sequences and metadata.
    """
    # Initialize state tracking on pool if not present
    if not hasattr(pool, '_current_state'):
        pool._current_state = 0
    if not hasattr(pool, '_master_seed'):
        pool._master_seed = None
    
    # Validate arguments
    if num_seqs is not None and num_cycles is not None:
        raise ValueError("Specify num_seqs OR num_cycles, not both")
    if num_seqs is None and num_cycles is None:
        raise ValueError("Must specify num_seqs or num_cycles")
    
    if num_cycles is not None:
        num_seqs = num_cycles * pool.counter.num_states
    if init_state is not None:
        pool._current_state = init_state
    if seed is not None:
        pool._master_seed = seed
    if pool._master_seed is None:
        pool._master_seed = 0
    
    # Build outputs dict
    outputs: dict[str, Pool_type] = {f'{pool.name}.seq': pool}
    for aux_pool in aux_pools:
        outputs[f'{aux_pool.name}.seq'] = aux_pool
    
    sorted_ops = _topo_sort_operations(outputs)
    _seed_random_operations(sorted_ops, pool._master_seed)
    
    # Determine which pools to report
    if pools_to_report == 'all':
        pools_filter = _collect_all_pools(outputs)
    elif pools_to_report == 'self':
        pools_filter = {pool}
    else:
        pools_filter = set(pools_to_report)
    
    ops_to_report = {p.operation.id for p in pools_filter}
    
    # Add filtered pools to outputs if not already present
    for p in pools_filter:
        key = f'{p.name}.seq'
        if key not in outputs:
            outputs[key] = p
    
    counters = _collect_counters(pools_filter, report_pool_states, report_op_states)
    
    # Generate rows
    rows = []
    for i in range(num_seqs):
        global_state = pool._current_state + i
        row = _compute_one(
            pool, sorted_ops, outputs, global_state, 
            counters, report_op_keys, ops_to_report
        )
        rows.append(row)
    
    pool._current_state += num_seqs
    
    # Build and format DataFrame
    df = pd.DataFrame(rows)
    df = clean_df_int_columns(df)
    df = organize_columns(df, pools_filter, organize_columns_by)
    df = finalize_generate_df(df, pool.name, report_seq, report_pool_seqs)
    return df


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
    """Set up shared RNG for random mode operations.
    
    Note: hybrid mode operations don't get an RNG here - their RNG
    is created per-state in _compute_one using SeedSequence.
    """
    shared_rng = np.random.default_rng(master_seed)
    for op in sorted_ops:
        if op.mode == 'random':
            op.rng = shared_rng
        else:
            # hybrid, sequential, fixed modes don't use shared RNG
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
    include_pool_counters: bool = True,
    include_op_counters: bool = True,
) -> list[sc.Counter]:
    """Collect counters from the specified pools."""
    visited: set[int] = set()
    result: list[sc.Counter] = []
    for pool in pools_filter:
        if include_pool_counters:
            counter_id = id(pool.counter)
            if counter_id not in visited:
                visited.add(counter_id)
                result.append(pool.counter)
        if include_op_counters:
            op_counter = pool.operation.counter
            op_counter_id = id(op_counter)
            if op_counter_id not in visited:
                visited.add(op_counter_id)
                result.append(op_counter)
    return result


def _compute_one(
    pool: Pool_type,
    sorted_ops: list,
    outputs: dict,
    global_state: int,
    counters: list[sc.Counter] = (),
    report_op_keys: bool = True,
    ops_to_report: set = None,
) -> dict:
    """Compute one row of output for the given global state."""
    cache: dict[int, dict] = {}
    card_cache: dict[int, dict] = {}
    row: dict = {}
    
    pool.counter.state = global_state % pool.counter.num_states
    
    for i, counter in enumerate(counters):
        col_name = counter_col_name(counter, i)
        row[col_name] = counter.state
    
    for op in sorted_ops:
        parent_seqs = []
        for parent in op.parent_pools:
            parent_result = cache[parent.operation.id]
            seq_key = f"seq_{parent.output_index}"
            parent_seqs.append(parent_result[seq_key])
        
        # Determine RNG for this operation
        if op.mode == 'hybrid':
            # Create state-specific RNG for hybrid mode using SeedSequence
            state = op.counter.state if op.counter.state is not None else 0
            seed_seq = np.random.SeedSequence([pool._master_seed, op.id, state])
            op_rng = np.random.default_rng(seed_seq)
        else:
            op_rng = op.rng
        
        # Compute design card first, then sequences from card
        card = op.compute_design_card(parent_seqs, op_rng)
        seqs = op.compute_seq_from_card(parent_seqs, card)
        
        # Store sequences in cache for downstream operations
        cache[op.id] = seqs
        # Store design card separately for reporting
        card_cache[op.id] = card
        
        if report_op_keys and (ops_to_report is None or op.id in ops_to_report):
            for key in op.design_card_keys:
                if key in card:
                    if op.counter.state is None:
                        row[f"{op.name}.key.{key}"] = None
                    else:
                        row[f"{op.name}.key.{key}"] = card[key]
    
    for output_name, output_pool in outputs.items():
        if output_pool.counter.state is None:
            row[output_name] = None
        else:
            result = cache[output_pool.operation.id]
            seq_key = f"seq_{output_pool.output_index}"
            row[output_name] = result[seq_key]
    
    return row

