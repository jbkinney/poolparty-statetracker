"""Library generation functions for poolparty."""
import statetracker as st
from .types import Pool_type, Union, Sequence, Literal, Optional, beartype, Seq
from .utils.utils import clean_df_int_columns
from .utils.df_utils import counter_col_name, organize_columns, finalize_generate_df
import numpy as np
import pandas as pd


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
    report_seq: bool = True,
    report_pool_seqs: bool = True,
    report_pool_states: bool = True,
    report_op_states: bool = True,
    report_op_keys: bool = True,
    pools_to_report: Union[str, Sequence[Pool_type]] = 'all',
    organize_columns_by: Literal['pool', 'type'] = 'type',
    suppress_styles: bool = False,
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
            When False (default), returns minimal DataFrame with "name" and "seq".
    
    Returns:
        DataFrame with generated sequences, or list of sequences if seqs_only=True.
    """
    # Initialize state tracking on pool if not present
    if not hasattr(pool, '_current_state'):
        pool._current_state = 0
    if not hasattr(pool, '_master_seed'):
        pool._master_seed = None
    
    # Validate arguments    
    if num_seqs is None:
        if pool.state is None:
            raise ValueError(
                "num_seqs must be specified when pool has no state (mode='random' with num_states=None). "
                "Cannot use num_cycles for stateless pools."
            )
        num_seqs = num_cycles * pool.state.num_values
    if init_state is not None:
        pool._current_state = init_state
    if seed is not None:
        pool._master_seed = seed
    if pool._master_seed is None:
        pool._master_seed = 0
    
    # Build outputs dict
    outputs: dict[str, Pool_type] = {f'{pool.name}.seq': pool}
    if report_design_cards:
        for aux_pool in aux_pools:
            outputs[f'{aux_pool.name}.seq'] = aux_pool
    
    sorted_ops = _topo_sort_operations(outputs)
    _seed_random_operations(sorted_ops, pool._master_seed)
    
    # Determine which pools to report (only used when report_design_cards=True)
    if report_design_cards:
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
        
        states = _collect_counters(pools_filter, report_pool_states, report_op_states)
    else:
        pools_filter = {pool}
        ops_to_report = set()
        states = []
    
    # Generate rows
    rows = []
    for i in range(num_seqs):
        global_state = pool._current_state + i
        row = _compute_one(
            pool, sorted_ops, outputs, global_state, 
            states, report_op_keys if report_design_cards else False, 
            ops_to_report, pools_filter, suppress_styles
        )
        rows.append(row)
    
    pool._current_state += num_seqs
    
    # Build and format DataFrame
    df = pd.DataFrame(rows)
    
    if not report_design_cards:
        # Minimal output: just "name" and "seq" columns
        df = df[['name', f'{pool.name}.seq']].rename(columns={f'{pool.name}.seq': 'seq'})
        if seqs_only:
            return list(df['seq'])
        return df
    
    # Full design card output
    df = clean_df_int_columns(df)
    df = organize_columns(df, pools_filter, organize_columns_by)
    df = finalize_generate_df(df, pool.name, report_seq, report_pool_seqs, pools_filter)
    if seqs_only:
        return list(df['seq'])
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
    """Set up shared RNG for random mode operations.
    
    Note: random mode operations with num_states > 1 don't get an RNG here - their RNG
    is created per-state in _compute_one using SeedSequence.
    """
    shared_rng = np.random.default_rng(master_seed)
    for op in sorted_ops:
        if op.mode == 'random' and op.state is None:
            # Pure random mode (num_states=None) uses shared RNG
            op.rng = shared_rng
        else:
            # random with num_states > 1, sequential, fixed modes don't use shared RNG
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
        if include_pool_states and pool.state is not None:
            counter_id = id(pool.state)
            if counter_id not in visited:
                visited.add(counter_id)
                result.append(pool.state)
        if include_op_states and pool.operation.state is not None:
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
    states: list[st.State] = (),
    report_op_keys: bool = True,
    ops_to_report: set = None,
    pools_filter: set = None,
    suppress_styles: bool = False,
) -> dict:
    """Compute one row of output for the given global state."""
    seq_cache: dict[int, Seq] = {}
    card_cache: dict[int, dict] = {}
    row: dict = {}
    
    # Sets the value of the pool state and, in doing so, the value
    # of all pool and operation states that affect this state.
    # Skip if pool has no state (fully random DAG)
    if pool.state is not None:
        pool.state.value = global_state % pool.state.num_values
    
    # Collect all name contributions from operations in topological order
    all_contributions: list[str] = []
    
    # Iterates over the operations in topological order (sources to final).
    # This is the code that effectively implements the DAG.
    for op in sorted_ops:
        # Get parent Seq objects (already cached because of topological sort)
        parents = [seq_cache[p.operation.id] for p in op.parent_pools]
        
        # Determine RNG for this operation
        if op.mode == 'random' and op.state is not None:
            # Create state-specific RNG for random mode with state using SeedSequence
            # This handles both explicit num_states and auto-synced states from parents
            state = op.state.value if op.state.value is not None else 0
            seed_seq = np.random.SeedSequence([pool._master_seed, op.id, state])
            op_rng = np.random.default_rng(seed_seq)
        else:
            op_rng = op.rng
        
        # Compute output Seq and design card (handles region wrapping automatically)
        output_seq, card = op.compute(parents, op_rng, suppress_styles)
        
        # Store in caches for downstream operations
        seq_cache[op.id] = output_seq
        card_cache[op.id] = card
        
        # Collect name contributions from this operation
        all_contributions.extend(op.compute_name_contributions())
        
        if report_op_keys and (ops_to_report is None or op.id in ops_to_report):
            for key in op.design_card_keys:
                if key in card:
                    # For ops with state: return None if state is inactive (value=None)
                    # For stateless ops (state=None): always report design card
                    if op.state is not None and op.state.value is None:
                        row[f"{op.name}.key.{key}"] = None
                    else:
                        row[f"{op.name}.key.{key}"] = card[key]
    
    # Read state values AFTER design card computation
    # (allows operations like StackOp to set state value during compute_design_card)
    for i, state in enumerate(states):
        col_name = counter_col_name(state, i)
        row[col_name] = state.value
    
    for output_name, output_pool in outputs.items():
        # For pools with state: return None if state is inactive (value=None)
        # For stateless pools (state=None): always generate sequences
        if output_pool.state is not None and output_pool.state.value is None:
            row[output_name] = None
        else:
            seq_obj = seq_cache[output_pool.operation.id]
            row[output_name] = seq_obj.string
    
    # Compute final name from contributions (already in topological order)
    final_name = '.'.join(all_contributions) if all_contributions else None
    row['name'] = final_name
    
    # Get inline styles from final Seq object
    final_seq = seq_cache[pool.operation.id]
    row['_inline_styles'] = final_seq.style
    
    return row

