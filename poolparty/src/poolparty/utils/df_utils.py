"""DataFrame formatting utilities for poolparty."""

import pandas as pd

import statetracker as st

from ..types import Literal


def counter_col_name(state: "st.State", index: int) -> str:
    """Get column name for a state's value.

    Args:
        state: The state to get a column name for.
        index: Fallback index if state has no name or id.

    Returns:
        A string column name for the state.
    """
    if state.name:
        return state.name
    elif state.id is not None:
        return f"id_{state.id}"
    else:
        return f"id_{index}"


def get_pools_reverse_topo(pools: set) -> list:
    """Get pools in reverse topological order (children before parents).

    Args:
        pools: Set of Pool objects to order.

    Returns:
        List of pools ordered with children before their parents.
    """
    visited: set[int] = set()
    topo_order: list = []

    def visit(pool) -> None:
        pool_id = id(pool)
        if pool_id in visited:
            return
        for parent in pool.operation.parent_pools:
            if parent in pools:
                visit(parent)
        visited.add(pool_id)
        if pool in pools:
            topo_order.append(pool)

    for pool in pools:
        visit(pool)
    return list(reversed(topo_order))


def organize_columns(
    df: pd.DataFrame,
    pools_filter: set,
    organize_by: Literal["pool", "type"],
) -> pd.DataFrame:
    """Organize DataFrame columns by pool or by type.

    Args:
        df: The DataFrame to reorganize.
        pools_filter: Set of pools to consider for ordering.
        organize_by: Either 'pool' (group by pool) or 'type' (group by column type).

    Returns:
        DataFrame with columns reordered.
    """
    all_cols = set(df.columns)
    pools_in_order = get_pools_reverse_topo(pools_filter)

    if organize_by == "type":
        pool_order = {pool.name: i for i, pool in enumerate(pools_in_order)}
        op_order = {pool.operation.name: i for i, pool in enumerate(pools_in_order)}

        def pool_sort_key(col: str) -> int:
            name = col.split(".")[0]
            return pool_order.get(name, op_order.get(name, len(pool_order)))

        seq_cols = sorted([c for c in df.columns if c.endswith(".seq")], key=pool_sort_key)
        pool_state_names = {f"{name}.state" for name in pool_order}
        pool_state_cols = sorted(
            [c for c in df.columns if c in pool_state_names], key=pool_sort_key
        )
        op_state_cols = sorted(
            [c for c in df.columns if c.endswith(".state") and c not in pool_state_names],
            key=pool_sort_key,
        )
        key_cols = sorted([c for c in df.columns if ".key." in c], key=pool_sort_key)
        ordered_cols = seq_cols + pool_state_cols + op_state_cols + key_cols
    else:
        ordered_cols = []
        for pool in pools_in_order:
            pool_name = pool.name
            op_name = pool.operation.name
            pool_cols = []
            seq_col = f"{pool_name}.seq"
            if seq_col in all_cols:
                pool_cols.append(seq_col)
            state_col = f"{pool_name}.state"
            if state_col in all_cols:
                pool_cols.append(state_col)
            op_state_col = f"{op_name}.state"
            if op_state_col in all_cols:
                pool_cols.append(op_state_col)
                all_cols.discard(op_state_col)
            key_prefix = f"{op_name}.key."
            key_cols = sorted([c for c in df.columns if c.startswith(key_prefix) and c in all_cols])
            pool_cols.extend(key_cols)
            for c in key_cols:
                all_cols.discard(c)
            ordered_cols.extend(pool_cols)

    remaining = [c for c in df.columns if c not in ordered_cols]
    ordered_cols.extend(remaining)
    return df[ordered_cols]


def finalize_generate_df(
    df: pd.DataFrame,
    pool_name: str,
    show_seq: bool,
    show_pool_seqs: bool,
    pools_filter: set = None,
    show_name: bool = True,
) -> pd.DataFrame:
    """Apply final column transforms to generated DataFrame."""
    # Move 'name' column to position 0 if it exists and has values
    if "name" in df.columns:
        name_col = df.pop("name")
        if show_name and name_col.notna().any():
            df.insert(0, "name", name_col)

    if show_seq:
        # Insert 'seq' after 'name' if name exists, else at position 0
        insert_pos = 1 if "name" in df.columns else 0
        df.insert(insert_pos, "seq", df[f"{pool_name}.seq"])
    if not show_pool_seqs:
        seq_cols = [c for c in df.columns if c.endswith(".seq")]
        df = df.drop(columns=seq_cols)
    return df
