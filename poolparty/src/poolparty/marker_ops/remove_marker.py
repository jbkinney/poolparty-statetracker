"""Remove a marker and its content from sequences."""
from numbers import Real
from poolparty.types import Optional

from .parsing import validate_single_marker


def remove_marker(
    pool,
    marker_name: str,
    keep_content: bool = True,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
):
    """
    Remove a marker from sequences.

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string containing the marker.
    marker_name : str
        Name of the marker to remove.
    keep_content : bool, default=True
        If True, keep the content inside the marker (just remove tags).
        If False, remove both the marker tags and their content.
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
        A Pool yielding sequences with the marker removed.

    Examples
    --------
    >>> with pp.Party():
    ...     bg = pp.from_seq('ACGT<region>TTAA</region>GCGC')
    ...
    ...     # Keep content (just remove tags)
    ...     result = pp.remove_marker(bg, 'region', keep_content=True)
    ...     # Result: 'ACGTTTAAGCGC'
    ...
    ...     # Remove content too
    ...     result = pp.remove_marker(bg, 'region', keep_content=False)
    ...     # Result: 'ACGTGCGC'
    """
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.fixed import fixed_operation
    
    # Convert string to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool
    
    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        marker = validate_single_marker(seq, marker_name)
        
        prefix = seq[:marker.start]
        suffix = seq[marker.end:]
        
        if keep_content:
            return prefix + marker.content + suffix
        else:
            return prefix + suffix
    
    # Sequence length changes when removing markers
    def seq_length_fn(pools) -> Optional[int]:
        return None
    
    result_pool = fixed_operation(
        parents=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pools_fn=seq_length_fn,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
        _factory_name='remove_marker',
    )
    
    # The marker is removed, so remove it from the pool's marker set
    result_pool._untrack_marker(marker_name)
    
    return result_pool