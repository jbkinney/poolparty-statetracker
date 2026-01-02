"""Insert a marker at a fixed position in sequences."""
from numbers import Real
from poolparty.types import Union, Optional

from .parsing import build_marker_tag, get_length_without_markers


def insert_marker(
    pool,
    marker_name: str,
    start: int,
    stop: Optional[int] = None,
    strand: str = '+',
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
):
    """
    Insert an XML-style marker at a fixed position in sequences.

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string to add marker to.
    marker_name : str
        Name for the marker (e.g., 'region', 'orf', 'insert').
    start : int
        Start position (0-based) for the marker.
    stop : Optional[int], default=None
        End position (exclusive). If None, creates a zero-length marker at start.
    strand : str, default='+'
        Strand annotation ('+' or '-').
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
        A Pool yielding sequences with the marker inserted.

    Examples
    --------
    >>> with pp.Party():
    ...     bg = pp.from_seq('ACGTACGT')
    ...     # Region marker encompassing positions 2-5
    ...     marked = pp.insert_marker(bg, 'region', start=2, stop=5)
    ...     # Result: 'AC<region>GTA</region>CGT'
    ...
    ...     # Zero-length marker at position 4
    ...     marked = pp.insert_marker(bg, 'ins', start=4)
    ...     # Result: 'ACGT<ins/>ACGT'
    """
    from ..operations.from_seq import from_seq
    from ..operations.fixed import fixed_operation
    from ..party import get_active_party
    
    # Convert string to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool
    
    # Validate strand
    if strand not in ('+', '-'):
        raise ValueError(f"strand must be '+' or '-', got {strand!r}")
    
    # Validate positions
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")
    if stop is not None and stop < start:
        raise ValueError(f"stop ({stop}) must be >= start ({start})")
    
    # Calculate marker seq_length and register with Party
    marker_seq_length = (stop - start) if stop is not None else 0
    party = get_active_party()
    marker = party.register_marker(marker_name, marker_seq_length)
    
    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        seq_len = get_length_without_markers(seq)
        
        # Validate positions against sequence length
        if start > seq_len:
            raise ValueError(
                f"start ({start}) exceeds sequence length ({seq_len})"
            )
        actual_stop = stop if stop is not None else start
        if actual_stop > seq_len:
            raise ValueError(
                f"stop ({actual_stop}) exceeds sequence length ({seq_len})"
            )
        
        # Build the marker tag
        if stop is None or stop == start:
            # Zero-length marker
            marker_tag = build_marker_tag(marker_name, '', strand)
            return seq[:start] + marker_tag + seq[start:]
        else:
            # Region marker
            content = seq[start:stop]
            marker_tag = build_marker_tag(marker_name, content, strand)
            return seq[:start] + marker_tag + seq[stop:]
    
    # Sequence length changes due to marker tags (unknown statically)
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
    )
    
    # Add the new marker to the pool's marker set
    result_pool.add_marker(marker)
    
    return result_pool