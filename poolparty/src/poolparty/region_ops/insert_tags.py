"""Insert region tags at a fixed position in sequences."""
from numbers import Real
from poolparty.types import Union, Optional

from .parsing import build_region_tags, get_nontag_positions, nontag_pos_to_literal_pos


def insert_tags(
    pool,
    region_name: str,
    start: int,
    stop: Optional[int] = None,
    strand: str = '+',
    iter_order: Optional[Real] = None,
):
    """
    Insert XML-style region tags at a fixed position in sequences.

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string to add tags to.
    region_name : str
        Name for the region (e.g., 'region', 'orf', 'insert').
    start : int
        Start position (0-based) for the region.
    stop : Optional[int], default=None
        End position (exclusive). If None, creates a zero-length region at start.
    strand : str, default='+'
        Strand annotation ('+' or '-').
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool yielding sequences with the region tags inserted.

    Examples
    --------
    >>> with pp.Party():
    ...     bg = pp.from_seq('ACGTACGT')
    ...     # Region tags encompassing positions 2-5
    ...     marked = pp.insert_tags(bg, 'region', start=2, stop=5)
    ...     # Result: 'AC<region>GTA</region>CGT'
    ...
    ...     # Zero-length region at position 4
    ...     marked = pp.insert_tags(bg, 'ins', start=4)
    ...     # Result: 'ACGT<ins/>ACGT'
    """
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.fixed import fixed_operation
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
    
    # Calculate region seq_length and register with Party
    region_seq_length = (stop - start) if stop is not None else 0
    party = get_active_party()
    region = party.register_region(region_name, region_seq_length)
    
    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        seq_len = len(get_nontag_positions(seq))
        
        # Validate against non-tag length
        if start > seq_len:
            raise ValueError(f"start ({start}) exceeds sequence length ({seq_len})")
        actual_stop = stop if stop is not None else start
        region_length = actual_stop - start
        if actual_stop > seq_len:
            raise ValueError(f"stop ({actual_stop}) exceeds sequence length ({seq_len})")
        
        # Convert to literal positions and build tags
        literal_start = nontag_pos_to_literal_pos(seq, start)
        literal_stop = nontag_pos_to_literal_pos(seq, actual_stop)
        content = seq[literal_start:literal_stop] if region_length > 0 else ''
        region_tag = build_region_tags(region_name, content, strand)
        return seq[:literal_start] + region_tag + seq[literal_stop:]
    
    result_pool = fixed_operation(
        parent_pools=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=lambda lengths: None,  # Length changes due to region tags
        iter_order=iter_order,
    )
    
    # Add the new region to the pool's region set
    result_pool.add_region(region)
    
    return result_pool
