"""Extract content from a marker region as a new Pool."""
from numbers import Real
from typing import Union, Optional

from .parsing import validate_single_marker


def extract_marker_content(
    pool,
    marker_name: str,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
):
    """
    Extract content from a named marker as a new Pool.

    Creates a Pool that yields the content inside the specified marker.
    If the marker has strand='-', the content is reverse-complemented
    so operations always see + strand orientation.

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string containing the marker.
    marker_name : str
        Name of the marker to extract content from.
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
        A Pool yielding the content inside the marker.
        If marker has strand='-', content is reverse-complemented.

    Examples
    --------
    >>> with pp.Party():
    ...     bg = pp.from_seq('ACGT<region>TTAA</region>GCGC')
    ...     content = pp.extract_marker_content(bg, 'region')
    ...     # content yields: 'TTAA'
    ...
    ...     # With strand='-', content is reverse-complemented
    ...     bg = pp.from_seq("ACGT<region strand='-'>TTAA</region>GCGC")
    ...     content = pp.extract_marker_content(bg, 'region')
    ...     # content yields: 'TTAA' (reverse complement of TTAA)
    """
    from ..operations.from_seq import from_seq
    from ..operations.fixed import fixed_operation
    from ..party import get_active_party
    
    # Convert string to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool
    
    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        marker = validate_single_marker(seq, marker_name)
        content = marker.content
        
        # If strand='-', reverse complement the content
        if marker.strand == '-':
            party = get_active_party()
            alphabet = party.alphabet
            content = ''.join(alphabet.get_complement(c) for c in reversed(content))
        
        return content
    
    # Get seq_length from the registered marker
    party = get_active_party()
    if party.has_marker(marker_name):
        registered_marker = party.get_marker_by_name(marker_name)
        marker_seq_length = registered_marker.seq_length
    else:
        # Marker not registered - this shouldn't happen in normal usage
        # but we handle it gracefully by inferring from content
        marker_seq_length = None
    
    def seq_length_fn(pools) -> Optional[int]:
        # Use the registered marker's seq_length if available
        return marker_seq_length
    
    result_pool = fixed_operation(
        parents=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pools_fn=seq_length_fn,
        name=name,
        op_name=op_name,
        iter_order=iter_order,
        op_iter_order=op_iter_order,
    )
    
    # The extracted content does not contain any markers
    # (we only inherit parent markers minus the extracted one)
    result_pool.remove_marker(marker_name)
    
    return result_pool