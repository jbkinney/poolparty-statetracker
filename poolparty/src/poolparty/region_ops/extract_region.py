"""Extract content from a region as a new Pool."""
from numbers import Real
from poolparty.types import Union, Optional

from .parsing import validate_single_region
from ..utils import dna_utils


def extract_region(
    pool,
    region_name: str,
    iter_order: Optional[Real] = None,
):
    """
    Extract content from a named region as a new Pool.

    Creates a Pool that yields the content inside the specified region.
    If the region has strand='-', the content is reverse-complemented
    so operations always see + strand orientation.

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string containing the region.
    region_name : str
        Name of the region to extract content from.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool yielding the content inside the region.
        If region has strand='-', content is reverse-complemented.

    Examples
    --------
    >>> with pp.Party():
    ...     bg = pp.from_seq('ACGT<region>TTAA</region>GCGC')
    ...     content = pp.extract_region(bg, 'region')
    ...     # content yields: 'TTAA'
    ...
    ...     # With strand='-', content is reverse-complemented
    ...     bg = pp.from_seq("ACGT<region strand='-'>TTAA</region>GCGC")
    ...     content = pp.extract_region(bg, 'region')
    ...     # content yields: 'TTAA' (reverse complement of TTAA)
    """
    from ..fixed_ops.from_seq import from_seq
    from ..fixed_ops.fixed import fixed_operation
    from ..party import get_active_party
    
    # Convert string to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool
    
    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        region = validate_single_region(seq, region_name)
        content = region.content
        
        # If strand='-', reverse complement the content
        if region.strand == '-':
            content = dna_utils.reverse_complement(content)
        
        return content
    
    # Get seq_length from the registered region
    party = get_active_party()
    if party.has_region(region_name):
        registered_region = party.get_region_by_name(region_name)
        region_seq_length = registered_region.seq_length
    else:
        # Region not registered - this shouldn't happen in normal usage
        # but we handle it gracefully by inferring from content
        region_seq_length = None
    
    result_pool = fixed_operation(
        parent_pools=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=lambda lengths: region_seq_length,  # Use registered region's seq_length
        iter_order=iter_order,
    )
    
    # The extracted content does not contain any regions
    # (we only inherit parent regions minus the extracted one)
    result_pool._untrack_region(region_name)
    
    return result_pool
