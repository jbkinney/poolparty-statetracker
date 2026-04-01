"""Extract content from a region as a new Pool."""

from numbers import Real

from poolparty.types import Optional

from ..utils import dna_utils
from ..utils.parsing_utils import validate_single_region


def extract_region(
    pool,
    region_name: str,
    rc: bool = False,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
):
    """
    Extract content from a named region as a new Pool.

    Creates a Pool that yields the content inside the specified region.

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string containing the region.
    region_name : str
        Name of the region to extract content from.
    rc : bool, default=False
        If True, reverse-complement the extracted content.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.

    Returns
    -------
    Pool
        A Pool yielding the content inside the region.

    Examples
    --------
    >>> with pp.Party():
    ...     bg = pp.from_seq('ACGT<region>TTAA</region>GCGC')
    ...     content = pp.extract_region(bg, 'region')
    ...     # content yields: 'TTAA'
    ...
    ...     # With rc=True, content is reverse-complemented
    ...     content_rc = pp.extract_region(bg, 'region', rc=True)
    ...     # content_rc yields: 'TTAA' (reverse complement of TTAA)
    """
    from ..fixed_ops.fixed import fixed_operation
    from ..fixed_ops.from_seq import from_seq
    from ..party import get_active_party

    # Convert string to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool

    def seq_from_seqs_fn(seqs: list[str]) -> str:
        seq = seqs[0]
        region = validate_single_region(seq, region_name)
        content = region.content

        # If rc=True, reverse complement the content
        if rc:
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
        prefix=prefix,
    )

    # The extracted content does not contain any regions
    # (we only inherit parent regions minus the extracted one)
    result_pool._untrack_region(region_name)

    return result_pool
