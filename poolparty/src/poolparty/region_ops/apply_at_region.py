"""Apply a transformation to content at a region."""
from numbers import Real
from poolparty.types import Optional, Callable
from ..utils import dna_utils


def apply_at_region(
    pool,
    region_name: str,
    transform_fn: Callable,
    remove_tags: bool = True,
    iter_order: Optional[Real] = None,
):
    """
    Apply a transformation to the content of a region.

    This is a high-level convenience function that:
    1. Extracts content from the named region (reverse-complementing if strand='-')
    2. Applies transform_fn to create a transformed content Pool
    3. Replaces the region with the transformed content (reverse-complementing back if strand='-')

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string containing the region.
    region_name : str
        Name of the region whose content to transform.
    transform_fn : Callable[[Pool], Pool]
        Function that takes a Pool and returns a transformed Pool.
        Examples: pp.rc, pp.shuffle_seq, lambda p: pp.mutagenize(p, ...)
    remove_tags : bool, default=True
        If True, region tags are removed from the result.
        If False, region tags are preserved around the transformed content.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool with the region content transformed.

    Examples
    --------
    >>> with pp.Party():
    ...     # Reverse complement a region (tags removed)
    ...     bg = pp.from_seq('ACGT<orf>ATGCCC</orf>TTTT')
    ...     result = pp.apply_at_region(bg, 'orf', pp.rc)
    ...     # Result: 'ACGTGGGCATTTTT'
    ...
    ...     # Keep tags around transformed content
    ...     bg = pp.from_seq('AAA<region>ACGT</region>TTT')
    ...     result = pp.apply_at_region(
    ...         bg, 'region',
    ...         lambda p: pp.mutagenize(p, num_mutations=1),
    ...         remove_tags=False,
    ...     )
    ...     # Result: 'AAA<region>ACCT</region>TTT' (tags preserved)

    Notes
    -----
    The transform_fn receives content in + strand orientation (strand='-' regions
    have their content reverse-complemented before extraction). The transformed
    content is reverse-complemented back before insertion if strand='-'.
    """
    from ..fixed_ops.from_seq import from_seq
    from .extract_region import extract_region
    from .replace_region import replace_region
    
    # Convert string to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool
    
    # Step 1: Extract content from the region
    # (automatically reverse-complements if strand='-')
    content_pool = extract_region(pool, region_name)
    
    # Step 2: Apply the transformation
    transformed_pool = transform_fn(content_pool)
    
    if remove_tags:
        # Step 3a: Replace region with transformed content (tags removed)
        result = replace_region(
            pool,
            transformed_pool,
            region_name,
            iter_order=iter_order,
        )
    else:
        # Step 3b: Replace region content but keep tags
        result = _replace_keeping_tags(
            pool,
            transformed_pool,
            region_name,
            iter_order=iter_order,
        )
    
    return result


def _replace_keeping_tags(
    pool,
    content_pool,
    region_name: str,
    iter_order: Optional[Real] = None,
):
    """Replace region content while preserving region tags."""
    from ..fixed_ops.fixed import fixed_operation
    from ..utils.parsing_utils import validate_single_region, build_region_tags
    
    def seq_from_seqs_fn(seqs: list[str]) -> str:
        bg_seq = seqs[0]
        content_seq = seqs[1]
        
        # Find the region in the background sequence
        region = validate_single_region(bg_seq, region_name)
        
        # If strand='-', reverse complement the content before insertion
        if region.strand == '-':
            content_seq = dna_utils.reverse_complement(content_seq)
        
        # Build wrapped content with region tags
        wrapped = build_region_tags(
            region_name,
            content_seq,
            strand=region.strand,
        )
        
        # Build result: prefix + wrapped + suffix
        prefix = bg_seq[:region.start]
        suffix = bg_seq[region.end:]
        return prefix + wrapped + suffix
    
    result_pool = fixed_operation(
        parent_pools=[pool, content_pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=lambda lengths: None,  # Variable length
        iter_order=iter_order,
    )
    
    # Region is preserved, so keep it in the pool's region set
    # (it was inherited from pool, so nothing to add)
    
    return result_pool
