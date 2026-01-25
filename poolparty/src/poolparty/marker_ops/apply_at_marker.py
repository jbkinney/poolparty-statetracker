"""Apply a transformation to content at a marked region."""
from numbers import Real
from poolparty.types import Optional, Callable
from .. import dna


def apply_at_marker(
    pool,
    marker_name: str,
    transform_fn: Callable,
    remove_marker: bool = True,
    iter_order: Optional[Real] = None,
):
    """
    Apply a transformation to the content of a marked region.

    This is a high-level convenience function that:
    1. Extracts content from the named marker (reverse-complementing if strand='-')
    2. Applies transform_fn to create a transformed content Pool
    3. Replaces the marker with the transformed content (reverse-complementing back if strand='-')

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string containing the marker.
    marker_name : str
        Name of the marker whose content to transform.
    transform_fn : Callable[[Pool], Pool]
        Function that takes a Pool and returns a transformed Pool.
        Examples: pp.rc, pp.shuffle_seq, lambda p: pp.mutagenize(p, ...)
    remove_marker : bool, default=True
        If True, marker tags are removed from the result.
        If False, marker tags are preserved around the transformed content.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool with the marker region transformed.

    Examples
    --------
    >>> with pp.Party():
    ...     # Reverse complement a marked region (marker removed)
    ...     bg = pp.from_seq('ACGT<orf>ATGCCC</orf>TTTT')
    ...     result = pp.apply_at_marker(bg, 'orf', pp.rc)
    ...     # Result: 'ACGTGGGCATTTTT'
    ...
    ...     # Keep marker tags around transformed content
    ...     bg = pp.from_seq('AAA<region>ACGT</region>TTT')
    ...     result = pp.apply_at_marker(
    ...         bg, 'region',
    ...         lambda p: pp.mutagenize(p, num_mutations=1),
    ...         remove_marker=False,
    ...     )
    ...     # Result: 'AAA<region>ACCT</region>TTT' (marker tags preserved)

    Notes
    -----
    The transform_fn receives content in + strand orientation (strand='-' markers
    have their content reverse-complemented before extraction). The transformed
    content is reverse-complemented back before insertion if strand='-'.
    """
    from ..fixed_ops.from_seq import from_seq
    from .extract_marker_content import extract_marker_content
    from .replace_marker_content import replace_marker_content
    
    # Convert string to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool
    
    # Step 1: Extract content from the marker
    # (automatically reverse-complements if strand='-')
    content_pool = extract_marker_content(pool, marker_name)
    
    # Step 2: Apply the transformation
    transformed_pool = transform_fn(content_pool)
    
    if remove_marker:
        # Step 3a: Replace marker with transformed content (tags removed)
        result = replace_marker_content(
            pool,
            transformed_pool,
            marker_name,
            iter_order=iter_order,
        )
    else:
        # Step 3b: Replace marker content but keep marker tags
        result = _replace_keeping_marker(
            pool,
            transformed_pool,
            marker_name,
            iter_order=iter_order,
        )
    
    return result


def _replace_keeping_marker(
    bg_pool,
    content_pool,
    marker_name: str,
    iter_order: Optional[Real] = None,
):
    """Replace marker content while preserving marker tags."""
    from ..fixed_ops.fixed import fixed_operation
    from .parsing import validate_single_marker, build_marker_tag
    
    def seq_from_seqs_fn(seqs: list[str]) -> str:
        bg_seq = seqs[0]
        content_seq = seqs[1]
        
        # Find the marker in the background sequence
        marker = validate_single_marker(bg_seq, marker_name)
        
        # If strand='-', reverse complement the content before insertion
        if marker.strand == '-':
            content_seq = dna.reverse_complement(content_seq)
        
        # Build wrapped content with marker tags
        wrapped = build_marker_tag(
            marker_name,
            content_seq,
            strand=marker.strand,
        )
        
        # Build result: prefix + wrapped + suffix
        prefix = bg_seq[:marker.start]
        suffix = bg_seq[marker.end:]
        return prefix + wrapped + suffix
    
    result_pool = fixed_operation(
        parent_pools=[bg_pool, content_pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=lambda lengths: None,  # Variable length
        iter_order=iter_order,
    )
    
    # Marker is preserved, so keep it in the pool's marker set
    # (it was inherited from bg_pool, so nothing to add)
    
    return result_pool
