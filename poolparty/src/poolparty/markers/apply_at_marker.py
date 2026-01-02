"""Apply a transformation to content at a marked region."""
from numbers import Real
from typing import Optional, Callable


def apply_at_marker(
    pool,
    marker_name: str,
    transform_fn: Callable,
    name: Optional[str] = None,
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
        Examples: pp.reverse_complement, pp.seq_shuffle, lambda p: pp.mutagenize(p, ...)
    name : Optional[str], default=None
        Name for the resulting Pool.
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.

    Returns
    -------
    Pool
        A Pool with the marker region transformed and the marker removed.

    Examples
    --------
    >>> with pp.Party():
    ...     # Reverse complement a marked region
    ...     bg = pp.from_seq('ACGT<orf>ATGCCC</orf>TTTT')
    ...     result = pp.apply_at_marker(bg, 'orf', pp.reverse_complement)
    ...     # Result: 'ACGTGGGCATTTTT'
    ...
    ...     # Shuffle a marked region
    ...     bg = pp.from_seq('AAA<region>ACGTACGT</region>TTT')
    ...     result = pp.apply_at_marker(
    ...         bg, 'region',
    ...         lambda p: pp.seq_shuffle(p, mode='random')
    ...     )
    ...     # Result: 'AAA[shuffled]TTT'
    ...
    ...     # Mutagenize a marked region
    ...     bg = pp.from_seq('FLANK<target>ACGT</target>FLANK')
    ...     result = pp.apply_at_marker(
    ...         bg, 'target',
    ...         lambda p: pp.mutagenize(p, num_mutations=1, mode='sequential')
    ...     )
    ...     # Result: multiple variants with 1 mutation in the target region

    Notes
    -----
    The transform_fn receives content in + strand orientation (strand='-' markers
    have their content reverse-complemented before extraction). The transformed
    content is reverse-complemented back before insertion if strand='-'.
    """
    from ..operations.from_seq import from_seq
    from .extract_marker_content import extract_marker_content
    from .replace_marker_content import replace_marker_content
    
    # Convert string to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool
    
    # Step 1: Extract content from the marker
    # (automatically reverse-complements if strand='-')
    content_pool = extract_marker_content(pool, marker_name)
    
    # Step 2: Apply the transformation
    transformed_pool = transform_fn(content_pool)
    
    # Step 3: Replace marker with transformed content
    # (automatically reverse-complements back if strand='-')
    result = replace_marker_content(
        pool,
        transformed_pool,
        marker_name,
        name=name,
        iter_order=iter_order,
    )
    
    return result
