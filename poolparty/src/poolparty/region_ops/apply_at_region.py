"""Apply a transformation to content at a region."""

from numbers import Real

from poolparty.types import Callable, Optional

from ..utils import dna_utils


def apply_at_region(
    pool,
    region_name: str,
    transform_fn: Callable,
    rc: bool = False,
    remove_tags: bool = True,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
):
    """
    Apply a transformation to the content of a region.

    This is a high-level convenience function that:
    1. Extracts content from the named region (reverse-complementing if rc=True)
    2. Applies transform_fn to create a transformed content Pool
    3. Replaces the region with the transformed content (reverse-complementing back if rc=True)

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string containing the region.
    region_name : str
        Name of the region whose content to transform.
    transform_fn : Callable[[Pool], Pool]
        Function that takes a Pool and returns a transformed Pool.
        Examples: pp.rc, pp.shuffle_seq, lambda p: pp.mutagenize(p, ...)
    rc : bool, default=False
        If True, reverse-complement content before transform and
        reverse-complement result back before insertion.
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
    If rc=True, the transform_fn receives reverse-complemented content,
    and the result is reverse-complemented back before insertion.
    """
    from ..fixed_ops.from_seq import from_seq
    from .extract_region import extract_region
    from .replace_region import replace_region

    # Convert string to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool

    # Step 1: Extract content from the region
    content_pool = extract_region(pool, region_name, rc=rc)

    # Step 2: Apply the transformation
    transformed_pool = transform_fn(content_pool)

    if remove_tags:
        # Step 3a: Replace region with transformed content (tags removed)
        result = replace_region(
            pool,
            transformed_pool,
            region_name,
            rc=rc,
            iter_order=iter_order,
            prefix=prefix,
        )
    else:
        # Step 3b: Replace region content but keep tags
        result = _replace_keeping_tags(
            pool,
            transformed_pool,
            region_name,
            rc=rc,
            iter_order=iter_order,
            prefix=prefix,
        )

    return result


def _replace_keeping_tags(
    pool,
    content_pool,
    region_name: str,
    rc: bool = False,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
):
    """Replace region content while preserving region tags."""
    from ..fixed_ops.fixed import fixed_operation
    from ..party import get_active_party
    from ..utils.parsing_utils import build_region_tags, validate_single_region

    # Compute output biological length: same formula as replace_region
    # (tags are preserved but don't affect biological length)
    party = get_active_party()
    old_region_len = None
    if party is not None and party.has_region(region_name):
        old_region_len = party.get_region(region_name).seq_length

    def _seq_length_fn(lengths):
        parent_len = lengths[0]
        content_len = lengths[1] if len(lengths) > 1 else None
        if parent_len is None or old_region_len is None or content_len is None:
            return None
        return parent_len - old_region_len + content_len

    def seq_from_seqs_fn(seqs: list[str]) -> str:
        bg_seq = seqs[0]
        content_seq = seqs[1]

        # Find the region in the background sequence
        region = validate_single_region(bg_seq, region_name)

        # If rc=True, reverse complement the content before insertion
        if rc:
            content_seq = dna_utils.reverse_complement(content_seq)

        # Build wrapped content with region tags
        wrapped = build_region_tags(region_name, content_seq)

        # Build result: prefix + wrapped + suffix
        prefix = bg_seq[: region.start]
        suffix = bg_seq[region.end :]
        return prefix + wrapped + suffix

    result_pool = fixed_operation(
        parent_pools=[pool, content_pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=_seq_length_fn,
        iter_order=iter_order,
        prefix=prefix,
    )

    # Region is preserved, so keep it in the pool's region set
    # (it was inherited from pool, so nothing to add)

    return result_pool
