"""Insert region tags at a fixed position in sequences."""

from numbers import Real

from poolparty.types import Optional

from ..utils.parsing_utils import build_region_tags, get_nontag_positions


def insert_tags(
    pool,
    region_name: str,
    start: int,
    stop: Optional[int] = None,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
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
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.

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
    from ..fixed_ops.fixed import fixed_operation
    from ..fixed_ops.from_seq import from_seq
    from ..party import get_active_party

    # Convert string to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool

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
        nontag_positions = get_nontag_positions(seq)
        seq_len = len(nontag_positions)

        # Validate against non-tag length
        if start > seq_len:
            raise ValueError(f"start ({start}) exceeds sequence length ({seq_len})")
        actual_stop = stop if stop is not None else start
        region_length = actual_stop - start
        if actual_stop > seq_len:
            raise ValueError(f"stop ({actual_stop}) exceeds sequence length ({seq_len})")

        # Convert to literal positions
        literal_start = nontag_positions[start] if start < seq_len else len(seq)
        literal_stop = nontag_positions[actual_stop] if actual_stop < seq_len else len(seq)

        # Walk literal_stop backward past any opening tags at the boundary
        # to avoid capturing them inside the region content (crossing XML).
        while literal_stop > literal_start and seq[literal_stop - 1] == '>':
            tag_start = seq.rfind('<', literal_start, literal_stop)
            if tag_start < 0:
                break
            tag = seq[tag_start:literal_stop]
            if tag.startswith('</') or tag.endswith('/>'):
                break
            literal_stop = tag_start

        content = seq[literal_start:literal_stop] if region_length > 0 else ""
        region_tag = build_region_tags(region_name, content)
        return seq[:literal_start] + region_tag + seq[literal_stop:]

    result_pool = fixed_operation(
        parent_pools=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=lambda lengths: lengths[0] if lengths else None,
        iter_order=iter_order,
        prefix=prefix,
    )

    # Add the new region to the pool's region set
    result_pool.add_region(region)

    return result_pool
