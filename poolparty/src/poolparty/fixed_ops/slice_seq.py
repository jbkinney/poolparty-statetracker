"""SliceSeq operation - slice SEQUENCES (string slicing)."""

from numbers import Integral, Real

from ..pool import Pool
from ..types import Optional, Pool_type, RegionType, Sequence, Union, beartype


@beartype
def slice_seq(
    pool: Union[Pool_type, str],
    region: RegionType = None,
    start: Optional[Integral] = None,
    stop: Optional[Integral] = None,
    step: Optional[Integral] = None,
    keep_context: bool = False,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
    style: Optional[str] = None,
) -> Pool:
    """
    Create a Pool containing sliced sequences from the input pool.

    Extracts a subsequence based on region and/or Python-style slice parameters.

    Parameters
    ----------
    pool : Union[Pool_type, str]
        The Pool (or sequence string) whose sequences will be sliced.
    region : RegionType, default=None
        Region to slice from. Can be:
        - str: Name of an annotated region (e.g., 'orf')
        - Sequence[int]: [start, stop] interval in the sequence
        - None: Use the full sequence
        If only region is specified (no start/stop/step), returns just that region.
    start : Optional[Integral], default=None
        Start position for slicing (0-indexed, Python-style).
        Applied after region extraction if region is specified.
    stop : Optional[Integral], default=None
        Stop position for slicing (exclusive, Python-style).
        Applied after region extraction if region is specified.
    step : Optional[Integral], default=None
        Step for slicing (Python-style).
        Applied after region extraction if region is specified.
    keep_context : bool, default=False
        If True, reassemble the sliced content back into the original sequence
        context (prefix + sliced_content + suffix). If False (default), return
        only the sliced content.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    prefix : Optional[str], default=None
        Prefix for sequence naming.
    style : Optional[str], default=None
        Style to apply to the resulting sliced sequences (e.g., 'red', 'blue bold').

    Returns
    -------
    Pool
        A Pool containing sliced sequences.

    Examples
    --------
    >>> with pp.Party():
    ...     # Slice positions 2-6 from the full sequence
    ...     pool = pp.from_seq('ACGTACGT')
    ...     sliced = pp.slice_seq(pool, start=2, stop=6)
    ...     # Result: 'GTAC'
    ...
    ...     # Extract just a named region
    ...     pool = pp.from_seq('AAA<orf>ATGCCC</orf>TTT')
    ...     orf = pp.slice_seq(pool, region='orf')
    ...     # Result: 'ATGCCC'
    ...
    ...     # Slice within a named region
    ...     pool = pp.from_seq('AAA<orf>ATGCCC</orf>TTT')
    ...     sliced = pp.slice_seq(pool, region='orf', start=0, stop=3)
    ...     # Result: 'ATG'
    ...
    ...     # Slice with step (every other character)
    ...     pool = pp.from_seq('ABCDEFGH')
    ...     sliced = pp.slice_seq(pool, step=2)
    ...     # Result: 'ACEG'
    ...
    ...     # Use as a method on Pool objects
    ...     pool = pp.from_seq('ACGTACGT')
    ...     sliced = pool.slice_seq(start=0, stop=4)
    ...     # Result: 'ACGT'
    ...
    ...     # Keep context - reassemble into original sequence
    ...     pool = pp.from_seq('AAA<orf>ATGCCC</orf>TTT')
    ...     sliced = pp.slice_seq(pool, region='orf', start=0, stop=3, keep_context=True)
    ...     # Result: 'AAAATGTTT' (prefix + sliced region + suffix)
    """
    from ..utils.parsing_utils import strip_all_tags, validate_single_region
    from .fixed import fixed_operation

    # Check if we need to apply a slice (any slice parameter is specified)
    has_slice = start is not None or stop is not None or step is not None

    # Build a slice object from the parameters
    slice_start = int(start) if start is not None else None
    slice_stop = int(stop) if stop is not None else None
    slice_step = int(step) if step is not None else None
    key = slice(slice_start, slice_stop, slice_step)

    # keep_context only makes sense when region is specified
    if keep_context and region is None:
        raise ValueError("keep_context=True requires a region to be specified")

    if region is not None:
        # Region specified - we need to extract the region content and optionally slice it
        if isinstance(region, str):
            # Named region - extract content from XML tags
            def seq_from_seqs_fn(seqs: list[str]) -> str:
                seq = seqs[0]
                parsed_region = validate_single_region(seq, region)
                content = strip_all_tags(parsed_region.content)
                if has_slice:
                    sliced_content = content[key]
                else:
                    sliced_content = content
                if keep_context:
                    # Reassemble: prefix + sliced_content + suffix
                    prefix_str = strip_all_tags(seq[: parsed_region.start])
                    suffix_str = strip_all_tags(seq[parsed_region.end :])
                    return prefix_str + sliced_content + suffix_str
                return sliced_content

            def seq_length_from_pool_lengths_fn(lengths: Sequence[Optional[int]]) -> Optional[int]:
                # Get the registered region's length from Party
                from ..party import get_active_party

                party = get_active_party()
                if party.has_region(region):
                    registered_region = party.get_region_by_name(region)
                    region_len = registered_region.seq_length
                else:
                    region_len = None

                if keep_context:
                    parent_len = lengths[0]
                    if parent_len is None or region_len is None:
                        return None
                    if not has_slice:
                        return parent_len
                    s_start, s_stop, s_step = key.indices(region_len)
                    sliced_len = max(
                        0,
                        (s_stop - s_start + (s_step - 1 if s_step > 0 else s_step + 1)) // s_step,
                    )
                    return parent_len - region_len + sliced_len

                if region_len is None:
                    return None
                if not has_slice:
                    return region_len
                # Apply slice to region length
                s_start, s_stop, s_step = key.indices(region_len)
                return max(
                    0, (s_stop - s_start + (s_step - 1 if s_step > 0 else s_step + 1)) // s_step
                )
        else:
            # Interval region [start, stop]
            region_start, region_stop = int(region[0]), int(region[1])

            def seq_from_seqs_fn(seqs: list[str]) -> str:
                seq = strip_all_tags(seqs[0])
                if region_stop > len(seq):
                    raise ValueError(
                        f"slice_seq region [{region_start}, {region_stop}] exceeds "
                        f"sequence length {len(seq)}"
                    )
                content = seq[region_start:region_stop]
                if has_slice:
                    sliced_content = content[key]
                else:
                    sliced_content = content
                if keep_context:
                    # Reassemble: prefix + sliced_content + suffix
                    prefix_str = seq[:region_start]
                    suffix_str = seq[region_stop:]
                    return prefix_str + sliced_content + suffix_str
                return sliced_content

            def seq_length_from_pool_lengths_fn(lengths: Sequence[Optional[int]]) -> Optional[int]:
                parent_len = lengths[0]
                region_len = region_stop - region_start

                if parent_len is not None and region_stop > parent_len:
                    raise ValueError(
                        f"slice_seq region [{region_start}, {region_stop}] exceeds "
                        f"sequence length {parent_len}"
                    )

                if keep_context:
                    # Length depends on parent length and slice
                    if parent_len is None:
                        return None
                    # prefix_len + sliced_len + suffix_len
                    prefix_len = region_start
                    suffix_len = parent_len - region_stop
                    if not has_slice:
                        return prefix_len + region_len + suffix_len
                    s_start, s_stop, s_step = key.indices(region_len)
                    sliced_len = max(
                        0, (s_stop - s_start + (s_step - 1 if s_step > 0 else s_step + 1)) // s_step
                    )
                    return prefix_len + sliced_len + suffix_len

                if not has_slice:
                    return region_len
                # Apply slice to region length
                s_start, s_stop, s_step = key.indices(region_len)
                return max(
                    0, (s_stop - s_start + (s_step - 1 if s_step > 0 else s_step + 1)) // s_step
                )
    else:
        # No region - slice the full sequence
        def seq_from_seqs_fn(seqs: list[str]) -> str:
            return seqs[0][key]

        def seq_length_from_pool_lengths_fn(lengths: Sequence[Optional[int]]) -> Optional[int]:
            parent_len = lengths[0]
            if parent_len is None:
                return None
            # Compute length from slice params
            s_start, s_stop, s_step = key.indices(parent_len)
            return max(0, (s_stop - s_start + (s_step - 1 if s_step > 0 else s_step + 1)) // s_step)

    result_pool = fixed_operation(
        parent_pools=[pool],
        seq_from_seqs_fn=seq_from_seqs_fn,
        seq_length_from_pool_lengths_fn=seq_length_from_pool_lengths_fn,
        iter_order=iter_order,
        prefix=prefix,
        _factory_name="slice_seq",
    )

    # Apply style if specified
    if style is not None:
        from .stylize import stylize

        result_pool = stylize(result_pool, style=style)

    return result_pool
