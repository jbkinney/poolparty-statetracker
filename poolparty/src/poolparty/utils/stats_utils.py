"""Summary statistics for a collection of generated sequences.

The functions here take plain sequence strings and know nothing about Pools, so
they can also describe a library read back from a file. :func:`poolparty.stats`
generates a library from a Pool and delegates here.
"""

from collections import Counter

import numpy as np

from .seq_properties import (
    calc_dust,
    calc_gc,
    has_homopolymer,
    has_restriction_site,
    longest_homopolymer,
)

# Sequences are compared as whole blocks of bytes, so the pairwise distance
# matrix for one block is (block x block x length) bytes. 2000 keeps that under
# a few hundred MB for typical library sequences.
_HAMMING_BLOCK = 2000


def pairwise_hamming(seqs: list[str], max_seqs: int, seed: int) -> dict:
    """Minimum, mean and maximum Hamming distance over pairs of sequences.

    All pairs are compared when there are no more than ``max_seqs`` sequences.
    Above that a random subsample of ``max_seqs`` sequences is compared instead,
    and ``hamming_exact`` in the result is ``False``. A subsample estimates the
    mean well but biases the extremes: it sees only ``(max_seqs / n) ** 2`` of
    the pairs, so the reported minimum is an upper bound on the true minimum and
    the maximum a lower bound.

    Parameters
    ----------
    seqs : list[str]
        Sequences to compare. Must all be the same length.
    max_seqs : int
        Largest number of sequences to compare.
    seed : int
        Seed for choosing the subsample.

    Returns
    -------
    dict
        Keys ``hamming_exact``, ``hamming_seqs_compared``, ``hamming_min``,
        ``hamming_mean`` and ``hamming_max``.
    """
    exact = len(seqs) <= max_seqs
    if not exact:
        chosen = np.random.default_rng(seed).choice(len(seqs), size=max_seqs, replace=False)
        seqs = [seqs[i] for i in chosen]

    n, length = len(seqs), len(seqs[0])
    encoded = np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(n, length)

    lowest, highest, total, num_pairs = length, 0, 0, 0
    for i in range(0, n, _HAMMING_BLOCK):
        left = encoded[i : i + _HAMMING_BLOCK]
        for j in range(i, n, _HAMMING_BLOCK):
            right = encoded[j : j + _HAMMING_BLOCK]
            block = (left[:, None, :] != right[None, :, :]).sum(axis=2)
            if i == j:
                # Compare each pair within a block once, and never to itself.
                distances = block[np.triu_indices(block.shape[0], k=1)]
            else:
                distances = block.ravel()
            if distances.size == 0:
                continue
            lowest = min(lowest, int(distances.min()))
            highest = max(highest, int(distances.max()))
            total += int(distances.sum())
            num_pairs += distances.size

    return {
        "hamming_exact": exact,
        "hamming_seqs_compared": n,
        "hamming_min": lowest,
        "hamming_mean": total / num_pairs,
        "hamming_max": highest,
    }


def stats_from_seqs(
    seqs: list,
    num_states: int | None = None,
    open_ended: bool = False,
    max_homopolymer_run: int | None = 6,
    sites: list[str] | None = None,
    max_hamming_seqs: int | None = 2000,
    seed: int = 0,
) -> dict:
    """Summarise a collection of generated sequences.

    Parameters
    ----------
    seqs : list
        Sequence strings, with ``None`` for sequences a filter rejected. Region
        tags must already be stripped and the strings uppercased.
    num_states : int, optional
        ``num_states`` of the pool the sequences came from, or ``None`` when
        that number does not describe the design (see ``open_ended``) or the
        sequences did not come from a pool.
    open_ended : bool, default False
        True when the design samples randomly without a fixed size, so it has no
        total number of sequences.
    max_homopolymer_run : int, optional
        Sequences with a single-base run longer than this are counted in
        ``frac_seqs_with_long_homopolymer``. ``None`` omits that key.
    sites : list[str], optional
        Recognition sequences to look for. ``None`` omits
        ``frac_seqs_with_restriction_site``.
    max_hamming_seqs : int, optional
        Largest number of sequences to compare pairwise. ``None`` omits the
        distance keys.
    seed : int, default 0
        Seed for choosing the pairwise subsample.

    Returns
    -------
    dict
        The composition keys are always present. The per-sequence keys are
        omitted when no sequence survived, and the distance keys are also
        omitted when the sequences differ in length.
    """
    valid = [seq for seq in seqs if seq is not None]
    counts = Counter(valid)

    result = {
        "num_states": num_states,
        "open_ended": open_ended,
        "num_generated_seqs": len(seqs),
        "frac_design_covered": None if num_states is None else len(seqs) / num_states,
        "num_filtered_out_seqs": len(seqs) - len(valid),
        "num_valid_seqs": len(valid),
        "num_unique_seqs": len(counts),
        "num_duplicate_seqs": len(valid) - len(counts),
        "frac_duplicate_seqs": (len(valid) - len(counts)) / len(valid) if valid else 0.0,
        "max_seq_copies": max(counts.values()) if counts else 0,
    }
    if not valid:
        return result

    lengths = [len(seq) for seq in valid]
    gc = [calc_gc(seq) for seq in valid]
    dust = [calc_dust(seq) for seq in valid]
    result.update(
        length_min=min(lengths),
        length_max=max(lengths),
        gc_min=min(gc),
        gc_mean=sum(gc) / len(gc),
        gc_max=max(gc),
        longest_homopolymer=max(longest_homopolymer(seq) for seq in valid),
        dust_mean=sum(dust) / len(dust),
        dust_max=max(dust),
    )
    if max_homopolymer_run is not None:
        long_runs = sum(has_homopolymer(seq, max_homopolymer_run) for seq in valid)
        result["frac_seqs_with_long_homopolymer"] = long_runs / len(valid)
    if sites is not None:
        with_site = sum(has_restriction_site(seq, sites) for seq in valid)
        result["frac_seqs_with_restriction_site"] = with_site / len(valid)

    # Hamming distance is only defined between sequences of equal length.
    if max_hamming_seqs is not None and len(valid) > 1 and min(lengths) == max(lengths):
        result.update(pairwise_hamming(valid, max_hamming_seqs, seed))
    return result
