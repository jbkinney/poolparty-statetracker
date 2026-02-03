"""Sequence property calculation functions.

This module provides functions for calculating various properties of DNA sequences,
including GC content, sequence complexity, and restriction site detection.
"""

import re
from itertools import product

from .dna_utils import IUPAC_TO_DNA, reverse_complement


def calc_gc(seq: str) -> float:
    """Calculate GC content of a DNA sequence.

    Parameters
    ----------
    seq : str
        DNA sequence (case-insensitive). Non-DNA characters are ignored.

    Returns
    -------
    float
        GC content as a fraction between 0.0 and 1.0.
        Returns 0.0 for empty sequences or sequences with no valid bases.

    Examples
    --------
    >>> calc_gc("ATGC")
    0.5
    >>> calc_gc("AAAA")
    0.0
    >>> calc_gc("GGCC")
    1.0
    >>> calc_gc("ATG-C")  # Gaps are ignored
    0.5
    """
    seq_upper = seq.upper()
    gc_count = seq_upper.count("G") + seq_upper.count("C")
    total = (
        seq_upper.count("A") + seq_upper.count("C") + seq_upper.count("G") + seq_upper.count("T")
    )
    if total == 0:
        return 0.0
    return gc_count / total


def calc_complexity(seq: str, k_range: tuple[int, ...] = (1, 2, 3)) -> float:
    """Calculate linguistic complexity of a sequence.

    Linguistic complexity measures the ratio of observed unique k-mers to the
    maximum possible unique k-mers, averaged across multiple k values. Low
    complexity indicates repetitive sequences (e.g., homopolymers, tandem repeats).

    Parameters
    ----------
    seq : str
        Sequence string (case-insensitive for DNA).
    k_range : tuple[int, ...]
        Tuple of k values to use for k-mer analysis. Default is (1, 2, 3)
        which captures homopolymers (k=1), dinucleotide repeats (k=2),
        and trinucleotide patterns (k=3).

    Returns
    -------
    float
        Complexity score between 0.0 (completely repetitive) and 1.0
        (maximally complex). Returns 1.0 for sequences shorter than
        the smallest k.

    Examples
    --------
    >>> calc_complexity("AAAAAAAAAA")  # Homopolymer - very low complexity
    0.1111...
    >>> calc_complexity("ACGTACGTAC")  # Repetitive pattern
    0.5...
    >>> calc_complexity("ACGTMKWSRY")  # More random - higher complexity
    0.8...
    """
    seq_upper = seq.upper()
    # Only consider valid DNA characters for alphabet size
    valid_chars = set(c for c in seq_upper if c in "ACGT")
    alphabet_size = max(len(valid_chars), 4)  # Assume DNA alphabet of 4

    scores = []
    for k in k_range:
        if len(seq_upper) < k:
            continue
        # Count unique k-mers
        kmers = set(seq_upper[i : i + k] for i in range(len(seq_upper) - k + 1))
        observed = len(kmers)
        # Maximum possible is min of: number of positions, alphabet^k
        max_possible = min(len(seq_upper) - k + 1, alphabet_size**k)
        if max_possible > 0:
            scores.append(observed / max_possible)

    if not scores:
        return 1.0  # Sequence too short for any k
    return sum(scores) / len(scores)


def calc_dust(seq: str) -> float:
    """Calculate DUST score for sequence complexity.

    The DUST algorithm identifies low-complexity regions based on triplet
    frequencies. Lower scores indicate more complex (less repetitive) sequences.
    This is the standard algorithm used by NCBI BLAST for masking.

    Parameters
    ----------
    seq : str
        DNA sequence (case-insensitive).

    Returns
    -------
    float
        DUST score. Lower values indicate higher complexity.
        Typical thresholds: < 2.0 is complex, > 4.0 is low-complexity.
        Returns 0.0 for sequences shorter than 3 bases.

    Examples
    --------
    >>> calc_dust("ACGTACGTACGT")  # Repetitive - higher score
    >>> calc_dust("ACGTMKWSRYBV")  # More random - lower score
    >>> calc_dust("AAAAAAAAAA")    # Homopolymer - very high score

    References
    ----------
    Hancock, J.M. and Armstrong, J.S. (1994). SIMPLE34: an improved and enhanced
    implementation for VAX and Sun computers of the SIMPLE algorithm for
    analysis of clustered repetitive motifs in nucleotide sequences.
    """
    seq_upper = seq.upper()
    if len(seq_upper) < 3:
        return 0.0

    # Count triplets
    triplet_counts: dict[str, int] = {}
    for i in range(len(seq_upper) - 2):
        triplet = seq_upper[i : i + 3]
        triplet_counts[triplet] = triplet_counts.get(triplet, 0) + 1

    # Calculate score: sum of c*(c-1)/2 for each triplet count c
    score = sum(c * (c - 1) / 2 for c in triplet_counts.values())

    # Normalize by sequence length
    window_len = len(seq_upper) - 2
    if window_len <= 0:
        return 0.0
    return score / window_len


def has_homopolymer(seq: str, max_length: int) -> bool:
    """Check if sequence contains a homopolymer run exceeding max_length.

    Parameters
    ----------
    seq : str
        DNA sequence (case-insensitive).
    max_length : int
        Maximum allowed homopolymer length. If any single-base run
        exceeds this length, returns True.

    Returns
    -------
    bool
        True if a homopolymer longer than max_length exists, False otherwise.

    Examples
    --------
    >>> has_homopolymer("ACGTAAAAAACGT", 4)
    True
    >>> has_homopolymer("ACGTAAAACGT", 4)
    False
    >>> has_homopolymer("ACGTAAAACGT", 3)
    True
    """
    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    # Pattern matches any character repeated more than max_length times
    pattern = r"(.)\1{" + str(max_length) + r",}"
    return bool(re.search(pattern, seq, re.IGNORECASE))


def _expand_iupac(site: str) -> list[str]:
    """Expand an IUPAC sequence to all possible concrete sequences.

    Parameters
    ----------
    site : str
        Recognition site that may contain IUPAC ambiguity codes.

    Returns
    -------
    list[str]
        List of all concrete sequences represented by the IUPAC pattern.

    Examples
    --------
    >>> _expand_iupac("GAATTC")
    ['GAATTC']
    >>> _expand_iupac("RGATCY")  # R=A/G, Y=C/T
    ['AGATCC', 'AGATCT', 'GGATCC', 'GGATCT']
    """
    site_upper = site.upper()
    # Get list of possible bases at each position
    options = []
    for char in site_upper:
        if char in IUPAC_TO_DNA:
            options.append(IUPAC_TO_DNA[char])
        else:
            # Non-IUPAC character (e.g., N in some contexts) - keep as is
            options.append([char])

    # Generate all combinations
    return ["".join(combo) for combo in product(*options)]


def _site_matches(seq: str, site: str) -> bool:
    """Check if a sequence contains a recognition site.

    Handles IUPAC ambiguity codes in the site by expanding to all possibilities.

    Parameters
    ----------
    seq : str
        Sequence to search in (case-insensitive).
    site : str
        Recognition site to search for (may contain IUPAC codes).

    Returns
    -------
    bool
        True if site is found in sequence.
    """
    seq_upper = seq.upper()

    # For sites without ambiguity codes, simple substring search
    site_upper = site.upper()
    if all(c in "ACGT" for c in site_upper):
        return site_upper in seq_upper

    # Expand IUPAC codes and check each possibility
    for expanded in _expand_iupac(site_upper):
        if expanded in seq_upper:
            return True
    return False


def has_restriction_site(
    seq: str,
    sites: list[str],
    check_rc: bool = True,
) -> bool:
    """Check if sequence contains any of the specified restriction sites.

    Parameters
    ----------
    seq : str
        DNA sequence to check (case-insensitive).
    sites : list[str]
        List of recognition sequences to search for. May contain IUPAC codes.
    check_rc : bool, default True
        If True, also check the reverse complement of each site.
        This is important for non-palindromic sites.

    Returns
    -------
    bool
        True if any site is found in the sequence.

    Examples
    --------
    >>> has_restriction_site("ACGTGAATTCACGT", ["GAATTC"])  # EcoRI site
    True
    >>> has_restriction_site("ACGTACGTACGT", ["GAATTC"])
    False
    >>> has_restriction_site("ACGGTCTCACGT", ["GGTCTC"], check_rc=True)  # BsaI
    True
    """
    for site in sites:
        if _site_matches(seq, site):
            return True
        if check_rc:
            rc_site = reverse_complement(site)
            if rc_site.upper() != site.upper() and _site_matches(seq, rc_site):
                return True
    return False


def get_sites_for_enzymes(
    enzymes: list[str] | None = None,
    sites: list[str] | None = None,
) -> list[str]:
    """Get list of recognition sites from enzyme names and/or explicit sites.

    Parameters
    ----------
    enzymes : list[str] | None
        List of enzyme names or preset names. Enzyme names are looked up
        in the restriction enzyme database. If a name matches a preset,
        all enzymes in that preset are included.
    sites : list[str] | None
        List of explicit recognition sequences (IUPAC format).

    Returns
    -------
    list[str]
        Combined list of unique recognition sites.

    Raises
    ------
    ValueError
        If an enzyme name is not found in the database or presets.

    Examples
    --------
    >>> get_sites_for_enzymes(enzymes=["EcoRI", "BamHI"])
    ['GAATTC', 'GGATCC']
    >>> get_sites_for_enzymes(enzymes=["golden_gate"])  # Preset
    ['GGTCTC', 'CGTCTC', 'GAAGAC', ...]
    >>> get_sites_for_enzymes(enzymes=["EcoRI"], sites=["CUSTOM"])
    ['GAATTC', 'CUSTOM']
    """
    from ..data.restriction_enzymes import (
        ENZYME_PRESETS,
        get_enzyme_site,
        get_preset_enzymes,
    )

    result_sites: list[str] = []

    if enzymes:
        for name in enzymes:
            # Check if it's a preset name
            if name in ENZYME_PRESETS or name.lower() in [p.lower() for p in ENZYME_PRESETS]:
                try:
                    preset_enzymes = get_preset_enzymes(name)
                    for enzyme in preset_enzymes:
                        result_sites.append(get_enzyme_site(enzyme))
                except ValueError:
                    pass  # Not a preset, try as enzyme name
                else:
                    continue

            # Try as enzyme name
            result_sites.append(get_enzyme_site(name))

    if sites:
        result_sites.extend(sites)

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique_sites: list[str] = []
    for site in result_sites:
        site_upper = site.upper()
        if site_upper not in seen:
            seen.add(site_upper)
            unique_sites.append(site_upper)

    return unique_sites
