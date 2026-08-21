"""Summary statistics describing a generated library."""

import warnings
from typing import Any

from .operation import Operation
from .types import Integral, Optional, Pool_type, Sequence, Union, beartype
from .utils.stats_utils import stats_from_seqs

# A pool declares how many sequences it can produce without generating any of
# them, so stats() would otherwise start an hours-long job on a large design.
# Reusing the operation-level limit keeps one such threshold in the package.
_AUTO_SEQ_LIMIT = Operation.max_num_sequential_states

# Above this many sequences the pairwise comparison is slow enough that the
# caller should hear about it first. 20,000 takes roughly 20 seconds.
_HAMMING_WARN_ABOVE = 20_000

# Matches the default chunk size of the export methods.
_CHUNK_SIZE = 1000

# How the report is laid out: (heading, [(label, key, formatter), ...]). A row
# whose key is absent from the statistics is skipped, and a section with no rows
# left is skipped with it.
_INT = "{:,}"
_FRAC = "{:.3f}"


def _percent(value: float) -> str:
    """Percentage with enough digits to see a rare feature but no more."""
    percent = value * 100
    return f"{percent:.1f}%" if percent == 0 or percent >= 1 else f"{percent:.2g}%"


_LAYOUT = [
    (
        "Composition",
        [
            ("design size (num_states)", "num_states", _INT),
            ("generated", "num_generated_seqs", _INT),
            ("filtered out", "num_filtered_out_seqs", _INT),
            ("unique sequences", "num_unique_seqs", _INT),
        ],
    ),
    ("Length", [("min / max", ("length_min", "length_max"), _INT)]),
    ("GC content", [("min / mean / max", ("gc_min", "gc_mean", "gc_max"), _FRAC)]),
    (
        "Homopolymer runs",
        [
            ("longest run", "longest_homopolymer", _INT),
            ("sequences with a long run", "frac_seqs_with_long_homopolymer", _percent),
        ],
    ),
    ("Repetitiveness (DUST)", [("mean / max", ("dust_mean", "dust_max"), "{:.2f}")]),
    (
        "Restriction sites",
        [("sequences containing one", "frac_seqs_with_restriction_site", _percent)],
    ),
    (
        "Pairwise distance (Hamming)",
        [
            (
                "min / mean / max",
                ("hamming_min", "hamming_mean", "hamming_max"),
                (_INT, "{:.1f}", _INT),
            )
        ],
    ),
]


def _format_value(keys, fmts, stats) -> Optional[str]:
    """Render one report row, or None when any of its keys is missing."""
    if isinstance(keys, str):
        keys = (keys,)
    if not isinstance(fmts, tuple):
        fmts = (fmts,) * len(keys)
    if any(stats.get(key) is None for key in keys):
        return None
    parts = [
        fmt(stats[key]) if callable(fmt) else fmt.format(stats[key]) for key, fmt in zip(keys, fmts)
    ]
    return " / ".join(parts)


def _format_report(stats) -> str:
    """Render the statistics as the plain-text report."""
    if stats.get("open_ended"):
        header = f"{stats['num_generated_seqs']:,} sequences drawn"
    elif stats.get("num_states") is None:
        header = f"{stats['num_generated_seqs']:,} sequences"
    else:
        header = (
            f"{stats['num_generated_seqs']:,} of {stats['num_states']:,} sequences in the design"
        )
    lines = [f"pool.stats()  -  {header}"]

    for heading, rows in _LAYOUT:
        rendered = [
            (label, value)
            for label, keys, fmt in rows
            if (value := _format_value(keys, fmt, stats)) is not None
        ]
        if heading == "Composition":
            rendered.append(
                (
                    "duplicate sequences",
                    f"{stats['num_duplicate_seqs']:,}   ({_percent(stats['frac_duplicate_seqs'])})",
                )
            )
            copies = stats["max_seq_copies"]
            rendered.append(
                ("most-repeated sequence", f"{copies:,} {'copy' if copies == 1 else 'copies'}")
            )
            if stats.get("open_ended"):
                rendered.insert(0, ("design size", "unbounded  (random sampling, no fixed size)"))
        if not rendered:
            continue
        lines.append("")
        lines.append(heading)
        lines += [f"  {label:<28} {value}" for label, value in rendered]

    if stats.get("hamming_exact") is False:
        lines.append(
            f"  sampled {stats['hamming_seqs_compared']:,} sequences; min is an upper "
            f"bound on the\n  true minimum, max a lower bound, mean unbiased"
        )
    if stats.get("open_ended"):
        lines.append("")
        lines.append(
            "  This design draws randomly without a fixed size, so the duplicate count\n"
            "  reflects how many sequences were drawn, not a property of the design."
        )
    return "\n".join(lines)


class _StatsDict(dict):
    """A dict of statistics that prints as a report.

    Private on purpose: nothing constructs one but :func:`stats`, and the
    documented return type is a plain ``dict``.
    """

    def __repr__(self) -> str:
        return _format_report(self)


@beartype
def stats(
    source: Union[Pool_type, Sequence[str]],
    num_seqs: Optional[Integral] = None,
    num_cycles: Optional[Integral] = None,
    seed: Optional[Integral] = None,
    max_hamming_seqs: Optional[Integral] = 2000,
    max_homopolymer_run: Optional[Integral] = 6,
    enzymes: Optional[list[str]] = None,
    sites: Optional[list[str]] = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Summarise the library a pool produces.

    Generates sequences and reports on them: how many are unique and how many
    are duplicates, how far apart they are, and how often they carry features
    that complicate synthesis. Nothing about the pool or its library is changed.

    A pool records how to build a library rather than the library itself, so a
    count is needed before anything can be measured. A pool whose design has a
    fixed size is measured in full when that size is at most
    ``Operation.max_num_sequential_states``; above that, and for a design that
    samples randomly without a fixed size, say how many sequences to look at.
    ``num_cycles=1`` always means "all of it".

    Parameters
    ----------
    source : Pool or sequence of str
        A ``DnaPool`` to generate from, or sequences to describe directly. With
        sequences, ``num_seqs`` and ``num_cycles`` do not apply.
    num_seqs : int, optional
        Generate exactly this many sequences. Mutually exclusive with
        ``num_cycles``.
    num_cycles : int, optional
        Generate this many complete passes through the state space. Mutually
        exclusive with ``num_seqs``.
    seed : int, optional
        Random seed. Fixes both the sequences a randomly-sampling pool produces
        and the subsample used for pairwise distances.
    max_hamming_seqs : int, optional
        Compare at most this many sequences pairwise. Comparing every pair costs
        time quadratic in the number of sequences, so larger libraries are
        subsampled and ``hamming_exact`` reports ``False``. ``None`` omits the
        distance keys.
    max_homopolymer_run : int, optional
        Longest single-base run to tolerate. Sequences with a longer run are
        counted in ``frac_seqs_with_long_homopolymer``. ``None`` omits that key.
        ``longest_homopolymer`` is reported either way.
    enzymes : list[str], optional
        Restriction enzyme names, or preset names such as ``'golden_gate'``.
    sites : list[str], optional
        Recognition sequences to look for, IUPAC codes allowed. Reverse
        complements are always checked as well. With neither ``enzymes`` nor
        ``sites``, ``frac_seqs_with_restriction_site`` is omitted.
    show_progress : bool, default True
        Show a progress bar while generating.

    Returns
    -------
    dict
        Statistics keyed by name, which print as a report. Sequence counts are
        of the tag-free, uppercased sequences; gap characters left by deletion
        operations are counted as characters. Keys that do not apply are absent
        rather than ``None``.

    Raises
    ------
    TypeError
        If ``source`` is a pool that is not a ``DnaPool``.
    ValueError
        If both ``num_seqs`` and ``num_cycles`` are given, if either is not
        positive, if neither is given for a design that has no fixed size or
        that declares more sequences than can be measured automatically, or if
        a count is given alongside sequences.

    Examples
    --------
    >>> print(pool.stats())
    >>> pool.stats(num_seqs=100_000)["frac_duplicate_seqs"]
    >>> pool.stats(enzymes=["golden_gate"])["frac_seqs_with_restriction_site"]
    """
    from .dna_pool import DnaPool
    from .pool import Pool
    from .pool_mixins.export_mixin import _strip_tags
    from .utils.seq_properties import get_sites_for_enzymes

    if num_seqs is not None and num_cycles is not None:
        raise ValueError("Specify only one of num_seqs or num_cycles, not both")
    for name, value in (("num_seqs", num_seqs), ("num_cycles", num_cycles)):
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if max_hamming_seqs is not None and int(max_hamming_seqs) < 2:
        raise ValueError(f"max_hamming_seqs must be at least 2, got {max_hamming_seqs}")
    if max_homopolymer_run is not None and int(max_homopolymer_run) < 1:
        raise ValueError(f"max_homopolymer_run must be at least 1, got {max_homopolymer_run}")

    resolved_sites = None
    if enzymes is not None or sites is not None:
        resolved_sites = get_sites_for_enzymes(enzymes=enzymes, sites=sites)

    if isinstance(source, Pool):
        if not isinstance(source, DnaPool):
            raise TypeError(
                f"stats() supports DnaPool; got {type(source).__name__}. "
                f"Statistics such as GC content and restriction sites are DNA-specific."
            )
        rows, num_states, open_ended = _generate_rows(
            source, num_seqs, num_cycles, seed, show_progress
        )
    else:
        if num_seqs is not None or num_cycles is not None:
            raise ValueError(
                "num_seqs and num_cycles apply to a pool; sequences are described as given"
            )
        rows, num_states, open_ended = list(source), None, False

    cleaned = [None if row is None else _strip_tags(row).upper() for row in rows]

    if max_hamming_seqs is not None:
        comparing = min(int(max_hamming_seqs), sum(row is not None for row in cleaned))
        if comparing > _HAMMING_WARN_ABOVE:
            warnings.warn(
                f"Comparing {comparing:,} sequences pairwise takes roughly "
                f"{comparing**2 / 2e7:.0f} s; lower max_hamming_seqs to speed this up, "
                f"or pass None to skip the distance statistics.",
                stacklevel=2,
            )
    return _StatsDict(
        stats_from_seqs(
            cleaned,
            num_states=num_states,
            open_ended=open_ended,
            max_homopolymer_run=None if max_homopolymer_run is None else int(max_homopolymer_run),
            sites=resolved_sites,
            max_hamming_seqs=None if max_hamming_seqs is None else int(max_hamming_seqs),
            seed=0 if seed is None else int(seed),
        )
    )


def _generate_rows(pool, num_seqs, num_cycles, seed, show_progress):
    """Generate the sequences to describe, and say what the design's size means.

    Returns the sequence strings (``None`` where a filter rejected one), the
    pool's ``num_states`` or ``None`` when that does not describe the design,
    and whether the design samples randomly without a fixed size.
    """
    # A pool that draws afresh for every row has no total number of sequences:
    # pool.num_states is a floor, so there is no "all of it" to measure.
    from .generate_library import _draws_fresh_sequences
    from .pool_mixins.export_mixin import _make_progress_bar

    open_ended = _draws_fresh_sequences(pool)
    num_states = None if open_ended else pool.num_states

    if num_seqs is None and num_cycles is None:
        if open_ended:
            raise ValueError(
                "This design samples randomly without a fixed size, so it has no total "
                "number of sequences. Say how many to examine, e.g. stats(num_seqs=10_000). "
                "To give the design a fixed size instead, pass num_states=... to the "
                "random operation."
            )
        if num_states > _AUTO_SEQ_LIMIT:
            raise ValueError(
                f"This design declares {num_states:,} sequences, above the "
                f"{_AUTO_SEQ_LIMIT:,} limit for an automatic report. Either examine part "
                f"of it, e.g. stats(num_seqs=100_000), or ask for all of it explicitly "
                f"with stats(num_cycles=1)."
            )
        num_cycles = 1

    total = int(num_seqs) if num_seqs is not None else int(num_cycles) * pool.num_states
    pbar = _make_progress_bar(total, "Generating sequences") if show_progress else None
    rows = []
    try:
        while len(rows) < total:
            # Keeping the null rows means one row is one state, so len(rows) is
            # also the next state to start from. Starting at 0 keeps repeated
            # calls on the same pool comparable.
            df = pool.generate_library(
                num_seqs=min(_CHUNK_SIZE, total - len(rows)),
                seed=seed,
                init_state=len(rows),
                discard_null_seqs=False,
            )
            rows += list(df["seq"])
            if pbar is not None:
                pbar.update(len(df))
    finally:
        if pbar is not None:
            pbar.close()
    return rows, num_states, open_ended
