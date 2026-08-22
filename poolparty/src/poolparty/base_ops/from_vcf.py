"""FromVcf operation - create a pool of windows around variants in a VCF file."""

import gzip
import warnings
from numbers import Real
from urllib.parse import unquote

import numpy as np
from pyfaidx import Fasta

from ..dna_pool import DnaPool
from ..operation import Operation
from ..types import CardsType, Literal, Optional, Seq, Sequence, beartype
from ..utils import dna_utils
from ..utils.dna_seq import DnaSeq

# Fields required of every data line: CHROM POS ID REF ALT QUAL FILTER INFO.
_MIN_FIELDS = 8

# Fraction of compared references above which the two files are taken to be
# incompatible. Below it, records are rejected individually; above it, the whole
# call fails, because a library built from partly-matching references may have
# every window displaced. Applied at any file size.
_MISMATCH_LIMIT = 0.2

_CARD_KEYS = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "allele",
    "variant_type",
    "variant_id",
    "filter",
    "window_start",
    "window_stop",
)

# Card keys describing the alternate allele. A reference window is shared by every
# alternate allele at its site, so none of these belongs to it.
_ALT_ONLY_KEYS = ("alt", "variant_type", "variant_id", "filter")


def _open_vcf(vcf_path: str):
    """Open a plain or gzipped VCF for line iteration.

    bgzipped VCFs are valid gzip streams, so ``gzip`` reads them sequentially.
    Indexed access is not supported; every record is read. ``utf-8-sig`` drops a
    leading byte-order mark, which Windows tools write ahead of ``##fileformat``.
    """
    if vcf_path.endswith(".gz"):
        return gzip.open(vcf_path, "rt", encoding="utf-8-sig")
    return open(vcf_path, encoding="utf-8-sig")


def _normalise_chrom(chrom: str, fasta: Fasta) -> Optional[str]:
    """Return the contig name as it appears in the FASTA, or None if absent.

    Both spellings of the ``chr`` prefix are tried, as are ``chrM`` and ``MT``.
    """
    if chrom in fasta:
        return chrom
    candidates = [chrom[3:] if chrom.startswith("chr") else "chr" + chrom]
    if chrom in ("chrM", "M"):
        candidates += ["MT", "chrMT"]
    elif chrom in ("MT", "chrMT"):
        candidates += ["chrM", "M"]
    return next((c for c in candidates if c in fasta), None)


def _variant_type(ref: str, alt: str) -> str:
    """Classify a variant by comparing allele lengths.

    ``'snv'`` is one base for one base, ``'substitution'`` is equal numbers of
    bases with more than one, and unequal lengths give ``'insertion'`` or
    ``'deletion'``.
    """
    if len(ref) == len(alt):
        return "snv" if len(ref) == 1 else "substitution"
    return "insertion" if len(alt) > len(ref) else "deletion"


def _parse_info(info: str, keys: Sequence[str]) -> dict:
    """Extract the requested INFO keys, percent-decoded.

    A key absent from the record maps to None; a flag key maps to ``''``.
    """
    if not keys:
        return {}
    found = {}
    if info not in (".", ""):
        for field in info.split(";"):
            key, _, value = field.partition("=")
            if key in keys:
                found[key] = unquote(value)
    return {f"info_{k}": found.get(k) for k in keys}


@beartype
def from_vcf(
    vcf_path: str,
    fasta_path: str,
    flank_left: int,
    flank_right: int,
    *,
    alleles: Literal["ref", "alt", "both"] = "both",
    variant_types: Optional[Sequence[str]] = None,
    max_allele_length: Optional[int] = 100,
    info_fields: Optional[Sequence[str]] = None,
    prefix: Optional[str] = None,
    style: Optional[str] = None,
    cards: CardsType = None,
    iter_order: Optional[Real] = None,
) -> DnaPool:
    """
    Create a Pool of reference-genome windows around each variant in a VCF file.

    Each variant contributes a window carrying its alternate allele, and each
    distinct site contributes one window carrying the reference allele. Windows are
    on the reference plus strand and are uppercased. Sequence length is
    ``flank_left + len(allele) + flank_right``, so a pool containing indels has no
    defined ``seq_length``.

    Parameters
    ----------
    vcf_path : str
        Path to a VCF file. ``.gz`` files are read as gzip, which covers bgzip.
        BCF and indexed access are not supported.
    fasta_path : str
        Path to the reference FASTA (indexed with pyfaidx).
    flank_left : int
        Bases of reference sequence before the variant. Must be >= 0.
    flank_right : int
        Bases of reference sequence after the variant. Must be >= 0.
    alleles : {'ref', 'alt', 'both'}, default='both'
        Which windows to emit. ``'ref'`` emits one window per distinct site.
    variant_types : Optional[Sequence[str]], default=None
        Keep only these types: ``'snv'``, ``'substitution'``, ``'insertion'``,
        ``'deletion'``. Records with no surviving allele are dropped entirely.
        ``['snv']`` gives a pool with a defined ``seq_length``.
    max_allele_length : Optional[int], default=100
        Skip records whose REF or ALT exceeds this many bases. ``None`` disables
        the check.
    info_fields : Optional[Sequence[str]], default=None
        INFO keys to expose as design card keys, prefixed with ``info_``, so
        ``['AF']`` gives the key ``'info_AF'``. Values are taken verbatim and are
        not split per allele.
    prefix : Optional[str], default=None
        Prefix for generated sequence names.
    style : Optional[str], default=None
        Style to apply to output sequences (e.g., 'red', 'blue bold').
    cards : list[str] or dict, optional
        Design card keys to include. Available keys: ``'chrom'``, ``'pos'``,
        ``'ref'``, ``'alt'``, ``'allele'``, ``'variant_type'``, ``'variant_id'``,
        ``'filter'``, ``'window_start'``, ``'window_stop'``, plus any
        ``info_``-prefixed keys named in ``info_fields``. On a reference row
        ``'alt'``, ``'variant_type'``, ``'variant_id'``, ``'filter'`` and every
        ``info_`` key are ``None``.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    DnaPool
        A Pool yielding one window per emitted allele, traversed sequentially.
        The whole VCF is read at construction time. ``pool.operation.skipped``
        counts the records that were skipped, by reason.

    Raises
    ------
    ValueError
        If a flank is negative, ``max_allele_length`` is less than 1, or
        ``variant_types`` is empty or names an unknown type.
    ValueError
        If a data line has fewer than eight tab-separated fields, its POS is not
        an integer, no record survives parsing, or more than 20% of compared
        references disagree with the FASTA.

    Examples
    --------
    >>> pool = pp.from_vcf("variants.vcf", "hg38.fa", 500, 500)
    >>> pool = pp.from_vcf("variants.vcf.gz", "hg38.fa", 100, 100, alleles="alt",
    ...                    variant_types=["snv"], cards=["chrom", "pos"])
    """
    if flank_left < 0 or flank_right < 0:
        raise ValueError(
            f"flanks must be >= 0, got flank_left={flank_left}, flank_right={flank_right}"
        )
    for name, value in (("variant_types", variant_types), ("info_fields", info_fields)):
        if isinstance(value, str):
            raise ValueError(f"{name} must be a list of strings, not a bare string: {value!r}")
    if max_allele_length is not None and max_allele_length < 1:
        raise ValueError(f"max_allele_length must be >= 1 or None, got {max_allele_length}")
    if variant_types is not None:
        if not variant_types:
            raise ValueError("variant_types must name at least one type, or be None")
        unknown = set(variant_types) - {"snv", "substitution", "insertion", "deletion"}
        if unknown:
            raise ValueError(
                f"Unknown variant_types {sorted(unknown)}. "
                "Valid: 'snv', 'substitution', 'insertion', 'deletion'."
            )

    info_keys = list(info_fields) if info_fields else []
    fasta = Fasta(fasta_path)
    try:
        rows, skipped, n_compared = _read_windows(
            vcf_path,
            fasta,
            flank_left,
            flank_right,
            alleles,
            variant_types,
            max_allele_length,
            info_keys,
        )
    finally:
        fasta.close()

    if n_compared and skipped["ref_mismatch"] / n_compared > _MISMATCH_LIMIT:
        raise ValueError(
            f"{skipped['ref_mismatch']} of {n_compared} records in {vcf_path} disagree "
            f"with {fasta_path} on the reference allele. Check that the two files use "
            "compatible reference sequences, assemblies and coordinates."
        )

    # variant_type exclusions are what the caller asked for, not records we could
    # not represent, so they are counted but not warned about.
    reported = {r: n for r, n in skipped.items() if n and r != "variant_type"}
    if reported:
        # The path is in the message so that warnings from different files are not
        # collapsed into one by the default duplicate filter.
        warnings.warn(
            f"from_vcf rejected input in {vcf_path}: "
            + ", ".join(f"{n} {reason.replace('_', ' ')}" for reason, n in reported.items()),
            stacklevel=3,
        )

    if not rows:
        raise ValueError(
            f"No usable records in {vcf_path}. "
            + (
                f"Skipped: { ({r: n for r, n in skipped.items() if n}) }."
                if any(skipped.values())
                else "The file contains no data lines."
            )
        )

    op = FromVcfOp(
        rows,
        info_fields=info_keys,
        skipped=skipped,
        prefix=prefix,
        style=style,
        cards=cards,
        iter_order=iter_order,
    )
    return DnaPool(operation=op)


def _read_windows(
    vcf_path: str,
    fasta: Fasta,
    flank_left: int,
    flank_right: int,
    alleles: str,
    variant_types: Optional[Sequence[str]],
    max_allele_length: Optional[int],
    info_keys: list[str],
) -> tuple[list[dict], dict[str, int], int]:
    """Read a VCF and cut one window per emitted allele, in a single pass.

    The order of the checks below is load-bearing, so it is written out rather
    than left to be re-derived. Each row states what the failure means, what it
    removes, and why it cannot move later:

    =========================== ================== =====================================
    Failure                     Removes            Must precede
    =========================== ================== =====================================
    contig absent from FASTA    the record         every slice; there is nothing to cut
    REF empty or not ACGT       the record         the REF comparison: an ``N`` in an
                                                   assembly gap equals the FASTA's
                                                   ``N``, so comparing first would
                                                   report a wrong genome build
    REF over the length cap     the record         the window bounds - ``len(REF)`` sets
                                                   the window's width
    window off the contig       the record         every slice
    gap anywhere in the window  the record         the REF comparison, for the same
                                                   reason as the REF check but on the
                                                   FASTA side: an ``N`` at the variant
                                                   position is a gap, not a mismatch
    REF disagrees with FASTA    the record         - (last record-level check)
    ALT empty or not ACGT       that allele        -
    ALT over the length cap     that allele        -
    ALT type not requested      that allele        -
    =========================== ================== =====================================

    A record-level failure leaves the site with no window at all. An allele-level
    failure drops only that allele and keeps the reference window, except under
    ``variant_types``: once the caller has filtered by type, a site with no
    surviving allele is dropped whatever made its alleles unusable.

    Returns the rows, a count of rejections by reason, and the number of records
    whose REF was compared against the FASTA, which is the denominator for the
    mismatch rate.

    The counts are not in one unit. Record-level reasons -- ``contig_absent``,
    ``non_dna_ref``, ``off_contig``, ``gap_in_window``, ``ref_mismatch`` -- count
    rejected records. ALT-level reasons -- ``non_dna_alt``, ``variant_type`` --
    count rejected alternate alleles. ``allele_too_long`` counts both: an
    over-long REF rejects the record, an over-long ALT rejects that allele.
    """
    rows: list[dict] = []
    seen_sites: set[tuple] = set()
    n_compared = 0
    skipped = dict.fromkeys(
        (
            "contig_absent",
            "non_dna_ref",
            "off_contig",
            "gap_in_window",
            "ref_mismatch",
            "non_dna_alt",
            "allele_too_long",
            "variant_type",
        ),
        0,
    )
    # The allele loop is needed only to decide what to emit or to apply a type
    # filter. Without either, it would count skips for alleles this mode never
    # emits.
    inspect_alts = alleles != "ref" or variant_types is not None

    with _open_vcf(vcf_path) as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < _MIN_FIELDS:
                raise ValueError(
                    f"{vcf_path} line {line_no}: expected at least {_MIN_FIELDS} "
                    f"tab-separated fields, found {len(fields)}: {line[:60]!r}"
                )
            chrom, pos_str, var_id, ref, alt_field, _qual, filt, info = fields[:_MIN_FIELDS]

            try:
                pos = int(pos_str)
            except ValueError:
                raise ValueError(
                    f"{vcf_path} line {line_no}: POS is not an integer: {pos_str!r}"
                ) from None

            contig = _normalise_chrom(chrom, fasta)
            if contig is None:
                skipped["contig_absent"] += 1
                continue
            if not ref or set(ref) - dna_utils.VALID_CHARS:
                skipped["non_dna_ref"] += 1
                continue
            if max_allele_length is not None and len(ref) > max_allele_length:
                skipped["allele_too_long"] += 1
                continue

            start = pos - 1 - flank_left  # VCF POS is 1-based; slices are 0-based
            ref_end = pos - 1 + len(ref)
            stop = ref_end + flank_right
            if start < 0 or stop > len(fasta[contig]):
                skipped["off_contig"] += 1
                continue

            left = str(fasta[contig][start : pos - 1].seq).upper()
            observed = str(fasta[contig][pos - 1 : ref_end].seq).upper()
            right = str(fasta[contig][ref_end:stop].seq).upper()
            if set(left + observed + right) - dna_utils.VALID_CHARS:
                skipped["gap_in_window"] += 1
                continue

            n_compared += 1
            if observed != ref.upper():
                skipped["ref_mismatch"] += 1
                continue

            usable_alts = []
            if inspect_alts:
                for alt in alt_field.split(","):
                    if not alt or set(alt) - dna_utils.VALID_CHARS:
                        skipped["non_dna_alt"] += 1
                        continue
                    if max_allele_length is not None and len(alt) > max_allele_length:
                        skipped["allele_too_long"] += 1
                        continue
                    vtype = _variant_type(ref, alt)
                    if variant_types is not None and vtype not in variant_types:
                        skipped["variant_type"] += 1
                        continue
                    usable_alts.append((alt.upper(), vtype))

                if variant_types is not None and not usable_alts:
                    # Filtered by type and nothing survived: the caller excluded
                    # this site, so its reference window goes too.
                    continue

            base = {
                "chrom": chrom,
                "pos": pos,
                "ref": ref.upper(),
                "window_start": start,
                "window_stop": stop,
                "name_stem": f"{chrom}_{pos}_{ref.upper()}",
            }

            # The site is keyed on the VCF's own spelling, so every card
            # round-trips against the supplied file and a reference name stays a
            # strict prefix of its variants' names. A file mixing `chr1` and `1`
            # therefore yields one reference window per spelling.
            site = (chrom, pos, ref.upper())
            if alleles in ("ref", "both") and site not in seen_sites:
                seen_sites.add(site)
                rows.append(
                    _make_row(
                        base,
                        "ref",
                        left + observed + right,
                        None,
                        [f"info_{k}" for k in info_keys],
                    )
                )

            if alleles == "ref":
                continue

            parsed_info = _parse_info(info, info_keys)
            for alt, vtype in usable_alts:
                rows.append(
                    _make_row(
                        base,
                        "alt",
                        left + alt + right,
                        {
                            "alt": alt,
                            "variant_type": vtype,
                            "variant_id": None if var_id == "." else var_id,
                            "filter": None if filt == "." else filt,
                        },
                        parsed_info,
                    )
                )

    return rows, skipped, n_compared


def _make_row(
    base: dict,
    allele: str,
    seq: str,
    alt_cards: Optional[dict],
    info: object,
) -> dict:
    """Assemble one output row from the shared site fields and the allele's own.

    For a reference window ``alt_cards`` is None and ``info`` is the list of
    ``info_`` keys to set to None; for an alternate window ``info`` holds the
    parsed values.
    """
    row = {k: base[k] for k in ("chrom", "pos", "ref", "window_start", "window_stop")}
    row["seq"] = seq
    row["allele"] = allele
    if alt_cards is None:
        row.update(dict.fromkeys(_ALT_ONLY_KEYS))
        row.update(dict.fromkeys(info))
        row["name"] = base["name_stem"]
    else:
        row.update(alt_cards)
        row.update(info)
        row["name"] = f"{base['name_stem']}_{alt_cards['alt']}"
    return row


class FromVcfOp(Operation):
    """Create a pool of reference-genome windows around VCF variants."""

    factory_name = "from_vcf"

    def __init__(
        self,
        rows: list[dict],
        info_fields: Optional[Sequence[str]] = None,
        skipped: Optional[dict] = None,
        prefix: Optional[str] = None,
        style: Optional[str] = None,
        cards: CardsType = None,
        iter_order: Optional[Real] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize FromVcfOp."""
        self.rows = rows
        self.info_fields = list(info_fields) if info_fields else []
        self.skipped = dict(skipped) if skipped else {}
        self._style = style
        self._current_idx = 0
        self.design_card_keys = [
            *_CARD_KEYS,
            *(f"info_{key}" for key in self.info_fields),
        ]

        lengths = [dna_utils.get_length_without_tags(row["seq"]) for row in rows]
        seq_length = lengths[0] if all(L == lengths[0] for L in lengths) else None

        super().__init__(
            parent_pools=[],
            num_states=len(rows),
            mode="sequential",
            seq_length=seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            cards=cards,
            _natural_num_states=len(rows),
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return the window for the current state and its design card."""
        state = self.state.value
        self._current_idx = 0 if state is None else state
        row = self.rows[self._current_idx]

        from ..utils.style_utils import SeqStyle, styles_suppressed

        if styles_suppressed():
            output_seq = DnaSeq(row["seq"], None)
        else:
            output_seq = DnaSeq(row["seq"], SeqStyle.full(len(row["seq"]), self._style))

        return output_seq, {key: row[key] for key in self.design_card_keys}

    def compute_name_contributions(self, global_state=None, max_global_state=None) -> list[str]:
        """Contribute the variant name for the current state, prefixed if asked."""
        if not self.state.is_active:
            return []
        name = self.rows[self._current_idx]["name"]
        return [f"{self.prefix}_{name}" if self.prefix else name]
