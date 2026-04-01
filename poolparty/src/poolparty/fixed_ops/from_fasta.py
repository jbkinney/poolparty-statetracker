"""FromFasta operation - create a pool from genomic region(s) in a FASTA file."""

from numbers import Real

from pyfaidx import Fasta

from ..dna_pool import DnaPool
from ..pool import Pool
from ..types import CardsType, Literal, Optional, Pool_type, RegionType, Sequence, Union, beartype
from ..utils import dna_utils

# Type alias for a single coordinate tuple: (chrom, start, stop, strand)
Coordinate = tuple[str, int, int, Literal["+", "-"]]


def _extract_sequence(
    fasta: Fasta,
    chrom: str,
    start: int,
    stop: int,
    strand: Literal["+", "-"],
) -> str:
    """Extract a single sequence from FASTA, handling circular genome wrap-around."""
    if start <= stop:
        seq = str(fasta[chrom][start:stop].seq)
    else:
        # Circular wrap-around: start > stop means sequence crosses origin
        chrom_len = len(fasta[chrom])
        seq = str(fasta[chrom][start:chrom_len].seq) + str(fasta[chrom][0:stop].seq)

    if strand == "-":
        seq = dna_utils.reverse_complement(seq)

    return seq


def _is_single_coordinate(coordinates) -> bool:
    """Check if coordinates is a single (chrom, start, stop, strand) tuple."""
    if not isinstance(coordinates, tuple | list):
        return False
    if len(coordinates) != 4:
        return False
    # Single coordinate: first element is string (chrom), rest are int/int/str
    return isinstance(coordinates[0], str) and isinstance(coordinates[1], int)


@beartype
def from_fasta(
    fasta_path: str,
    coordinates: Union[Coordinate, Sequence[Coordinate]],
    pool: Optional[Union[Pool, str]] = None,
    region: RegionType = None,
    remove_tags: Optional[bool] = None,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
    style: Optional[str] = None,
    cards: CardsType = None,
) -> DnaPool:
    """
    Extract genomic region(s) from a FASTA file and create a Pool.

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file (will be indexed with pyfaidx).
    coordinates : tuple or list of tuples
        Single coordinate as (chrom, start, stop, strand) or list of such tuples.
        Coordinates are 0-based [start, stop). If strand='-', sequence is reverse
        complemented. For circular genomes, start > stop indicates wrap-around.
    pool : Optional[Union[Pool, str]], default=None
        Background pool or sequence. If provided with region, extracted sequence(s)
        replace the region content.
    region : RegionType, default=None
        Region to replace in pool. Can be a marker name or [start, stop] interval.
        Required if pool is provided.
    remove_tags : Optional[bool], default=None
        If True and region is a marker name, remove marker tags from the output.
        Only relevant in single-coordinate mode (has no effect in batch mode).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation (batch mode only).
    prefix : str, optional
        Prefix for sequence names. Names are "{prefix}_{chrom}:{start}-{stop}({strand})"
        or "{chrom}:{start}-{stop}({strand})" if no prefix.
    style : Optional[str], default=None
        Style to apply to extracted sequences (e.g., 'red', 'blue bold').
    cards : list[str] or dict[str, str], optional
        Design card keys to include. Available keys (batch mode only):
        ``'seq_name'``, ``'seq_index'``. Ignored in single-coordinate mode.

    Returns
    -------
    Pool_type
        A Pool yielding the extracted genomic sequence(s).
    """
    from ..base_ops.from_seqs import from_seqs
    from ..party import get_active_party
    from .from_seq import from_seq

    party = get_active_party()
    if party is None:
        raise RuntimeError(
            "from_fasta requires an active Party context. "
            "Use 'with pp.Party() as party:' to create one."
        )

    # Check if single coordinate or batch
    is_single = _is_single_coordinate(coordinates)

    if is_single:
        # Single region mode
        chrom, start, stop, strand = coordinates

        with Fasta(fasta_path) as fasta:
            seq = _extract_sequence(fasta, chrom, start, stop, strand)

        return from_seq(
            seq=seq,
            pool=pool,
            region=region,
            remove_tags=remove_tags,
            iter_order=iter_order,
            prefix=prefix,
            style=style,
            _factory_name="from_fasta",
        )
    else:
        # Batch mode: list of coordinate tuples
        coords_list = list(coordinates)

        # Validate all tuples have 4 elements
        for i, coord in enumerate(coords_list):
            if len(coord) != 4:
                raise ValueError(
                    f"Coordinate at index {i} must be (chrom, start, stop, strand), "
                    f"got {len(coord)} elements"
                )

        # Extract all sequences
        with Fasta(fasta_path) as fasta:
            seqs = [
                _extract_sequence(fasta, chrom, start, stop, strand)
                for chrom, start, stop, strand in coords_list
            ]

        # Generate names: "{prefix}_{chrom}:{start}-{stop}({strand})" or "{chrom}:{start}-{stop}({strand})"
        if prefix:
            seq_names = [
                f"{prefix}_{chrom}:{start}-{stop}({strand})"
                for chrom, start, stop, strand in coords_list
            ]
        else:
            seq_names = [
                f"{chrom}:{start}-{stop}({strand})" for chrom, start, stop, strand in coords_list
            ]

        return from_seqs(
            seqs=seqs,
            pool=pool,
            region=region,
            style=style,
            seq_names=seq_names,
            mode="sequential",
            iter_order=iter_order,
            cards=cards,
            _factory_name="from_fasta",
        )
