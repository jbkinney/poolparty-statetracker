"""FromFasta operation - create a pool from a genomic region in a FASTA file."""

from numbers import Real

from pyfaidx import Fasta

from ..pool import Pool
from ..types import Literal, Optional, Pool_type, RegionType, Union, beartype
from ..utils import dna_utils


@beartype
def from_fasta(
    fasta_path: str,
    chrom: str,
    start: int,
    end: int,
    strand: Literal["+", "-"] = "+",
    pool: Optional[Union[Pool, str]] = None,
    region: RegionType = None,
    remove_tags: Optional[bool] = None,
    iter_order: Optional[Real] = None,
    style: Optional[str] = None,
) -> Pool_type:
    """
    Extract a genomic region from a FASTA file and create a Pool.

    Uses pyfaidx for indexed access. Coordinates are 0-based [start, end).
    If strand='-', the sequence is reverse complemented.
    Additional parameters (pool, region, etc.) are passed to from_seq().
    """
    from ..party import get_active_party
    from .from_seq import from_seq

    party = get_active_party()
    if party is None:
        raise RuntimeError(
            "from_fasta requires an active Party context. "
            "Use 'with pp.Party() as party:' to create one."
        )

    # Load FASTA and extract sequence
    fasta = Fasta(fasta_path)
    try:
        seq = str(fasta[chrom][start:end].seq)
    finally:
        fasta.close()  # Ensure file handle is released (critical for Windows)

    # Reverse complement if strand is '-'
    if strand == "-":
        seq = dna_utils.reverse_complement(seq)

    # Delegate to from_seq with appropriate factory name
    return from_seq(
        seq=seq,
        pool=pool,
        region=region,
        remove_tags=remove_tags,
        iter_order=iter_order,
        style=style,
        _factory_name="from_fasta",
    )
