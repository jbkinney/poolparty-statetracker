"""DnaPool class for DNA sequence pools."""

from .pool import Pool
from .pool_mixins import DnaMixin, ExportMixin, FilterMixin, StatsMixin


class DnaPool(Pool, DnaMixin, FilterMixin, ExportMixin, StatsMixin):
    """Pool specialized for DNA sequences.

    Inherits all generic operations from Pool and adds DNA-specific
    operations via DnaMixin:
    - rc() - reverse complement
    - translate() - DNA to protein translation
    - insert_from_iupac() - insert sequences from IUPAC codes
    - insert_from_motif() - insert sequences from probability matrix
    - insert_kmers() - insert all k-mers
    - annotate_orf() - annotate an ORF region
    - stylize_orf() - apply ORF-aware styling
    - mutagenize_orf() - apply codon-level mutations

    And filtering operations via FilterMixin:
    - filter_gc() - filter by GC content
    - filter_homopolymer() - filter out long homopolymer runs
    - filter_complexity() - filter by linguistic complexity
    - filter_dust() - filter by DUST complexity score
    - filter_restriction_sites() - filter out sequences with restriction sites

    And export operations via ExportMixin:
    - to_file() - export to CSV, TSV, FASTA, or JSONL

    And the library readout via StatsMixin:
    - stats() - summarise the library this pool produces
    """

    def __repr__(self) -> str:
        num_states_str = "None" if self.num_states is None else str(self.num_states)
        return f"DnaPool(id={self._id}, name={self.name!r}, op={self.operation.name!r}, num_states={num_states_str})"
