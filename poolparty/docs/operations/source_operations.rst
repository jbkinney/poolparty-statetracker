Source Operations
=================

Source operations create the initial Pools from which all downstream Pools
are constructed. They do not require an existing pool as input. Sources may
represent user-specified sequences, DNA sequences generated using IUPAC
ambiguity codes or position weight matrices, random k-mers, constrained
barcodes, and more.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`from_seq`
     - Create a pool from a single sequence string, optionally with region tags.
   * - :doc:`from_seqs`
     - Create a pool that draws uniformly from a list of sequences.
   * - :doc:`from_fasta`
     - Load sequences from a FASTA file.
   * - :doc:`from_vcf`
     - Create a pool of reference-genome windows around variants in a VCF file.
   * - :doc:`from_iupac`
     - Enumerate all sequences consistent with an IUPAC ambiguity string.
   * - :doc:`from_motif`
     - Sample sequences from a position-probability matrix.
   * - :doc:`get_kmers`
     - Enumerate every k-mer of a given length over a specified alphabet.
   * - :doc:`get_barcodes`
     - Generate DNA barcodes satisfying distance, GC, and homopolymer constraints.

.. toctree::
   :hidden:

   from_seq
   from_seqs
   from_fasta
   from_vcf
   from_iupac
   from_motif
   get_kmers
   get_barcodes
