ORF Operations
==============

ORF operations target open reading frames within sequences, enabling
codon-level manipulations and translations while respecting reading frame
boundaries. A named region annotated with :doc:`annotate_orf` carries its
frame automatically. You can also target the whole sequence or an interval
and pass ``frame=`` explicitly.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`mutagenize_orf`
     - Introduce codon-level missense mutations within a coding sequence.
   * - :doc:`deletion_scan_orf`
     - Scan whole-codon deletions in coding order across any of six frames.
   * - :doc:`insertion_scan_orf`
     - Splice or overwrite coding-oriented whole codons across any frame.
   * - :doc:`translate`
     - Translate a DNA pool to a protein pool.
   * - :doc:`annotate_orf`
     - Tag an ORF region within a longer sequence.
   * - :doc:`stylize_orf`
     - Apply alternating codon colours to visualize reading frames.
   * - :doc:`reverse_translate`
     - Back-translate a protein pool to DNA using a codon table.
   * - :doc:`genetic_code`
     - Configure the standard or a custom genetic code and codon ordering.

.. toctree::
   :hidden:

   mutagenize_orf
   deletion_scan_orf
   insertion_scan_orf
   translate
   annotate_orf
   stylize_orf
   reverse_translate
   genetic_code
