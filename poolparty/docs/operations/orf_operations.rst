ORF Operations
==============

ORF operations target open reading frames within sequences, enabling
codon-level manipulations and translations while respecting reading frame
boundaries. These operations require that the target region be annotated
with :doc:`annotate_orf` or :doc:`insert_tags` so that the reading frame
is known.

.. code-block:: python

    import poolparty as pp
    pp.init()

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`mutagenize_orf`
     - Introduce codon-level missense mutations within a coding sequence.
   * - :doc:`translate`
     - Translate a DNA pool to a protein pool.
   * - :doc:`annotate_orf`
     - Tag an ORF region within a longer sequence.
   * - :doc:`stylize_orf`
     - Apply alternating codon colours to visualize reading frames.
   * - :doc:`reverse_translate`
     - Back-translate a protein pool to DNA using a codon table.

.. toctree::
   :hidden:

   mutagenize_orf
   translate
   annotate_orf
   stylize_orf
   reverse_translate
