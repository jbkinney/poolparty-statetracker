ORF Operations
==============

ORF operations target open reading frames within sequences, enabling
codon-level manipulations and translations while respecting reading frame
boundaries. These operations require that the target region be annotated
with :doc:`annotate_orf` or :doc:`insert_tags` so that the reading frame
is known.

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

Genetic code
------------

PoolParty uses the standard genetic code by default, with codons in each
amino-acid group sorted by human codon usage frequency (source: `Kazusa
<https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=9606>`_).
The first codon in each list is the most frequent. This ordering matters:
``mutation_type="missense_only_first"`` in :doc:`mutagenize_orf` and
``codon_selection="first"`` in :doc:`reverse_translate` both select the
first codon per amino acid.

The built-in table is a dict mapping single-letter amino acids to lists
of codons:

.. code-block:: python

    {
        "M": ["ATG"],
        "F": ["TTC", "TTT"],
        "L": ["CTG", "CTC", "CTT", "TTG", "TTA", "CTA"],
        "S": ["AGC", "TCC", "TCT", "TCA", "AGT", "TCG"],
        ...
        "*": ["TGA", "TAA", "TAG"],
    }

To supply a custom genetic code (e.g. for a non-standard organism or a
host-optimized codon table), pass a dict in the same format:

.. code-block:: python

    my_table = {
        "M": ["ATG"],
        "F": ["TTT", "TTC"],   # TTT listed first
        ...
        "*": ["TAA", "TAG", "TGA"],
    }
    pp.init(genetic_code=my_table)

    # or change after init:
    pp.set_genetic_code(my_table)

.. toctree::
   :hidden:

   mutagenize_orf
   translate
   annotate_orf
   stylize_orf
   reverse_translate
