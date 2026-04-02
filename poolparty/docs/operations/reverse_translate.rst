reverse_translate
=================

Convert a protein pool (or a protein sequence string) to a DNA pool by
back-translating each amino acid to a codon. Two selection strategies are
available: ``"first"`` always picks the most frequent codon (deterministic),
while ``"random"`` samples from all synonymous codons (stochastic).

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Parameters
----------

.. list-table::
   :widths: 20 18 12 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pool``
     - ``Pool | str``
     - *(required)*
     - A :class:`~poolparty.ProteinPool` or a protein sequence string
       (e.g. ``"MKTL"``). DNA pools are not accepted.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Restrict translation to a named region or ``[start, stop]``
       interval. ``None`` translates the entire sequence.
   * - ``codon_selection``
     - ``str``
     - ``"first"``
     - ``"first"`` uses the most frequent codon for each amino acid
       (deterministic). ``"random"`` samples uniformly from synonymous
       codons (stochastic — each draw may differ).
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Number of states. Only relevant with ``codon_selection="random"``.
   * - ``genetic_code``
     - ``str | dict``
     - ``"standard"``
     - Genetic code to use. Pass a string identifier or a custom codon
       dictionary.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Iteration priority for downstream multi-pool iteration.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.

.. note::

   The output sequence length is always **3 × the protein length** (each
   amino acid maps to exactly one codon).

----

Examples
--------

Deterministic back-translation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``codon_selection="first"`` (the default) always picks the same codon for
each amino acid, producing a single fixed DNA sequence.

.. code-block:: python

    dna = pp.reverse_translate("MKTL")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; deterministic, most-frequent codons)</em>
    ATGAAGACCCTG
    </div>

Stochastic back-translation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``codon_selection="random"`` samples from synonymous codons, so each draw
may produce a different DNA sequence encoding the same protein.

.. code-block:: python

    dna = pp.reverse_translate("MKTL", codon_selection="random", num_states=4)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (4 sequences &mdash; random synonymous codons for MKTL)</em>
    ATGAAGACCCTG<br>
    ATGAAAACTCTT<br>
    ATGAAGACCCTA<br>
    ATGAAGACATTG<br>
    <span class="pp-ellipsis">... (each encodes the same protein)</span>
    </div>

Chain: translate then reverse-translate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Round-trip a DNA sequence through protein and back — the resulting DNA may
differ due to codon degeneracy, but the encoded protein is identical.

.. code-block:: python

    dna     = pp.from_seq("ATGAAACCCGGG")
    protein = pp.translate(dna)
    back    = pp.reverse_translate(protein, codon_selection="first")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; round-trip: DNA &rarr; protein &rarr; DNA)</em>
    ATGAAGCCCGGG
    </div>

See :func:`~poolparty.reverse_translate`.
