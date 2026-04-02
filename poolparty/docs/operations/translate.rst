translate
=========

Translate a DNA pool into a protein pool using the standard genetic code.
Pass ``region=`` to translate only a named ORF segment; ``include_stop=False``
omits the trailing stop codon from the output.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Examples
--------

Translate a 5-codon CDS
~~~~~~~~~~~~~~~~~~~~~~~~

Translate the 15-nucleotide sequence ``ATGAAATTTGGGCCC`` to the 5-residue
peptide M-K-F-G-P.

.. code-block:: python

    cds     = pp.from_seq("ATGAAATTTGGGCCC")
    protein = pp.translate(cds)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; ATG&thinsp;AAA&thinsp;TTT&thinsp;GGG&thinsp;CCC &rarr; M&thinsp;K&thinsp;F&thinsp;G&thinsp;P)</em>
    MKFGP
    </div>

Translate a mutagenized CDS pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chain ``mutagenize_orf`` with ``translate`` to observe the amino-acid
consequences of each codon mutation.

.. code-block:: python

    cds     = pp.from_seq("ATGAAATTTGGGCCC")
    mutants = pp.mutagenize_orf(cds, num_mutations=1)
    protein = pp.translate(mutants)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; protein translated from each codon-mutant draw)</em>
    M<span class="pp-mut">R</span>FGP<br>
    MK<span class="pp-mut">T</span>GP<br>
    MKF<span class="pp-mut">H</span>P<br>
    <span class="pp-ellipsis">... each draw is the protein product of one mutagenized CDS</span>
    </div>

Translate only the ORF region within a longer sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the CDS is embedded in flanking UTR context, tag it with
``annotate_orf`` and pass the region name to ``translate`` so only the
annotated ORF is translated.

.. code-block:: python

    seq  = pp.from_seq("TATAATGAAATTTGGGCCCTAA")
    seq  = pp.annotate_orf(seq, "gene", extent=(3, 21))
    prot = pp.translate(seq, region="gene", include_stop=False)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; only the <em>gene</em> ORF translated; stop codon excluded)</em>
    MKFGP
    </div>

See :func:`~poolparty.translate`.
