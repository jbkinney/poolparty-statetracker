annotate_orf
============

Tag an ORF region within a sequence and register its reading frame, optionally
applying a display style. If the region does not yet exist, ``extent=`` sets
its boundaries; once registered as an ``OrfRegion``, downstream operations
such as ``mutagenize_orf`` and ``translate`` can look up the frame
automatically.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Examples
--------

Annotate an ORF within a longer sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tag the coding sequence ``ATGAAATTTGGGCCC`` (positions 3&ndash;18) inside a
sequence that also contains 5-prime and 3-prime UTR flanks.

.. code-block:: python

    seq = pp.from_seq("TATAATGAAATTTGGGCCCTAA")
    seq = pp.annotate_orf(seq, "gene", extent=(3, 18))

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; ORF tagged as <em>gene</em> at positions 3&ndash;18)</em>
    TAT<span class="pp-region">ATGAAATTTGGGCCC</span>TAA
    </div>

Annotate with a display style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply alternating codon colours at the same time by passing
``style_codons=`` so the reading frame is immediately visible.

.. code-block:: python

    seq = pp.from_seq("TATAATGAAATTTGGGCCCTAA")
    seq = pp.annotate_orf(
        seq, "gene", extent=(3, 18),
        style_codons=["codon_a", "codon_b"],
    )

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; <em>gene</em> ORF tagged and codon-coloured)</em>
    TAT<span class="pp-region"><span class="pp-codon-a">ATG</span><span class="pp-codon-b">AAA</span><span class="pp-codon-a">TTT</span><span class="pp-codon-b">GGG</span><span class="pp-codon-a">CCC</span></span>TAA
    </div>

Chain annotate_orf with mutagenize_orf to visualize mutations in context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After tagging the ORF, pass the region name to ``mutagenize_orf`` to mutate
only within the annotated coding sequence while keeping the UTR flanks intact.

.. code-block:: python

    seq  = pp.from_seq("TATAATGAAATTTGGGCCCTAA")
    seq  = pp.annotate_orf(seq, "gene", extent=(3, 18))
    muts = pp.mutagenize_orf(seq, region="gene", num_mutations=1,
                             style="bold_red")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 1 missense mutation inside <em>gene</em>; UTR flanks fixed)</em>
    TAT<span class="pp-region">ATG<span class="pp-mut">CGT</span>TTTGGGCCC</span>TAA<br>
    TAT<span class="pp-region">ATGAAA<span class="pp-mut">ACT</span>GGGCCC</span>TAA<br>
    TAT<span class="pp-region">ATGAAATTT<span class="pp-mut">CAT</span>CCC</span>TAA<br>
    <span class="pp-ellipsis">... each draw mutates one codon; the <em>gene</em> region frame is reused automatically</span>
    </div>

See :func:`~poolparty.annotate_orf`.
