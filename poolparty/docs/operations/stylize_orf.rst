stylize_orf
===========

Apply codon-aware inline styling to sequences without modifying them. Pass
``style_codons=`` to cycle a list of styles across whole codons, or
``style_frames=`` (length a multiple of 3) to style each base position within
a codon independently. Use ``region=`` to restrict coloring to a named ORF
segment.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Examples
--------

Basic codon-colored display of a CDS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply alternating ``codon_a`` / ``codon_b`` styles across the 5-codon CDS
``ATGAAATTTGGGCCC`` to make the reading frame immediately visible.

.. code-block:: python

    cds    = pp.from_seq("ATGAAATTTGGGCCC")
    styled = pp.stylize_orf(cds, style_codons=["codon_a", "codon_b"])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; alternating codon colours, frame +1)</em>
    <span class="pp-codon-a">ATG</span><span class="pp-codon-b">AAA</span><span class="pp-codon-a">TTT</span><span class="pp-codon-b">GGG</span><span class="pp-codon-a">CCC</span>
    </div>

Apply to a longer ORF
~~~~~~~~~~~~~~~~~~~~~~~

Style a 7-codon CDS (``ATGAAATTTGGGCCCAGCGAT``, 21 nt) with the same
two-colour cycle; the pattern repeats automatically.

.. code-block:: python

    cds    = pp.from_seq("ATGAAATTTGGGCCCAGCGAT")
    styled = pp.stylize_orf(cds, style_codons=["codon_a", "codon_b"])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; 7-codon CDS with alternating codon colours)</em>
    <span class="pp-codon-a">ATG</span><span class="pp-codon-b">AAA</span><span class="pp-codon-a">TTT</span><span class="pp-codon-b">GGG</span><span class="pp-codon-a">CCC</span><span class="pp-codon-b">AGC</span><span class="pp-codon-a">GAT</span>
    </div>

Chain with mutagenize_orf to visualize mutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply ``mutagenize_orf`` first to introduce codon mutations, then
``stylize_orf`` to colour the reading frame so mutated codons are visible in
both their position and their codon context.

.. code-block:: python

    cds     = pp.from_seq("ATGAAATTTGGGCCC")
    mutants = pp.mutagenize_orf(cds, num_mutations=1, style="bold_red")
    styled  = pp.stylize_orf(mutants, style_codons=["codon_a", "codon_b"])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; codon-coloured frame with 1 missense mutation highlighted red)</em>
    <span class="pp-codon-a">ATG</span><span class="pp-codon-b"><span class="pp-mut">CGT</span></span><span class="pp-codon-a">TTT</span><span class="pp-codon-b">GGG</span><span class="pp-codon-a">CCC</span><br>
    <span class="pp-codon-a">ATG</span><span class="pp-codon-b">AAA</span><span class="pp-codon-a"><span class="pp-mut">ACT</span></span><span class="pp-codon-b">GGG</span><span class="pp-codon-a">CCC</span><br>
    <span class="pp-ellipsis">... each draw shows the mutated codon in its reading-frame colour</span>
    </div>

See :func:`~poolparty.stylize_orf`.
