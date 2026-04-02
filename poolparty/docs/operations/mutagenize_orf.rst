mutagenize_orf
==============

Introduce codon-level mutations into an ORF sequence. Exactly one of
``num_mutations`` or ``mutation_rate`` must be provided; use ``region=`` to
restrict mutations to a named ORF segment and ``codon_positions=`` to limit
which codons are eligible.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Examples
--------

Single codon substitution in a 5-codon CDS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Draw one missense codon mutation per sequence from the 5-codon CDS
``ATGAAATTTGGGCCC`` (M-K-F-G-P).

.. code-block:: python

    cds     = pp.from_seq("ATGAAATTTGGGCCC")
    mutants = pp.mutagenize_orf(cds, num_mutations=1)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 1 random missense codon substitution per draw)</em>
    ATG<span class="pp-mut">CGT</span>TTTGGGCCC<br>
    ATGAAA<span class="pp-mut">ACT</span>GGGCCC<br>
    ATGAAATTT<span class="pp-mut">CAT</span>CCC<br>
    <span class="pp-ellipsis">... each draw mutates one codon to a different amino acid</span>
    </div>

Two simultaneous codon mutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply exactly two independent missense substitutions per draw, chosen from
distinct codon positions.

.. code-block:: python

    cds     = pp.from_seq("ATGAAATTTGGGCCC")
    mutants = pp.mutagenize_orf(cds, num_mutations=2)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 2 random missense codon substitutions per draw)</em>
    ATG<span class="pp-mut">CGT</span>TTT<span class="pp-mut">CAT</span>CCC<br>
    ATG<span class="pp-mut">GAG</span>TTTGGG<span class="pp-mut">GTG</span><br>
    ATGAAA<span class="pp-mut">ACT</span>GGG<span class="pp-mut">GTG</span><br>
    <span class="pp-ellipsis">... each draw carries exactly 2 missense substitutions at distinct codons</span>
    </div>

Restrict mutations to specific codons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``codon_positions=`` (0-indexed list) to limit which codons are eligible.
Here only codon positions 1 and 3 (AAA and GGG) can be mutated.

.. code-block:: python

    cds     = pp.from_seq("ATGAAATTTGGGCCC")
    mutants = pp.mutagenize_orf(cds, num_mutations=1, codon_positions=[1, 3])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 1 missense mutation restricted to codons 1 and 3)</em>
    ATG<span class="pp-mut">GAG</span>TTTGGGCCC<br>
    ATGAAATTT<span class="pp-mut">CAT</span>CCC<br>
    ATG<span class="pp-mut">CGT</span>TTTGGGCCC<br>
    <span class="pp-ellipsis">... only the AAA (pos 1) or GGG (pos 3) codon is mutated each draw</span>
    </div>

Apply to a CDS embedded in flanking UTR context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tag the ORF with ``annotate_orf``, then mutate only within that region; the
5-prime and 3-prime UTR flanks are always returned unchanged.

.. code-block:: python

    seq  = pp.from_seq("TATAATGAAATTTGGGCCCTAA")
    seq  = pp.annotate_orf(seq, "gene", extent=(3, 18))
    muts = pp.mutagenize_orf(seq, region="gene", num_mutations=1)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 1 missense mutation inside <em>gene</em>; UTR flanks unchanged)</em>
    TAT<span class="pp-region">ATG<span class="pp-mut">CGT</span>TTTGGGCCC</span>TAA<br>
    TAT<span class="pp-region">ATGAAA<span class="pp-mut">ACT</span>GGGCCC</span>TAA<br>
    TAT<span class="pp-region">ATGAAATTT<span class="pp-mut">CAT</span>CCC</span>TAA<br>
    <span class="pp-ellipsis">... each draw mutates one codon within the ORF; flanks are fixed</span>
    </div>

See :func:`~poolparty.mutagenize_orf`.
