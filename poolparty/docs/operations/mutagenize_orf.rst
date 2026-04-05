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

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pool``
     - ``Pool | str``
     - *(required)*
     - Parent pool or plain DNA sequence string to mutate.
   * - ``num_mutations``
     - ``int | None``
     - ``None``
     - Fixed number of codon mutations per draw. Mutually exclusive with
       ``mutation_rate``.
   * - ``mutation_rate``
     - ``float | None``
     - ``None``
     - Per-codon probability of mutation. Mutually exclusive with
       ``num_mutations``.
   * - ``mutation_type``
     - ``str``
     - ``'missense_only_first'``
     - Type of codon mutation. One of ``'any_codon'``,
       ``'nonsynonymous_first'``, ``'nonsynonymous_random'``,
       ``'missense_only_first'``, ``'missense_only_random'``,
       ``'synonymous'``, ``'nonsense'``.
   * - ``region``
     - ``str | list[int] | None``
     - ``None``
     - Region to mutate: a tagged ORF name, a ``[start, stop]`` interval, or
       ``None`` to mutate the full sequence.
   * - ``codon_positions``
     - ``list[int] | slice | None``
     - ``None``
     - Eligible codon indices (0-based); ``None`` means every codon in the
       mutated span may change.
   * - ``frame``
     - ``int | None``
     - ``None``
     - Reading frame (e.g. ``1``..``3`` or ``-1``..``-3``). If ``None`` and
       ``region`` names an :class:`~poolparty.OrfRegion`, the frame is taken
       from that region.
   * - ``style``
     - ``str | None``
     - ``None``
     - Named display style applied to mutated codons (e.g. ``"red"``).
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names in the output pool.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` enumerates codon-mutation variants in order (requires
       ``num_mutations`` and a uniform ``mutation_type``); ``'random'`` samples
       each draw independently; ``'fixed'`` is available for fixed-parameter
       pools.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Total number of output states. In random mode, ``None`` defaults to a
       single stochastic state unless set otherwise.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``cards``
     - ``list[str] | dict | None``
     - ``None``
     - Design card keys (e.g. ``codon_positions``, ``wt_codons``,
       ``mut_codons``, ``wt_aas``, ``mut_aas``).

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.mutagenize_orf` in the
   :doc:`API Reference </api>`.

Examples
--------

Single codon substitution in a 5-codon CDS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Draw one missense codon mutation per sequence from the 5-codon CDS
``ATGAAATTTGGGCCC`` (M-K-F-G-P).

.. code-block:: python

    cds     = pp.from_seq("ATGAAATTTGGGCCC")
    mutants = pp.mutagenize_orf(cds, num_mutations=1, mode="random", style="red")
    mutants.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mutants: seq_length=15, num_states=1</em>
    ATGAAA<span class="pp-mut">TGG</span>GGGCCC<br>
    ATGAAATTT<span class="pp-mut">TGC</span>CCC<br>
    <span class="pp-mut">TGG</span>AAATTTGGGCCC<br>
    <span class="pp-mut">AGC</span>AAATTTGGGCCC
    <span class="pp-ellipsis">... (stochastic; each draw mutates one codon)</span>
    </div>

Two simultaneous codon mutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply exactly two independent missense substitutions per draw, chosen from
distinct codon positions.

.. code-block:: python

    cds     = pp.from_seq("ATGAAATTTGGGCCC")
    mutants = pp.mutagenize_orf(cds, num_mutations=2, mode="random")
    mutants.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mutants: seq_length=15, num_states=1</em>
    ATGAAAAGCTGGCCC<br>
    CAGAAAAACGGGCCC<br>
    ATGAAAGAGTTCCCC<br>
    ATGAAAAACGGGGGC
    <span class="pp-ellipsis">... (stochastic; two codons mutated per draw)</span>
    </div>

Restrict mutations to specific codons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``codon_positions=`` (0-indexed list) to limit which codons are eligible.
Here only codon positions 1 and 3 (AAA and GGG) can be mutated.

.. code-block:: python

    cds     = pp.from_seq("ATGAAATTTGGGCCC")
    mutants = pp.mutagenize_orf(
        cds, num_mutations=1, codon_positions=[1, 3], mode="random"
    )
    mutants.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mutants: seq_length=15, num_states=1</em>
    ATGAAATTTGAGCCC<br>
    ATGCAGTTTGGGCCC<br>
    ATGAAATTTTACCCC<br>
    ATGAGATTTGGGCCC
    <span class="pp-ellipsis">... (stochastic; only codons 1 or 3 change)</span>
    </div>

Apply to a CDS embedded in flanking UTR context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tag the ORF with ``annotate_orf``, then mutate only within that region; the
5-prime and 3-prime UTR flanks are always returned unchanged.

.. code-block:: python

    import poolparty as pp

    pp.init()
    seq  = pp.from_seq("TATAATGAAATTTGGGCCCTAA")
    seq  = pp.annotate_orf(seq, "gene", extent=(3, 18))
    muts = pp.mutagenize_orf(seq, region="gene", num_mutations=1, mode="random")
    muts.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">muts: seq_length=22, num_states=1</em>
    TAT<span class="pp-xtag-cre">&lt;gene&gt;</span>AATAGCATTTGGGCC<span class="pp-xtag-cre">&lt;/gene&gt;</span>CTAA<br>
    TAT<span class="pp-xtag-cre">&lt;gene&gt;</span>TACGAAATTTGGGCC<span class="pp-xtag-cre">&lt;/gene&gt;</span>CTAA<br>
    TAT<span class="pp-xtag-cre">&lt;gene&gt;</span>AATACCATTTGGGCC<span class="pp-xtag-cre">&lt;/gene&gt;</span>CTAA<br>
    TAT<span class="pp-xtag-cre">&lt;gene&gt;</span>AATTTCATTTGGGCC<span class="pp-xtag-cre">&lt;/gene&gt;</span>CTAA<br>
    TAT<span class="pp-xtag-cre">&lt;gene&gt;</span>CCCGAAATTTGGGCC<span class="pp-xtag-cre">&lt;/gene&gt;</span>CTAA
    <span class="pp-ellipsis">... (stochastic; one codon within the ORF per draw)</span>
    </div>

See :func:`~poolparty.mutagenize_orf`.
