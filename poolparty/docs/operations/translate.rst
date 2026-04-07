translate
=========

Translate a DNA pool into a protein pool using the standard genetic code.
Pass ``region=`` to translate only a named ORF segment; ``include_stop=False``
omits the trailing stop codon from the output.

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
     - DNA pool or plain sequence string to translate.
   * - ``region``
     - ``str | Sequence[int] | None``
     - ``None``
     - Region name (from ``annotate_orf``) or ``[start, stop]`` interval.
       ``None`` translates the full sequence.
   * - ``frame``
     - ``int | None``
     - ``None``
     - Reading frame (``+1`` … ``+3``, ``-1`` … ``-3``). For a named
       :class:`~poolparty.OrfRegion`, defaults to the region's frame;
       otherwise ``+1``.
   * - ``include_stop``
     - ``bool``
     - ``True``
     - When ``True``, include the stop codon as ``*`` in the protein.
   * - ``preserve_codon_styles``
     - ``bool``
     - ``True``
     - If ``True``, copy nucleotide styles to amino acids when all three
       bases of a codon share the same style.
   * - ``genetic_code``
     - ``str | dict``
     - ``"standard"``
     - Built-in code name or a custom codon table mapping.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Enumeration order when combined with other pools.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for sequence names in the resulting pool.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.translate` in the
   :doc:`API Reference </api>`.

Examples
--------

Translate a 5-codon CDS
~~~~~~~~~~~~~~~~~~~~~~~~

Translate the 15-nucleotide sequence ``ATGAAATTTGGGCCC`` to the 5-residue
peptide M-K-F-G-P.

.. code-block:: python

    cds     = pp.from_seq("ATGAAATTTGGGCCC")
    protein = pp.translate(cds)
    protein.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">protein: seq_length=5, num_states=1</em>
    MKFGP
    </div>

Translate a mutagenized CDS pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chain ``mutagenize_orf`` with ``translate``. When ``mutagenize_orf`` applies
``style="red"``, ``translate`` preserves that style on the resulting amino
acid (controlled by ``preserve_codon_styles``, default ``True``).

.. code-block:: python

    cds     = pp.from_seq("ATGAAATTTGGGCCC")
    mutants = pp.mutagenize_orf(cds, num_mutations=1, mode="random", style="red")
    protein = pp.translate(mutants)
    protein.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">protein: seq_length=5, num_states=1</em>
    MK<span class="pp-mut">W</span>GP<br>
    M<span class="pp-mut">T</span>FGP<br>
    MKFG<span class="pp-mut">R</span><br>
    <span class="pp-mut">L</span>KFGP
    <span class="pp-ellipsis">... (stochastic; mutated residue highlighted)</span>
    </div>

Translate only the ORF region within a longer sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the CDS is embedded in flanking UTR context, tag it with
``annotate_orf`` and pass the region name to ``translate`` so only the
annotated ORF is translated.

.. code-block:: python

    seq  = pp.from_seq("TATAATGAAATTTGGGCCCTAA")
    seq  = pp.annotate_orf(seq, "gene", extent=(4, 19))
    prot = pp.translate(seq, region="gene", include_stop=False)
    prot.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">prot: seq_length=None, num_states=1</em>
    MKFGP
    </div>

See :func:`~poolparty.translate`.
