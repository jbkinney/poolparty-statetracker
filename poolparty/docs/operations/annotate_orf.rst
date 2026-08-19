annotate_orf
============

Register a region as an ``OrfRegion`` so that downstream operations such as
``mutagenize_orf``, ``stylize_orf``, and ``translate`` can look up the reading
frame automatically. If the region does not yet exist, ``extent=`` sets its
boundaries. Optionally apply codon-based styling at the same time via
``style_codons=`` or ``style_frames=``.

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
     - ``Pool``
     - *(required)*
     - The Pool to annotate.
   * - ``region_name``
     - ``str``
     - *(required)*
     - Name for the ORF region.
   * - ``extent``
     - ``tuple[int, int] | None``
     - ``None``
     - ``(start, stop)`` half-open interval defining the region boundaries.
       If ``None`` and the region doesn't exist, the entire sequence is used.
       Must be ``None`` if the region already exists.
   * - ``frame``
     - ``int``
     - ``1``
     - Reading frame (+1..+3 or -1..-3). Positive = 5'->3', negative =
       3'->5'. Determines which codon grid is used by downstream ORF
       operations.
       The first complete codon begins at base ``|frame|`` of the region,
       counted from the 5' end for positive frames and from the 3' end for
       negative frames. Bases outside a complete codon are ignored.
   * - ``style``
     - ``str | None``
     - ``None``
     - A single display style applied uniformly to the ORF region. Mutually
       exclusive with ``style_codons`` and ``style_frames``.
   * - ``style_codons``
     - ``list[str] | None``
     - ``None``
     - List of style names cycled across whole codons within the ORF.
       Mutually exclusive with ``style`` and ``style_frames``.
   * - ``style_frames``
     - ``list[str] | None``
     - ``None``
     - List of style names (length a multiple of 3) applied per base
       position within each codon. Mutually exclusive with ``style`` and
       ``style_codons``.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Enumeration order when combined with other pools.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.annotate_orf` in the
   :doc:`API Reference </api>`.

Examples
--------

Annotate a pre-tagged ORF
~~~~~~~~~~~~~~~~~~~~~~~~~

When the region already exists as XML tags in the sequence, just pass the
region name. ``annotate_orf`` registers it as an ``OrfRegion`` (with reading
frame) without changing the sequence.

.. code-block:: python

    wt  = pp.from_seq("<gene>ATGAAATTTGGGCCC</gene>")
    orf = pp.annotate_orf(wt, "gene")
    orf.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">orf: seq_length=15, num_states=1</em>
    <span class="pp-xtag-light">&lt;gene&gt;</span>ATGAAATTTGGGCCC<span class="pp-xtag-light">&lt;/gene&gt;</span>
    </div>

Define region boundaries with extent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``extent=(start, stop)`` to tag positions 4 through 19 (half-open) as the
ORF, without needing XML tags in the original sequence.

.. code-block:: python

    seq = pp.from_seq("TATAATGAAATTTGGGCCCTAA")
    orf = pp.annotate_orf(seq, "gene", extent=(4, 19))
    orf.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">orf: seq_length=22, num_states=1</em>
    TATA<span class="pp-xtag-light">&lt;gene&gt;</span>ATGAAATTTGGGCCC<span class="pp-xtag-light">&lt;/gene&gt;</span>TAA
    </div>

Style the ORF with codon colouring (style_codons)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``style_codons=`` to apply alternating codon colours at annotation time,
making the reading frame immediately visible.

.. code-block:: python

    seq    = pp.from_seq("TATAATGAAATTTGGGCCCTAA")
    styled = pp.annotate_orf(seq, "gene", extent=(4, 19),
                             style_codons=["blue", "red"])
    styled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">styled: seq_length=22, num_states=1</em>
    TATA<span class="pp-xtag-light">&lt;gene&gt;</span><span class="pp-codon-a">ATG</span><span class="pp-codon-b">AAA</span><span class="pp-codon-a">TTT</span><span class="pp-codon-b">GGG</span><span class="pp-codon-a">CCC</span><span class="pp-xtag-light">&lt;/gene&gt;</span>TAA
    </div>

Chain with mutagenize_orf
~~~~~~~~~~~~~~~~~~~~~~~~~

After annotating the ORF, ``mutagenize_orf`` can look up the frame
automatically. Here every single-codon missense variant is enumerated with
the mutated codon highlighted in red.

.. code-block:: python

    seq  = pp.from_seq("TATAATGAAATTTGGGCCCTAA")
    orf  = pp.annotate_orf(seq, "gene", extent=(4, 19))
    muts = pp.mutagenize_orf(orf, region="gene", num_mutations=1,
                             style="red", mode="sequential")
    muts.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">muts: seq_length=22, num_states=95</em>
    TATA<span class="pp-xtag-light">&lt;gene&gt;</span><span class="pp-mut">TTC</span>AAATTTGGGCCC<span class="pp-xtag-light">&lt;/gene&gt;</span>TAA<br>
    TATA<span class="pp-xtag-light">&lt;gene&gt;</span><span class="pp-mut">CTG</span>AAATTTGGGCCC<span class="pp-xtag-light">&lt;/gene&gt;</span>TAA<br>
    TATA<span class="pp-xtag-light">&lt;gene&gt;</span><span class="pp-mut">ATC</span>AAATTTGGGCCC<span class="pp-xtag-light">&lt;/gene&gt;</span>TAA<br>
    TATA<span class="pp-xtag-light">&lt;gene&gt;</span><span class="pp-mut">GTG</span>AAATTTGGGCCC<span class="pp-xtag-light">&lt;/gene&gt;</span>TAA<br>
    TATA<span class="pp-xtag-light">&lt;gene&gt;</span><span class="pp-mut">AGC</span>AAATTTGGGCCC<span class="pp-xtag-light">&lt;/gene&gt;</span>TAA
    <span class="pp-ellipsis">... (95 total)</span>
    </div>

See :func:`~poolparty.annotate_orf`.
