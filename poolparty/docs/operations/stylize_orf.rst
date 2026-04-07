stylize_orf
===========

Apply codon-aware inline styling to sequences without modifying them. Pass
``style_codons=`` to cycle a list of styles across whole codons, or
``style_frames=`` (length a multiple of 3) to style each base position within
a codon independently. Use ``region=`` to restrict coloring to a named ORF
segment. Use ``mode='sequential'`` to preserve all upstream states.

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
     - The Pool or sequence string to style.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Restrict styling to a named ORF region or a ``[start, stop]``
       coordinate pair. The region must be an ``OrfRegion`` (created by
       ``annotate_orf``) or ``frame`` must be specified explicitly.
   * - ``style_codons``
     - ``list[str] | None``
     - ``None``
     - List of style names cycled across whole codons. Mutually exclusive
       with ``style_frames``.
   * - ``style_frames``
     - ``list[str] | None``
     - ``None``
     - List of style names (length a multiple of 3) applied per base
       position within each codon. Mutually exclusive with
       ``style_codons``.
   * - ``frame``
     - ``int | None``
     - ``None``
     - Reading frame (+1..+3 or -1..-3). Auto-detected from an
       ``OrfRegion`` if ``region`` is set; must be specified otherwise.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Iteration priority for downstream multi-pool iteration.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.stylize_orf` in the
   :doc:`API Reference </api>`.

Examples
--------

Alternating codon colours (style_codons)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cycle two colours across the 5 codons of ``ATGAAATTTGGGCCC`` so the reading
frame is immediately visible.

.. code-block:: python

    cds    = pp.from_seq("ATGAAATTTGGGCCC")
    styled = pp.stylize_orf(cds, style_codons=["blue", "red"])
    styled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">styled: seq_length=15, num_states=1</em>
    <span class="pp-codon-a">ATG</span><span class="pp-codon-b">AAA</span><span class="pp-codon-a">TTT</span><span class="pp-codon-b">GGG</span><span class="pp-codon-a">CCC</span>
    </div>

Per-base position styling (style_frames)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Colour each position within a codon independently — each of the three codon
positions gets its own colour — to highlight wobble positions.

.. code-block:: python

    cds    = pp.from_seq("ATGAAATTTGGGCCC")
    styled = pp.stylize_orf(cds, style_frames=["blue", "red", "green"])
    styled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">styled: seq_length=15, num_states=1</em>
    <span class="pp-codon-a">A</span><span class="pp-codon-b">T</span><span class="pp-codon-c">G</span><span class="pp-codon-a">A</span><span class="pp-codon-b">A</span><span class="pp-codon-c">A</span><span class="pp-codon-a">T</span><span class="pp-codon-b">T</span><span class="pp-codon-c">T</span><span class="pp-codon-a">G</span><span class="pp-codon-b">G</span><span class="pp-codon-c">G</span><span class="pp-codon-a">C</span><span class="pp-codon-b">C</span><span class="pp-codon-c">C</span>
    </div>

Restrict styling to a named ORF region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``annotate_orf`` to register the reading frame, then ``stylize_orf`` with
``region=`` to colour only the ORF while leaving flanking bases unstyled.

.. code-block:: python

    wt     = pp.from_seq("AAAA<cds>ATGAAATTTGGGCCC</cds>TTTT")
    orf    = pp.annotate_orf(wt, "cds")
    styled = pp.stylize_orf(orf, region="cds",
                             style_codons=["blue", "red"])
    styled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">styled: seq_length=23, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;cds&gt;</span><span class="pp-codon-a">ATG</span><span class="pp-codon-b">AAA</span><span class="pp-codon-a">TTT</span><span class="pp-codon-b">GGG</span><span class="pp-codon-a">CCC</span><span class="pp-xtag-cre">&lt;/cds&gt;</span>TTTT
    </div>

See :func:`~poolparty.stylize_orf`.
