deletion_scan_orf
=================

Slide a whole-codon deletion window across an ORF. Positions are numbered in
**coding order**, so ``codon_positions=[0]`` always selects the first translated
codon. On a negative frame that codon is physically at the right-hand end of
the stored plus/reference sequence.

By default each deleted nucleotide is replaced by ``-``, preserving sequence
length. Pass ``deletion_marker=None`` for a true deletion.

.. code-block:: python

    import poolparty as pp
    pp.init()

    orf = pp.from_seq("GGATGAAATTTCC").annotate_orf(
        "orf", extent=(2, 11), frame=1
    )
    deletions = orf.deletion_scan_orf(
        deletion_codons=1,
        region="orf",
        mode="sequential",
    )

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
     - DNA pool containing the ORF.
   * - ``deletion_codons``
     - ``int``
     - *(required)*
     - Number of consecutive codons deleted in each state.
   * - ``deletion_marker``
     - ``str | None``
     - ``'-'``
     - One-character marker repeated across the deleted nucleotides. ``None``
       excises them.
   * - ``codon_positions``
     - ``list[int] | slice | None``
     - ``None``
     - Eligible window starts in 0-based coding-order codon units.
   * - ``region``
     - ``str | list[int] | None``
     - ``None``
     - Named ORF, ``[start, stop]`` interval, or the whole sequence.
   * - ``frame``
     - ``int | None``
     - ``None``
     - One of ``+1``, ``+2``, ``+3``, ``-1``, ``-2``, ``-3``. A named
       :class:`~poolparty.OrfRegion` supplies its stored frame when omitted;
       whole-sequence and interval calls otherwise default to ``+1``.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` enumerates eligible windows in coding order;
       ``'random'`` samples a window independently for each state.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Number of position states. ``None`` uses every eligible window in
       sequential mode and defaults to one randomized state in random mode.
   * - ``style``
     - ``str | None``
     - ``None``
     - Named display style applied to deletion markers. Ignored when
       ``deletion_marker=None``.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Enumeration order when combined with other stateful operations.
   * - ``cards``
     - ``list[str] | dict[str, str] | None``
     - ``None``
     - Design card keys: ``seq``, ``state``, ``codon_positions``,
       ``wt_codons``, ``wt_aas``, ``start``, and ``end``.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.deletion_scan_orf` in the
   :doc:`API Reference </api>`.

.. note:: Initial input scope

   This first version requires a fixed-length, ungapped ``ACGT`` target ORF.
   Nested region annotations inside that target are rejected; pass the outer
   ORF name through ``region=`` rather than scanning an annotated construct as
   an unscoped whole sequence. Broad IUPAC, gapped, and nested-annotation
   policies are deferred.

   A true deletion shortens sequence content but v2's Party-level named-region
   length metadata is immutable. Translation and random mutagenesis, which use
   runtime sequence geometry, still work. A second true ``deletion_scan_orf``
   or a sequential geometry-dependent ORF operation on that same named region
   is not yet supported. Whole-sequence true-deletion chains do not depend on
   named-region metadata. A marked deletion does not stale the registered
   length, but its gapped output remains outside this version's accepted input
   scope and cannot be fed into another ORF deletion scan.

Examples
--------

Scan one-codon deletions
~~~~~~~~~~~~~~~~~~~~~~~~

Scan each codon in a three-codon CDS. By default, the affected codon is
replaced with gap markers so every output remains the same length.

.. code-block:: python

    cds = pp.from_seq("ATGAAATTT")
    deletions = cds.deletion_scan_orf(
        deletion_codons=1,
        mode="sequential",
        style="grey",
    ).named("deletions")
    deletions.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">deletions: seq_length=9, num_states=3</em>
    <span class="pp-del">---</span>AAATTT<br>
    ATG<span class="pp-del">---</span>TTT<br>
    ATGAAA<span class="pp-del">---</span>
    </div>

Make a true deletion inside a named ORF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``deletion_marker=None`` to excise the selected codon. The surrounding
sequence and ORF tags are preserved.

.. code-block:: python

    construct = pp.from_seq("GGATGAAATTTCC").annotate_orf(
        "orf", extent=(2, 11), frame=1
    )
    deletion = construct.deletion_scan_orf(
        deletion_codons=1,
        deletion_marker=None,
        codon_positions=[1],
        region="orf",
        mode="sequential",
    ).named("deletion")
    deletion.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">deletion: seq_length=10, num_states=1</em>
    GG<span class="pp-xtag-light">&lt;orf&gt;</span>ATGTTT<span class="pp-xtag-light">&lt;/orf&gt;</span>CC
    </div>

Frames, orphans, and coordinates
--------------------------------

The absolute frame value skips zero, one, or two bases before the first
complete codon: from the left for positive frames and from the right for
negative frames. Any bases left outside the complete-codon grid are orphans;
they remain unchanged.

For a negative-frame ORF, PoolParty keeps the DNA in stored plus/reference
orientation and maps coding codons onto physical intervals:

.. code-block:: text

    stored plus/reference DNA:   AAA CCC GGG TTT
    physical interval:           0   3   6   9  12
    frame -1 coding codons:       3   2   1   0
    coding-oriented WT:          TTT GGG CCC AAA

Deleting coding codon 1 marks physical interval ``[6, 9)``. Its
``wt_codons`` card is ``('CCC',)`` because cards report coding orientation,
not the literal plus-strand substring ``GGG``. Its ``wt_aas`` card is
``('P',)``.

The physical card coordinates are always relative to the operation's input
region and describe the pre-edit sequence:

.. list-table::
   :header-rows: 1

   * - Card
     - Meaning
   * - ``codon_positions``
     - A tuple of affected 0-based codon indices in coding order.
   * - ``wt_codons``
     - WT codons in coding orientation, aligned with ``codon_positions``.
   * - ``wt_aas``
     - Amino acids encoded by ``wt_codons`` using the Party codon table selected
       when the Operation is created.
   * - ``start``
     - Inclusive physical plus/reference coordinate in the input region.
   * - ``end``
     - Exclusive physical plus/reference coordinate in the input region.

Reverse-complement normalization
--------------------------------

Negative-frame deletion follows this invariant for ``N`` in ``1, 2, 3``:

.. code-block:: text

    delete(S, frame=-N) == RC(delete(RC(S), frame=+N))

The two runs have identical coding cards. Their physical intervals mirror:
``[start, end)`` becomes ``[L - end, L - start)`` for input-region length
``L``.

See :func:`~poolparty.deletion_scan_orf`.
