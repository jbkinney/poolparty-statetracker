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
     - ``str | list | None``
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
       ``'random'`` samples them.
   * - ``cards``
     - ``list[str] | dict[str, str] | None``
     - ``None``
     - Opt into ``codon_positions``, ``wt_codons``, ``start``, and ``end``.

See :func:`~poolparty.deletion_scan_orf` for the remaining shared scan
controls, including ``num_states``, ``style``, ``prefix``, and ``iter_order``.

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
not the literal plus-strand substring ``GGG``.

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
