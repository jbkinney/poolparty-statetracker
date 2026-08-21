insertion_scan_orf
==================

Insert coding-oriented whole codons into an ORF. With ``replace=False``
(the default), positions are boundaries between codons. With ``replace=True``,
the insert overwrites the same number of complete codons as its own length.

.. code-block:: python

    import poolparty as pp
    pp.init()

    orf = pp.from_seq("GGATGAAATTTCC").annotate_orf(
        "orf", extent=(2, 11), frame=1
    )
    stops = pp.from_seqs(["TAG", "TAA", "TGA"], mode="sequential")
    insertions = orf.insertion_scan_orf(
        stops,
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
     - DNA pool containing the target ORF.
   * - ``insertion_pool``
     - ``Pool | str``
     - *(required)*
     - Fixed-length, ungapped ``ACGT`` content in coding orientation. Its
       length must be a positive multiple of three.
   * - ``codon_positions``
     - ``list[int] | slice | None``
     - ``None``
     - Coding-order splice slots or overwrite-window starts.
   * - ``region``
     - ``str | list[int] | None``
     - ``None``
     - Named ORF, ``[start, stop]`` interval, or the whole sequence.
   * - ``frame``
     - ``int | None``
     - ``None``
     - One of ``+1``, ``+2``, ``+3``, ``-1``, ``-2``, ``-3``. A named
       :class:`~poolparty.OrfRegion` supplies its frame when omitted; other
       targets default to ``+1``.
   * - ``replace``
     - ``bool``
     - ``False``
     - Splice between codons when ``False``; overwrite whole codons when
       ``True``.
   * - ``style``
     - ``str | None``
     - ``None``
     - Named display style applied to the inserted content.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Name the combined position-by-insert state index.
   * - ``prefix_position``
     - ``str | None``
     - ``None``
     - Name the position-state index separately.
   * - ``prefix_insert``
     - ``str | None``
     - ``None``
     - Name the insertion-pool state index separately.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` enumerates configured position states;
       ``'random'`` samples a position independently for each state. Insert
       states retain the mode of ``insertion_pool``.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Number of position states. ``None`` uses every configured position in
       sequential mode and defaults to one randomized position state in
       random mode.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Enumeration order when combined with other stateful operations.
   * - ``cards``
     - ``list[str] | dict[str, str] | None``
     - ``None``
     - Universal cards: ``seq`` and ``state``. Splice cards:
       ``codon_slot``, ``start``, and ``end``. Overwrite cards:
       ``codon_positions``, ``wt_codons``, ``start``, and ``end``.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.insertion_scan_orf` in the
   :doc:`API Reference </api>`.

Examples
--------

Splice a stop codon at every coding boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A three-codon CDS has four splice slots: before the first codon, between
codons, and after the last codon.

.. code-block:: python

    cds = pp.from_seq("ATGAAATTT")
    insertions = cds.insertion_scan_orf(
        "TAG",
        mode="sequential",
        style="green",
    ).named("insertions")
    insertions.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">insertions: seq_length=12, num_states=4</em>
    <span class="pp-ins">TAG</span>ATGAAATTT<br>
    ATG<span class="pp-ins">TAG</span>AAATTT<br>
    ATGAAA<span class="pp-ins">TAG</span>TTT<br>
    ATGAAATTT<span class="pp-ins">TAG</span>
    </div>

Overwrite one codon at every position
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``replace=True``, the insert replaces the same number of complete codons
as its own length. Sequence length therefore stays unchanged.

.. code-block:: python

    cds = pp.from_seq("ATGAAATTT")
    replacements = cds.insertion_scan_orf(
        "TAG",
        replace=True,
        mode="sequential",
        style="green",
    ).named("replacements")
    replacements.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">replacements: seq_length=9, num_states=3</em>
    <span class="pp-ins">TAG</span>AAATTT<br>
    ATG<span class="pp-ins">TAG</span>TTT<br>
    ATGAAA<span class="pp-ins">TAG</span>
    </div>

Insert into a negative-frame ORF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The caller still supplies ``TAG`` in coding orientation. PoolParty writes its
reverse complement, ``CTA``, into the stored plus/reference sequence.

.. code-block:: python

    insertions = pp.insertion_scan_orf(
        "AAACCC",
        "TAG",
        frame=-1,
        mode="sequential",
        style="green",
    ).named("insertions")
    insertions.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">insertions: seq_length=9, num_states=3</em>
    AAACCC<span class="pp-ins">CTA</span><br>
    AAA<span class="pp-ins">CTA</span>CCC<br>
    <span class="pp-ins">CTA</span>AAACCC
    </div>

Splice slots and overwrite windows
----------------------------------

For three complete codons, splice mode has four slots:

.. code-block:: text

    slot 0   codon 0   slot 1   codon 1   slot 2   codon 2   slot 3
       |       AAA        |       CCC        |       GGG        |

An insert containing two codons still has four splice slots. In overwrite
mode, that same insert has two eligible windows: codons ``(0, 1)`` and
``(1, 2)``.

Frames and coding orientation
-----------------------------

The insert is always supplied in coding orientation. For a negative frame,
PoolParty reverse-complements the **entire selected insert once** before
writing it into the stored plus/reference sequence. This reverses codon order
for multi-codon inserts as required.

.. code-block:: text

    coding ORF:                  AAA | CCC
    stored negative-frame DNA:  GGG | TTT

    insert coding TAG at slot 1

    coding result:               AAA | TAG | CCC
    stored physical result:      GGG | CTA | TTT
                                       ^^^
                                       RC(TAG)

Supplying coding insert ``ATGGAA`` writes physical ``TTCCAT`` on a negative
frame—not the per-codon reverse complements ``CATTTC``.

Cards and insertion provenance
------------------------------

Coding cards and physical cards deliberately use different coordinate views:

.. list-table::
   :header-rows: 1

   * - Card
     - Meaning
   * - ``codon_slot``
     - Splice boundary in coding order, from ``0`` through ``n_codons``.
   * - ``codon_positions``
     - Tuple of overwritten 0-based codon indices in coding order.
   * - ``wt_codons``
     - Overwritten WT codons in coding orientation.
   * - ``start``, ``end``
     - Half-open physical plus/reference coordinates relative to the input
       region. A splice has ``start == end``.

The insertion pool owns the inserted sequence and provenance cards. The ORF
wrapper does not duplicate them:

.. code-block:: python

    inserts = pp.from_seqs(
        ["TAG", "TAA", "TGA"],
        mode="sequential",
        cards={"seq": "coding_insert", "seq_index": "insert_index"},
    )
    library = orf.insertion_scan_orf(
        inserts,
        region="orf",
        mode="sequential",
        cards={"codon_slot": "slot"},
    )

On a negative frame, ``coding_insert`` remains ``TAG`` even though the stored
physical output contains ``CTA``. The final state space is the product of
background states, configured position states, and insertion-pool states. In
sequential mode, the default position states enumerate every eligible slot or
window; random mode defaults to one randomized position state.

Input scope and named-region lengths
------------------------------------

This first version requires fixed-length, ungapped ``ACGT`` target and insert
states. Nested annotations inside the target and tagged insertion content are
rejected. Broad IUPAC, gapped, and annotation-overlap policies are deferred.

A splice lengthens sequence content, while v2's Party-level named-region
length metadata is immutable. Translation and other operations that determine
geometry from the runtime sequence can still work, but a subsequent sequential
geometry-dependent ORF operation on that same named region is not yet
supported. Overwrite mode preserves the registered length and does not have
this limitation.

Reverse-complement normalization
--------------------------------

For ``N`` in ``1, 2, 3`` and the same coding-oriented insert:

.. code-block:: text

    insert(S, frame=-N) == RC(insert(RC(S), frame=+N))

Coding cards are identical. Physical intervals mirror as
``[start, end)`` to ``[L - end, L - start)`` for input-region length ``L``;
for a splice point ``p``, the mirror is ``L - p``.

See :func:`~poolparty.insertion_scan_orf`.
