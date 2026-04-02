deletion_scan
=============

Slide a deletion window of fixed length across the sequence (or a named
region) and, at each position, remove those bases. By default deleted
positions are filled with ``-`` gap characters so all output sequences remain
alignment-compatible. Pass ``deletion_marker=None`` to produce shorter
sequences instead.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 18 12 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pool``
     - ``Pool | str``
     - *(required)*
     - The Pool to scan. Can also be a plain sequence string.
   * - ``deletion_length``
     - ``int``
     - *(required)*
     - Width of the deletion window in bases. A sequence of length *L*
       produces *L* - ``deletion_length`` + 1 variants.
   * - ``deletion_marker``
     - ``str | None``
     - ``'-'``
     - Character used to fill deleted positions. Pass ``None`` to remove
       deleted bases entirely (output sequences are shorter than the input).
   * - ``region``
     - ``str | None``
     - ``None``
     - Name of a tagged region to restrict the scan to. Flanking sequences
       are never modified.
   * - ``positions``
     - ``list[int] | None``
     - ``None``
     - Explicit list of window start positions. ``None`` = all valid positions.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` iterates left-to-right; ``'random'`` shuffles.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Fix the total number of output states.
   * - ``style``
     - ``str | None``
     - ``None``
     - Named display style applied to the deletion marker. Only takes
       effect when ``deletion_marker`` is not ``None``.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.

----

Examples
--------

Single-base deletion with default marker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Delete one base at each of the 8 positions in an 8-mer; deleted positions
are marked with ``-``.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    dels = wt.deletion_scan(deletion_length=1, mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (8 sequences &mdash; one 1-base deletion per position)</em>
    <span class="pp-del">-</span>CGTACGT<br>
    A<span class="pp-del">-</span>GTACGT<br>
    AC<span class="pp-del">-</span>TACGT<br>
    ACG<span class="pp-del">-</span>ACGT<br>
    ACGT<span class="pp-del">-</span>CGT<br>
    ACGTA<span class="pp-del">-</span>GT<br>
    ACGTAC<span class="pp-del">-</span>T<br>
    ACGTACG<span class="pp-del">-</span>
    </div>

True deletion (deletion_marker=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``deletion_marker=None`` removes the bases entirely; output sequences are
shorter than the input.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    dels = wt.deletion_scan(deletion_length=2, deletion_marker=None, mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (7 sequences &mdash; 2 bases removed, output length 6)</em>
    GTACGT<br>
    ATACGT<br>
    ACACGT<br>
    ACGCGT<br>
    ACGTGT<br>
    ACGTAT<br>
    ACGTAC
    </div>

2-base window deletion
~~~~~~~~~~~~~~~~~~~~~~~~

Delete two consecutive bases at each position. An 8-mer yields 7 variants.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    dels = wt.deletion_scan(deletion_length=2, mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (7 sequences &mdash; 2-base deletion at each position)</em>
    <span class="pp-del">--</span>GTACGT<br>
    A<span class="pp-del">--</span>TACGT<br>
    AC<span class="pp-del">--</span>ACGT<br>
    ACG<span class="pp-del">--</span>CGT<br>
    ACGT<span class="pp-del">--</span>GT<br>
    ACGTA<span class="pp-del">--</span>T<br>
    ACGTAC<span class="pp-del">--</span>
    </div>

Deletion scan within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict the scan to the ``cre`` region; the ``AAAA`` and ``TTTT`` flanks
are always returned unchanged.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    dels = wt.deletion_scan(deletion_length=2, region="cre", mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (7 sequences &mdash; 2-base deletions within <em>cre</em> only)</em>
    AAAA<span class="pp-region"><span class="pp-del">--</span>CGATCG</span>TTTT<br>
    AAAA<span class="pp-region">A<span class="pp-del">--</span>GATCG</span>TTTT<br>
    AAAA<span class="pp-region">AT<span class="pp-del">--</span>ATCG</span>TTTT<br>
    AAAA<span class="pp-region">ATC<span class="pp-del">--</span>TCG</span>TTTT<br>
    AAAA<span class="pp-region">ATCG<span class="pp-del">--</span>CG</span>TTTT<br>
    AAAA<span class="pp-region">ATCGA<span class="pp-del">--</span>G</span>TTTT<br>
    AAAA<span class="pp-region">ATCGAT<span class="pp-del">--</span></span>TTTT
    </div>

Scan only specific positions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supply an explicit ``positions`` list to delete at chosen sites only.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    dels = wt.deletion_scan(deletion_length=1, positions=[1, 3, 5], mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (3 sequences &mdash; 1-base deletion at positions 1, 3, and 5 only)</em>
    A<span class="pp-del">-</span>GTACGT<br>
    ACG<span class="pp-del">-</span>ACGT<br>
    ACGTA<span class="pp-del">-</span>GT
    </div>

See :func:`~poolparty.deletion_scan`.
