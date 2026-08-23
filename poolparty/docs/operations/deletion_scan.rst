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
   :widths: auto

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
     - ``str | list | None``
     - ``None``
     - Restrict the scan to a named region or ``[start, stop]`` interval.
       Flanking sequences are never modified.
   * - ``positions``
     - ``list[int] | slice | None``
     - ``None``
     - Window start positions in nucleotide units. ``None`` selects all valid
       positions; a ``slice`` selects from that range.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` iterates left-to-right; ``'random'`` shuffles.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Number of output states. ``None`` auto-computes in sequential mode
       or defaults to 1 in random mode.
   * - ``style``
     - ``str | None``
     - ``None``
     - Named display style applied to the deletion marker. Only takes
       effect when ``deletion_marker`` is not ``None``.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Enumeration order when combined with other pools.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.deletion_scan` in the
   :doc:`API Reference </api>`.

.. warning::

   ``deletion_scan`` uses nucleotide positions and can start a deletion inside
   a codon. For whole-codon edits, use :doc:`deletion_scan_orf`; it understands
   reading frames, coding orientation, and incomplete orphan bases.

Examples
--------

Single-base deletion with default marker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Delete one base at each of the 8 positions in an 8-mer; deleted positions
are marked with ``-``.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    dels = wt.deletion_scan(deletion_length=1, mode="sequential", style="grey")
    dels.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">dels: seq_length=8, num_states=8</em>
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
    dels.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">dels: seq_length=6, num_states=7</em>
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
    dels = wt.deletion_scan(deletion_length=2, mode="sequential", style="grey")
    dels.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">dels: seq_length=8, num_states=7</em>
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
    dels = wt.deletion_scan(deletion_length=2, region="cre", mode="sequential",
                            style="grey")
    dels.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">dels: seq_length=16, num_states=7</em>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span><span class="pp-del">--</span>CGATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>A<span class="pp-del">--</span>GATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>AT<span class="pp-del">--</span>ATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ATC<span class="pp-del">--</span>TCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ATCG<span class="pp-del">--</span>CG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ATCGA<span class="pp-del">--</span>G<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ATCGAT<span class="pp-del">--</span><span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT
    </div>

Scan only specific positions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supply an explicit ``positions`` list to delete at chosen sites only.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    dels = wt.deletion_scan(deletion_length=1, positions=[1, 3, 5], mode="sequential",
                            style="grey")
    dels.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">dels: seq_length=8, num_states=3</em>
    A<span class="pp-del">-</span>GTACGT<br>
    ACG<span class="pp-del">-</span>ACGT<br>
    ACGTA<span class="pp-del">-</span>GT
    </div>

Random deletion positions (mode="random")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode='random'`` draws deletion positions stochastically. Here a 3-base
deletion window samples 5 random positions along a 12-mer.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGTACGT")
    dels = wt.deletion_scan(deletion_length=3, mode="random", num_states=5,
                            style="grey")
    dels.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">dels: seq_length=12, num_states=5</em>
    ACGTA<span class="pp-del">---</span>ACGT<br>
    ACGTAC<span class="pp-del">---</span>CGT<br>
    ACGTA<span class="pp-del">---</span>ACGT<br>
    A<span class="pp-del">---</span>ACGTACGT<br>
    A<span class="pp-del">---</span>ACGTACGT
    </div>

See :func:`~poolparty.deletion_scan`.
