deletion_multiscan
==================

Delete segments at multiple positions simultaneously, placing deletion
markers at each site. Deletion sites are guaranteed to be non-overlapping.
Use ``mode='sequential'`` to enumerate arrangements as separate states;
``mode='random'`` samples one arrangement per draw (default ``num_states``
is 1 unless you set it higher).

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Parameters
----------

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pool``
     - ``Pool | str``
     - *(required)*
     - Input pool or sequence string.
   * - ``deletion_length``
     - ``int``
     - *(required)*
     - Length of each deletion window.
   * - ``num_deletions``
     - ``int``
     - *(required)*
     - Number of simultaneous non-overlapping deletions per draw.
   * - ``deletion_marker``
     - ``str | None``
     - ``"-"``
     - String to place at each deletion site. ``None`` removes the bases
       outright, producing shorter output sequences.
   * - ``positions``
     - ``list | None``
     - ``None``
     - Allowed position sets for each deletion window. ``None`` allows any
       valid non-overlapping arrangement.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Named region or interval to restrict deletions to.
   * - ``names``
     - ``list[str] | None``
     - ``None``
     - Names for each deletion window.
   * - ``min_spacing``
     - ``int | None``
     - ``None``
     - Minimum gap (in bases) between deletion windows.
   * - ``max_spacing``
     - ``int | None``
     - ``None``
     - Maximum gap (in bases) between deletion windows.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.
   * - ``mode``
     - ``str``
     - ``"random"``
     - ``"random"`` or ``"sequential"``.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Number of states. ``None`` lets PoolParty choose automatically.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style for deletion markers.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Enumeration order when combined with other pools.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.deletion_multiscan` in the
   :doc:`API Reference </api>`.

Examples
--------

Two simultaneous single-base deletions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mark two randomly chosen single-base positions with the default ``"-"``
deletion marker.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCGATCG")
    scan = wt.deletion_multiscan(deletion_length=1, num_deletions=2,
                                 mode="random", style="grey")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=12, num_states=1</em>
    ATCGAT<span class="pp-del">-</span>GAT<span class="pp-del">-</span>G
    </div>

Two simultaneous 2-base deletions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Delete two non-overlapping dinucleotide windows per draw.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCGATCG")
    scan = wt.deletion_multiscan(deletion_length=2, num_deletions=2,
                                 mode="random", style="grey")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=12, num_states=1</em>
    ATCGA<span class="pp-del">--</span>GA<span class="pp-del">--</span>G
    </div>

Multiscan deletion within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict both deletion windows to positions inside the ``cre`` region,
keeping the flanking sequence intact.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    scan = wt.deletion_multiscan(deletion_length=1, num_deletions=2,
                                 region="cre", mode="random", style="grey")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=16, num_states=1</em>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ATCG<span class="pp-del">-</span>TC<span class="pp-del">-</span><span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT
    </div>

Spacing constraints (min_spacing, max_spacing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``min_spacing`` and ``max_spacing`` control the gap between deletion windows.
Two 3-base deletions must be 4–8 bases apart on a 24-mer.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCGATCGATCGATCGATCG")
    scan = wt.deletion_multiscan(deletion_length=3, num_deletions=2,
                                 min_spacing=4, max_spacing=8,
                                 mode="sequential", style="grey")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=24, num_states=65</em>
    <span class="pp-del">---</span>GATC<span class="pp-del">---</span>CGATCGATCGATCG<br>
    <span class="pp-del">---</span>GATCG<span class="pp-del">---</span>GATCGATCGATCG<br>
    <span class="pp-del">---</span>GATCGA<span class="pp-del">---</span>ATCGATCGATCG<br>
    <span class="pp-del">---</span>GATCGAT<span class="pp-del">---</span>TCGATCGATCG<br>
    <span class="pp-del">---</span>GATCGATC<span class="pp-del">---</span>CGATCGATCG
    <span class="pp-ellipsis">... (65 total)</span>
    </div>

Explicit position sets (positions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify allowed starts for each deletion window. Below, the first deletion
can start at 0, 3, or 6 and the second at 10 or 14.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCGATCGATCG")
    scan = wt.deletion_multiscan(deletion_length=2, num_deletions=2,
                                 positions=[[0, 3, 6], [10, 14]],
                                 mode="sequential", style="grey")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=16, num_states=6</em>
    <span class="pp-del">--</span>CGATCGAT<span class="pp-del">--</span>ATCG<br>
    <span class="pp-del">--</span>CGATCGATCGAT<span class="pp-del">--</span><br>
    ATC<span class="pp-del">--</span>TCGAT<span class="pp-del">--</span>ATCG<br>
    ATC<span class="pp-del">--</span>TCGATCGAT<span class="pp-del">--</span><br>
    ATCGAT<span class="pp-del">--</span>AT<span class="pp-del">--</span>ATCG<br>
    ATCGAT<span class="pp-del">--</span>ATCGAT<span class="pp-del">--</span>
    </div>

See :func:`~poolparty.deletion_multiscan`.
