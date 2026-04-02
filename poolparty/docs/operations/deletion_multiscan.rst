deletion_multiscan
==================

Delete segments at multiple positions simultaneously, placing deletion
markers at each site. Deletion sites are guaranteed to be non-overlapping.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Parameters
----------

.. list-table::
   :widths: 20 18 12 50
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
     - Iteration priority for downstream multi-pool iteration.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

----

Examples
--------

Two simultaneous single-base deletions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mark two randomly chosen single-base positions with the default ``"-"``
deletion marker.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCGATCG")
    scan = pp.deletion_multiscan(wt, deletion_length=1, num_deletions=2)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 2 simultaneous 1-base deletions per draw)</em>
    <span class="pp-del">-</span>TCGA<span class="pp-del">-</span>CGATCG<br>
    AT<span class="pp-del">-</span>GATCG<span class="pp-del">-</span>TCG<br>
    ATCG<span class="pp-del">-</span>TCGAT<span class="pp-del">-</span>G<br>
    <span class="pp-ellipsis">... each draw places 2 single-base deletions at distinct positions</span>
    </div>

Two simultaneous 2-base deletions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Delete two non-overlapping dinucleotide windows per draw.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCGATCG")
    scan = pp.deletion_multiscan(wt, deletion_length=2, num_deletions=2)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 2 simultaneous 2-base deletions per draw)</em>
    <span class="pp-del">--</span>CGAT<span class="pp-del">--</span>GATCG<br>
    AT<span class="pp-del">--</span>ATCG<span class="pp-del">--</span>CG<br>
    ATCG<span class="pp-del">--</span>CG<span class="pp-del">--</span>CG<br>
    <span class="pp-ellipsis">... each draw places 2 non-overlapping 2-base deletions</span>
    </div>

Multiscan deletion within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict both deletion windows to positions inside the ``cre`` region,
keeping the flanking sequence intact.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    scan = pp.deletion_multiscan(wt, deletion_length=1, num_deletions=2,
                                 positions=range(4, 12))

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 2 single-base deletions confined to <em>cre</em>)</em>
    AAAA<span class="pp-region"><span class="pp-del">-</span>TCGAT<span class="pp-del">-</span>G</span>TTTT<br>
    AAAA<span class="pp-region">A<span class="pp-del">-</span>CGAT<span class="pp-del">-</span>G</span>TTTT<br>
    AAAA<span class="pp-region">ATCG<span class="pp-del">-</span>TCG<span class="pp-del">-</span></span>TTTT<br>
    <span class="pp-ellipsis">... flanks always AAAA...TTTT; deletions inside cre only</span>
    </div>

See :func:`~poolparty.deletion_multiscan`.
