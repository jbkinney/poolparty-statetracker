subseq_scan
===========

Slide a window across the sequence and extract the subsequence at each
position. Unlike ``deletion_scan`` (which removes the window) or
``replacement_scan`` (which replaces it), ``subseq_scan`` returns only the
window content — producing a pool of short subsequences tiling across the
input.

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
   * - ``seq_length``
     - ``int``
     - *(required)*
     - Length of the subsequence window to extract at each position.
   * - ``positions``
     - ``list[int] | None``
     - ``None``
     - Explicit window start positions. ``None`` uses all valid positions.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Restrict the scan to a named region or ``[start, stop]`` interval.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.
   * - ``mode``
     - ``str``
     - ``"random"``
     - ``"sequential"`` iterates positions left-to-right;
       ``"random"`` samples one position per draw.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Override the automatically-computed state count.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Iteration priority for downstream multi-pool iteration.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output. Available keys:
       ``"position_index"``, ``"start"``, ``"end"``, ``"name"``,
       ``"region_seq"``.

----

Examples
--------

Extract all 4-mers from an 8-mer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A window of length 4 over an 8-base sequence yields 5 subsequences.

.. code-block:: python

    pool    = pp.from_seq("ACGTACGT")
    submers = pp.subseq_scan(pool, seq_length=4, mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (5 sequences &mdash; 4-base sliding window)</em>
    ACGT<br>
    CGTA<br>
    GTAC<br>
    TACG<br>
    ACGT
    </div>

Extract at specific positions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supply ``positions`` to extract from chosen sites only.

.. code-block:: python

    pool    = pp.from_seq("ACGTACGT")
    submers = pp.subseq_scan(pool, seq_length=3, positions=[0, 3, 5],
                             mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (3 sequences &mdash; 3-base windows at positions 0, 3, 5)</em>
    ACG<br>
    TAC<br>
    CGT
    </div>

Tile within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict the scan to a tagged region; only bases inside the region are
considered.

.. code-block:: python

    pool    = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    submers = pp.subseq_scan(pool, seq_length=4, region="cre",
                             mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (5 sequences &mdash; 4-base windows within <em>cre</em>)</em>
    ATCG<br>
    TCGA<br>
    CGAT<br>
    GATC<br>
    ATCG
    </div>

See :func:`~poolparty.subseq_scan`.
