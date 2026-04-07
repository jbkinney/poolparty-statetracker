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
   * - ``subseq_length``
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
     - Enumeration order when combined with other pools.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output. Available keys:
       ``"position_index"``, ``"start"``, ``"end"``, ``"name"``,
       ``"region_seq"``.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.subseq_scan` in the
   :doc:`API Reference </api>`.

Examples
--------

Extract all 4-mers from an 8-mer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A window of length 4 over an 8-base sequence yields 5 subsequences.

.. code-block:: python

    pool    = pp.from_seq("ACGTACGT")
    submers = pool.subseq_scan(subseq_length=4, mode="sequential")
    submers.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">submers: seq_length=4, num_states=5</em>
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
    submers = pool.subseq_scan(subseq_length=3, positions=[0, 3, 5],
                               mode="sequential")
    submers.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">submers: seq_length=3, num_states=3</em>
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
    submers = pool.subseq_scan(subseq_length=4, region="cre",
                               mode="sequential")
    submers.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">submers: seq_length=4, num_states=5</em>
    ATCG<br>
    TCGA<br>
    CGAT<br>
    GATC<br>
    ATCG
    </div>

Random subsequence sampling (mode="random")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode='random'`` draws window positions stochastically. Use ``num_states``
to control how many subsequences are sampled.

.. code-block:: python

    pool    = pp.from_seq("ACGTACGTACGT")
    submers = pool.subseq_scan(subseq_length=4, mode="random", num_states=5)
    submers.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">submers: seq_length=4, num_states=5</em>
    ACGT<br>
    GTAC<br>
    ACGT<br>
    CGTA<br>
    CGTA
    </div>

See :func:`~poolparty.subseq_scan`.
