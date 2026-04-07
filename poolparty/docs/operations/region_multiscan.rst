region_multiscan
================

Insert multiple named region tags into a sequence simultaneously,
producing one variant per combination of non-overlapping tag positions.
Use ``mode='sequential'`` to enumerate every valid arrangement
deterministically; ``mode='random'`` samples one arrangement per draw.

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
   * - ``tag_names``
     - ``list[str] | str``
     - *(required)*
     - Tag name(s) to insert. A single string is reused for every
       insertion; a list assigns one name per insertion.
   * - ``num_insertions``
     - ``int``
     - *(required)*
     - Number of region tags to insert simultaneously.
   * - ``positions``
     - ``list | list[list] | None``
     - ``None``
     - Valid insertion positions (0-based, nontag-relative). A flat
       list or ``None`` shares positions across all insertions; a
       list-of-lists assigns per-insertion positions.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Restrict the scan to a named region (string) or a
       ``[start, stop]`` coordinate pair.
   * - ``remove_tags``
     - ``bool | None``
     - ``None``
     - When ``True`` and ``region`` is a region name, strip the
       constraint region tags from the output.
   * - ``region_length``
     - ``int | list[int]``
     - ``0``
     - Number of bases each scanning window spans. ``0`` = zero-length
       point tag. A single int applies to all insertions; a list
       assigns per-insertion lengths.
   * - ``insertion_mode``
     - ``str``
     - ``'ordered'``
     - ``'ordered'`` assigns tag_names left-to-right to positions;
       ``'unordered'`` enumerates all valid tag-to-position
       assignments.
   * - ``min_spacing``
     - ``int | None``
     - ``None``
     - Minimum gap (in bases) between adjacent region tags.
   * - ``max_spacing``
     - ``int | None``
     - ``None``
     - Maximum gap (in bases) between adjacent region tags.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` enumerates every valid arrangement as a
       separate state; ``'random'`` samples one per draw.
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
     - Design card columns to include in library output.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.region_multiscan` in the
   :doc:`API Reference </api>`.

Examples
--------

Two simultaneous point tags (region_length=0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``region_length=0`` places two zero-length tags at every valid pair of
positions in an 8-mer. With ``insertion_mode='ordered'`` (the default),
``a`` always appears to the left of ``b``.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    scan = pp.region_multiscan(wt, tag_names=["a", "b"],
                               num_insertions=2, region_length=0,
                               mode="sequential")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=8, num_states=36</em>
    <span class="pp-xtag">&lt;a/&gt;</span>A<span class="pp-xtag-b">&lt;b/&gt;</span>TCGATCG<br>
    <span class="pp-xtag">&lt;a/&gt;</span>AT<span class="pp-xtag-b">&lt;b/&gt;</span>CGATCG<br>
    <span class="pp-xtag">&lt;a/&gt;</span>ATC<span class="pp-xtag-b">&lt;b/&gt;</span>GATCG<br>
    <span class="pp-xtag">&lt;a/&gt;</span>ATCG<span class="pp-xtag-b">&lt;b/&gt;</span>ATCG<br>
    <span class="pp-xtag">&lt;a/&gt;</span>ATCGA<span class="pp-xtag-b">&lt;b/&gt;</span>TCG<br>
    <span class="pp-ellipsis">... (36 total)</span>
    </div>

Spanning tags with minimum spacing (min_spacing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``min_spacing`` enforces a minimum gap between the end of one tagged
region and the start of the next. Here two 2-base windows must be
separated by at least 2 bases, reducing the state count from 15 to 6.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    scan = pp.region_multiscan(wt, tag_names=["a", "b"],
                               num_insertions=2, region_length=2,
                               min_spacing=2, mode="sequential")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=8, num_states=6</em>
    <span class="pp-xtag">&lt;a&gt;</span>AT<span class="pp-xtag">&lt;/a&gt;</span>CG<span class="pp-xtag-b">&lt;b&gt;</span>AT<span class="pp-xtag-b">&lt;/b&gt;</span>CG<br>
    <span class="pp-xtag">&lt;a&gt;</span>AT<span class="pp-xtag">&lt;/a&gt;</span>CGA<span class="pp-xtag-b">&lt;b&gt;</span>TC<span class="pp-xtag-b">&lt;/b&gt;</span>G<br>
    <span class="pp-xtag">&lt;a&gt;</span>AT<span class="pp-xtag">&lt;/a&gt;</span>CGAT<span class="pp-xtag-b">&lt;b&gt;</span>CG<span class="pp-xtag-b">&lt;/b&gt;</span><br>
    A<span class="pp-xtag">&lt;a&gt;</span>TC<span class="pp-xtag">&lt;/a&gt;</span>GA<span class="pp-xtag-b">&lt;b&gt;</span>TC<span class="pp-xtag-b">&lt;/b&gt;</span>G<br>
    A<span class="pp-xtag">&lt;a&gt;</span>TC<span class="pp-xtag">&lt;/a&gt;</span>GAT<span class="pp-xtag-b">&lt;b&gt;</span>CG<span class="pp-xtag-b">&lt;/b&gt;</span><br>
    AT<span class="pp-xtag">&lt;a&gt;</span>CG<span class="pp-xtag">&lt;/a&gt;</span>AT<span class="pp-xtag-b">&lt;b&gt;</span>CG<span class="pp-xtag-b">&lt;/b&gt;</span>
    </div>

Unordered insertion mode (insertion_mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default (``insertion_mode='ordered'``), ``a`` is always placed to the
left of ``b``. Setting ``insertion_mode='unordered'`` allows either tag
to appear at any position, doubling the state count (10 ordered vs 20
unordered on a 4-mer).

.. code-block:: python

    wt   = pp.from_seq("ATCG")
    scan = pp.region_multiscan(wt, tag_names=["a", "b"],
                               num_insertions=2, region_length=0,
                               insertion_mode="unordered",
                               mode="sequential")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=4, num_states=20</em>
    <span class="pp-xtag">&lt;a/&gt;</span>A<span class="pp-xtag-b">&lt;b/&gt;</span>TCG<br>
    <span class="pp-xtag">&lt;a/&gt;</span>AT<span class="pp-xtag-b">&lt;b/&gt;</span>CG<br>
    <span class="pp-xtag">&lt;a/&gt;</span>ATC<span class="pp-xtag-b">&lt;b/&gt;</span>G<br>
    <span class="pp-xtag">&lt;a/&gt;</span>ATCG<span class="pp-xtag-b">&lt;b/&gt;</span><br>
    <span class="pp-xtag-b">&lt;b/&gt;</span>A<span class="pp-xtag">&lt;a/&gt;</span>TCG<br>
    A<span class="pp-xtag">&lt;a/&gt;</span>T<span class="pp-xtag-b">&lt;b/&gt;</span>CG<br>
    A<span class="pp-xtag">&lt;a/&gt;</span>TC<span class="pp-xtag-b">&lt;b/&gt;</span>G<br>
    A<span class="pp-xtag">&lt;a/&gt;</span>TCG<span class="pp-xtag-b">&lt;b/&gt;</span><br>
    <span class="pp-xtag-b">&lt;b/&gt;</span>AT<span class="pp-xtag">&lt;a/&gt;</span>CG<br>
    A<span class="pp-xtag-b">&lt;b/&gt;</span>T<span class="pp-xtag">&lt;a/&gt;</span>CG<br>
    AT<span class="pp-xtag">&lt;a/&gt;</span>C<span class="pp-xtag-b">&lt;b/&gt;</span>G<br>
    AT<span class="pp-xtag">&lt;a/&gt;</span>CG<span class="pp-xtag-b">&lt;b/&gt;</span><br>
    <span class="pp-xtag-b">&lt;b/&gt;</span>ATC<span class="pp-xtag">&lt;a/&gt;</span>G<br>
    A<span class="pp-xtag-b">&lt;b/&gt;</span>TC<span class="pp-xtag">&lt;a/&gt;</span>G<br>
    AT<span class="pp-xtag-b">&lt;b/&gt;</span>C<span class="pp-xtag">&lt;a/&gt;</span>G<br>
    ATC<span class="pp-xtag">&lt;a/&gt;</span>G<span class="pp-xtag-b">&lt;b/&gt;</span><br>
    <span class="pp-xtag-b">&lt;b/&gt;</span>ATCG<span class="pp-xtag">&lt;a/&gt;</span><br>
    A<span class="pp-xtag-b">&lt;b/&gt;</span>TCG<span class="pp-xtag">&lt;a/&gt;</span><br>
    AT<span class="pp-xtag-b">&lt;b/&gt;</span>CG<span class="pp-xtag">&lt;a/&gt;</span><br>
    ATC<span class="pp-xtag-b">&lt;b/&gt;</span>G<span class="pp-xtag">&lt;a/&gt;</span>
    </div>

Multiscan within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict both tag insertions to within the ``cre`` region; flanking
bases are never modified.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    scan = pp.region_multiscan(wt, tag_names=["a", "b"],
                               num_insertions=2, region_length=0,
                               region="cre", mode="sequential")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=16, num_states=36</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><span class="pp-xtag">&lt;a/&gt;</span>A<span class="pp-xtag-b">&lt;b/&gt;</span>TCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><span class="pp-xtag">&lt;a/&gt;</span>AT<span class="pp-xtag-b">&lt;b/&gt;</span>CGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><span class="pp-xtag">&lt;a/&gt;</span>ATC<span class="pp-xtag-b">&lt;b/&gt;</span>GATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><span class="pp-xtag">&lt;a/&gt;</span>ATCG<span class="pp-xtag-b">&lt;b/&gt;</span>ATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><span class="pp-xtag">&lt;a/&gt;</span>ATCGA<span class="pp-xtag-b">&lt;b/&gt;</span>TCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    <span class="pp-ellipsis">... (36 total)</span>
    </div>

See :func:`~poolparty.region_multiscan`.
