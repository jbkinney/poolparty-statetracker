region_scan
===========

Insert a named region tag at successive positions within a sequence,
producing one variant per window position. Combine with
:func:`~poolparty.replace_region` or other operations to act on the tagged
window. Use ``mode='sequential'`` to enumerate every position
deterministically.

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
   * - ``region``
     - ``str``
     - ``'region'``
     - Name for the scanning region tag inserted at each position.
   * - ``positions``
     - ``list[int] | None``
     - ``None``
     - Explicit list of 0-based positions to visit. ``None`` = all valid
       positions.
   * - ``region_constraint``
     - ``str | list | None``
     - ``None``
     - Restrict the scan to a named region (string) or a
       ``[start, stop]`` coordinate pair.
   * - ``remove_tags``
     - ``bool | None``
     - ``None``
     - When ``True`` and ``region_constraint`` is a region name, strip the
       constraint region tags from the output.
   * - ``region_length``
     - ``int``
     - ``0``
     - Number of bases the scanning window spans. ``0`` = zero-length
       point tag; positive = spanning tags enclosing that many bases.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` enumerates every valid position as a separate
       state; ``'random'`` samples one position per draw.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Override the automatically-computed state count.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

----

Examples
--------

Point tags across every inter-base position (region_length=0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``region_length=0`` places a zero-length tag at each of the 9 valid
positions in an 8-mer (between and at the ends of every base).

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    scan = pp.region_scan(wt, region="ins", region_length=0,
                          mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (9 sequences &mdash; point tag at each inter-base position)</em>
    <span class="pp-region"></span>ATCGATCG<br>
    A<span class="pp-region"></span>TCGATCG<br>
    AT<span class="pp-region"></span>CGATCG<br>
    ATC<span class="pp-region"></span>GATCG<br>
    ATCG<span class="pp-region"></span>ATCG<br>
    ATCGA<span class="pp-region"></span>TCG<br>
    ATCGAT<span class="pp-region"></span>CG<br>
    ATCGATC<span class="pp-region"></span>G<br>
    ATCGATCG<span class="pp-region"></span>
    </div>

2-base spanning window (region_length=2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A positive ``region_length`` creates spanning tags that enclose that many
bases at each scan position.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    scan = pp.region_scan(wt, region="win", region_length=2,
                          mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (7 sequences &mdash; 2-base window at each position)</em>
    <span class="pp-region">AT</span>CGATCG<br>
    A<span class="pp-region">TC</span>GATCG<br>
    AT<span class="pp-region">CG</span>ATCG<br>
    ATC<span class="pp-region">GA</span>TCG<br>
    ATCG<span class="pp-region">AT</span>CG<br>
    ATCGA<span class="pp-region">TC</span>G<br>
    ATCGAT<span class="pp-region">CG</span>
    </div>

Scan constrained to a named region (region_constraint)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict the window scan to the ``cre`` region; flanks are fixed.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    scan = pp.region_scan(wt, region="win", region_length=2,
                          region_constraint="cre", mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (7 sequences &mdash; 2-base window confined to <em>cre</em>; flanks fixed)</em>
    AAAA<span class="pp-region">AT</span>CGATCGTTTT<br>
    AAAAA<span class="pp-region">TC</span>GATCGTTTT<br>
    AAAAATCG<span class="pp-region">AT</span>CGTTTT<br>
    <span class="pp-ellipsis">... (7 total)</span>
    </div>

Strip the constraint region tags (remove_tags=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``remove_tags=True`` removes the ``cre`` tags while keeping the scanning
window tag in the output.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    scan = pp.region_scan(wt, region="win", region_length=2,
                          region_constraint="cre", remove_tags=True,
                          mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (7 sequences &mdash; <em>cre</em> tags stripped; <em>win</em> tag present)</em>
    AAAA<span class="pp-region">AT</span>CGATCGTTTT<br>
    AAAAA<span class="pp-region">TC</span>GATCGTTTT<br>
    AAAAATCGAT<span class="pp-region">CG</span>TTTT<br>
    <span class="pp-ellipsis">... (7 total)</span>
    </div>

Scan only specific positions (positions parameter)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass an explicit list to ``positions`` to visit only chosen window starts.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    scan = pp.region_scan(wt, region="win", region_length=2,
                          positions=[0, 3, 6], mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (3 sequences &mdash; window at positions 0, 3, and 6 only)</em>
    <span class="pp-region">AT</span>CGATCG<br>
    ATC<span class="pp-region">GA</span>TCG<br>
    ATCGAT<span class="pp-region">CG</span>
    </div>

See :func:`~poolparty.region_scan`.
