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
   :widths: auto

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pool``
     - ``Pool | str``
     - *(required)*
     - The Pool to scan. Can also be a plain sequence string.
   * - ``tag_name``
     - ``str``
     - ``'region'``
     - Name for the scanning region tag inserted at each position.
   * - ``positions``
     - ``list[int] | None``
     - ``None``
     - Explicit list of 0-based positions to visit. ``None`` = all valid
       positions.
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
     - Enumeration order when combined with other pools.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.region_scan` in the
   :doc:`API Reference </api>`.

Examples
--------

Point tags across every inter-base position (region_length=0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``region_length=0`` places a zero-length tag at each of the 9 valid
positions in an 8-mer (between and at the ends of every base).

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    scan = pp.region_scan(wt, tag_name="ins", region_length=0,
                          mode="sequential")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=8, num_states=9</em>
    <span class="pp-xtag">&lt;ins/&gt;</span>ATCGATCG<br>
    A<span class="pp-xtag">&lt;ins/&gt;</span>TCGATCG<br>
    AT<span class="pp-xtag">&lt;ins/&gt;</span>CGATCG<br>
    ATC<span class="pp-xtag">&lt;ins/&gt;</span>GATCG<br>
    ATCG<span class="pp-xtag">&lt;ins/&gt;</span>ATCG<br>
    ATCGA<span class="pp-xtag">&lt;ins/&gt;</span>TCG<br>
    ATCGAT<span class="pp-xtag">&lt;ins/&gt;</span>CG<br>
    ATCGATC<span class="pp-xtag">&lt;ins/&gt;</span>G<br>
    ATCGATCG<span class="pp-xtag">&lt;ins/&gt;</span>
    </div>

2-base spanning window (region_length=2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A positive ``region_length`` creates spanning tags that enclose that many
bases at each scan position.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    scan = pp.region_scan(wt, tag_name="win", region_length=2,
                          mode="sequential")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=8, num_states=7</em>
    <span class="pp-xtag">&lt;win&gt;</span>AT<span class="pp-xtag">&lt;/win&gt;</span>CGATCG<br>
    A<span class="pp-xtag">&lt;win&gt;</span>TC<span class="pp-xtag">&lt;/win&gt;</span>GATCG<br>
    AT<span class="pp-xtag">&lt;win&gt;</span>CG<span class="pp-xtag">&lt;/win&gt;</span>ATCG<br>
    ATC<span class="pp-xtag">&lt;win&gt;</span>GA<span class="pp-xtag">&lt;/win&gt;</span>TCG<br>
    ATCG<span class="pp-xtag">&lt;win&gt;</span>AT<span class="pp-xtag">&lt;/win&gt;</span>CG<br>
    ATCGA<span class="pp-xtag">&lt;win&gt;</span>TC<span class="pp-xtag">&lt;/win&gt;</span>G<br>
    ATCGAT<span class="pp-xtag">&lt;win&gt;</span>CG<span class="pp-xtag">&lt;/win&gt;</span>
    </div>

Scan constrained to a named region (region)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict the window scan to the ``cre`` region; flanks are fixed.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    scan = pp.region_scan(wt, tag_name="win", region_length=2,
                          region="cre", mode="sequential")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=16, num_states=7</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><span class="pp-xtag">&lt;win&gt;</span>AT<span class="pp-xtag">&lt;/win&gt;</span>CGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>A<span class="pp-xtag">&lt;win&gt;</span>TC<span class="pp-xtag">&lt;/win&gt;</span>GATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>AT<span class="pp-xtag">&lt;win&gt;</span>CG<span class="pp-xtag">&lt;/win&gt;</span>ATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    <span class="pp-ellipsis">... (7 total)</span>
    </div>

Strip the constraint region tags (remove_tags=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``remove_tags=True`` removes the ``cre`` tags while keeping the scanning
window tag in the output.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    scan = pp.region_scan(wt, tag_name="win", region_length=2,
                          region="cre", remove_tags=True,
                          mode="sequential")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=16, num_states=7</em>
    AAAA<span class="pp-xtag">&lt;win&gt;</span>AT<span class="pp-xtag">&lt;/win&gt;</span>CGATCGTTTT<br>
    AAAAA<span class="pp-xtag">&lt;win&gt;</span>TC<span class="pp-xtag">&lt;/win&gt;</span>GATCGTTTT<br>
    AAAAAT<span class="pp-xtag">&lt;win&gt;</span>CG<span class="pp-xtag">&lt;/win&gt;</span>ATCGTTTT<br>
    <span class="pp-ellipsis">... (7 total)</span>
    </div>

Scan only specific positions (positions parameter)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass an explicit list to ``positions`` to visit only chosen window starts.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    scan = pp.region_scan(wt, tag_name="win", region_length=2,
                          positions=[0, 3, 6], mode="sequential")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=8, num_states=3</em>
    <span class="pp-xtag">&lt;win&gt;</span>AT<span class="pp-xtag">&lt;/win&gt;</span>CGATCG<br>
    ATC<span class="pp-xtag">&lt;win&gt;</span>GA<span class="pp-xtag">&lt;/win&gt;</span>TCG<br>
    ATCGAT<span class="pp-xtag">&lt;win&gt;</span>CG<span class="pp-xtag">&lt;/win&gt;</span>
    </div>

See :func:`~poolparty.region_scan`.
