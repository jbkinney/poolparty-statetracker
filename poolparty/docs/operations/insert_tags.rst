insert_tags
===========

Insert XML-style region tags at a fixed position in a sequence, creating a
named segment that downstream operations can target by name. ``start`` and
``stop`` are 0-based indices into the underlying (non-tag) characters.
Omitting ``stop`` creates a zero-length *point* tag at ``start``.

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
     - The Pool to insert tags into. Can also be a plain sequence string.
   * - ``region_name``
     - ``str``
     - *(required)*
     - Name for the region (e.g. ``'cre'``, ``'orf'``).
   * - ``start``
     - ``int``
     - *(required)*
     - 0-based start index, counting only non-tag characters.
   * - ``stop``
     - ``int | None``
     - ``None``
     - Exclusive end index (non-tag characters). When ``None`` a
       zero-length point tag is created at ``start``.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.

----

Examples
--------

Tag a central region
~~~~~~~~~~~~~~~~~~~~

Mark positions 4–12 of a 16-mer as the ``cre`` region.

.. code-block:: python

    wt     = pp.from_seq("AAAAATCGATCGTTTT")
    tagged = pp.insert_tags(wt, "cre", start=4, stop=12)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; <em>cre</em> tagged at positions 4&ndash;12)</em>
    AAAA<span class="pp-region">ATCGATCG</span>TTTT
    </div>

Zero-length point tag (stop omitted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Omit ``stop`` to place a self-closing tag at position 4 without enclosing
any bases — useful as a landmark for :func:`~poolparty.replace_region`.

.. code-block:: python

    wt    = pp.from_seq("AAAAATCGATCGTTTT")
    point = pp.insert_tags(wt, "ins", start=4)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; zero-length point tag at position 4)</em>
    AAAA<span class="pp-region"></span>ATCGATCGTTTT
    </div>

Two non-overlapping regions in the same sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply ``insert_tags`` twice to define ``left`` and ``right`` regions.

.. code-block:: python

    wt    = pp.from_seq("AAAAATCGGGGGCCCTTTT")
    step1 = pp.insert_tags(wt,    "left",  start=4,  stop=8)
    step2 = pp.insert_tags(step1, "right", start=13, stop=17)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; two independent named regions)</em>
    AAAA<span class="pp-region">ATCG</span>GGGGG<span class="pp-region">CCCT</span>TTT
    </div>

Chain into mutagenize to restrict mutations to the tagged region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tag a region then mutagenize only that segment; the flanks stay intact.

.. code-block:: python

    wt      = pp.from_seq("AAAAATCGATCGTTTT")
    tagged  = pp.insert_tags(wt, "cre", start=4, stop=12)
    mutants = pp.mutagenize(tagged, num_mutations=1, region="cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 1 mutation per draw, restricted to <em>cre</em>)</em>
    AAAA<span class="pp-region">A<span class="pp-mut">G</span>CGATCG</span>TTTT<br>
    AAAA<span class="pp-region">ATCG<span class="pp-mut">C</span>TCG</span>TTTT<br>
    AAAA<span class="pp-region">ATCGAT<span class="pp-mut">A</span>G</span>TTTT<br>
    <span class="pp-ellipsis">... (stochastic; flanks always unchanged)</span>
    </div>

See :func:`~poolparty.insert_tags`.
