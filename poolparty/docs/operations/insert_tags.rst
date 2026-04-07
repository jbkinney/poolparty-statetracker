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
   :widths: auto

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
     - Enumeration order when combined with other pools.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.insert_tags` in the
   :doc:`API Reference </api>`.

Examples
--------

Tag a central region
~~~~~~~~~~~~~~~~~~~~

Mark positions 4–12 of a 16-mer as the ``cre`` region.

.. code-block:: python

    wt     = pp.from_seq("AAAAATCGATCGTTTT")
    tagged = pp.insert_tags(wt, "cre", start=4, stop=12)
    tagged.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">tagged: seq_length=16, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>

Zero-length point tag (stop omitted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Omit ``stop`` to place a self-closing tag at position 4 without enclosing
any bases — useful as a landmark for :func:`~poolparty.replace_region`.

.. code-block:: python

    wt    = pp.from_seq("AAAAATCGATCGTTTT")
    point = pp.insert_tags(wt, "ins", start=4)
    point.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">point: seq_length=16, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;ins/&gt;</span>ATCGATCGTTTT
    </div>

Two non-overlapping regions in the same sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply ``insert_tags`` twice to define ``left`` and ``right`` regions.

.. code-block:: python

    wt    = pp.from_seq("AAAAATCGGGGGCCCTTTT")
    step1 = pp.insert_tags(wt,    "left",  start=4,  stop=8)
    step2 = pp.insert_tags(step1, "right", start=13, stop=17)
    step2.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">step2: seq_length=19, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;left&gt;</span>ATCG<span class="pp-xtag-cre">&lt;/left&gt;</span>GGGGC<span class="pp-xtag-cre">&lt;right&gt;</span>CCTT<span class="pp-xtag-cre">&lt;/right&gt;</span>TT
    </div>

Chain into mutagenize to restrict mutations to the tagged region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tag a region then mutagenize only that segment; the flanks stay intact.
Use ``mode="sequential"`` for deterministic ordering of single mutants.

.. code-block:: python

    wt      = pp.from_seq("AAAAATCGATCGTTTT")
    tagged  = pp.insert_tags(wt, "cre", start=4, stop=12)
    mutants = pp.mutagenize(tagged, num_mutations=1, region="cre", mode="sequential")
    mutants.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mutants: seq_length=16, num_states=24</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>CTCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>GTCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>TTCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>AACGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ACCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    <span class="pp-ellipsis">... (24 total)</span>
    </div>

See :func:`~poolparty.insert_tags`.
