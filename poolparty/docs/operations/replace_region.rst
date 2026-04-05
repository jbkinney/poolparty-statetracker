replace_region
==============

Replace the entire content of a named region with sequences drawn from a
content pool. Every combination of background sequence and content sequence
is produced (Cartesian product). The region tags are removed in the output;
only the new content occupies that position.

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
     - The background Pool whose named region will be replaced.
   * - ``content_pool``
     - ``Pool``
     - *(required)*
     - Pool whose sequences replace the region content.
   * - ``region_name``
     - ``str``
     - *(required)*
     - Name of the region to replace (must exist in the background pool).
   * - ``rc``
     - ``bool``
     - ``False``
     - When ``True``, reverse-complement each content sequence before
       inserting it.
   * - ``sync``
     - ``bool``
     - ``False``
     - When ``True``, pair background and content states 1:1 (lockstep
       iteration) instead of taking the Cartesian product.
   * - ``keep_tags``
     - ``bool``
     - ``False``
     - When ``True``, preserve the region tags around the replaced content
       so the region can still be referenced by downstream operations.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.replace_region` in the
   :doc:`API Reference </api>`.

Examples
--------

Replace a region with all 4-mers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enumerate all 256 4-mers inside the ``cre`` region using
:func:`~poolparty.from_iupac`.

.. code-block:: python

    import poolparty as pp
    pp.init()

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    inserts = pp.from_iupac("NNNN", mode="sequential")
    library = pp.replace_region(wt, inserts, region_name="cre")
    library.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">library: seq_length=12, num_states=256</em>
    AAAAAAAATTTT<br>
    AAAAAAACTTTT<br>
    AAAAAAAGTTTT<br>
    AAAAAAATTTTT<br>
    AAAAAACATTTT<br>
    <span class="pp-ellipsis">... (256 total)</span>
    </div>

Replace with a small explicit set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supply :func:`~poolparty.from_seqs` to substitute only specific sequences.

.. code-block:: python

    import poolparty as pp
    pp.init()

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    inserts = pp.from_seqs(["AAA", "TTT", "CCC"], mode="sequential")
    library = pp.replace_region(wt, inserts, region_name="cre")
    library.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">library: seq_length=11, num_states=3</em>
    AAAAAAAATTTT<br>
    AAAATTTTTTT<br>
    AAAACCCTTTT
    </div>

Replace a zero-length point tag (pure insertion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the region is a zero-length point tag, ``replace_region`` inserts
without deleting any bases.

.. code-block:: python

    import poolparty as pp
    pp.init()

    wt      = pp.from_seq("AAAA<ins/>TTTT")
    inserts = pp.from_seqs(["GC", "AT"], mode="sequential")
    library = pp.replace_region(wt, inserts, region_name="ins")
    library.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">library: seq_length=10, num_states=2</em>
    AAAAGCTTTT<br>
    AAAAATTTTT
    </div>

Insert reverse-complemented content (rc=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rc=True`` reverse-complements each content sequence before insertion.

.. code-block:: python

    import poolparty as pp
    pp.init()

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    inserts = pp.from_seqs(["GCGC", "ATAT"], mode="sequential")
    library = pp.replace_region(wt, inserts, region_name="cre", rc=True)
    library.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">library: seq_length=12, num_states=2</em>
    AAAAGCGCTTTT<br>
    AAAAATATTTTT
    </div>

See :func:`~poolparty.replace_region`.
