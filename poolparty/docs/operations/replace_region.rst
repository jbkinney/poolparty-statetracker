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
   :widths: 20 18 12 50

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

Examples
--------

Replace a region with all 4-mers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enumerate all 256 4-mers inside the ``cre`` region using
:func:`~poolparty.from_iupac`.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    inserts = pp.from_iupac("NNNN", mode="sequential")
    library = pp.replace_region(wt, inserts, region_name="cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (256 sequences &mdash; all 4-mers substituted into <em>cre</em>)</em>
    AAAA<span class="pp-region">AAAA</span>TTTT<br>
    AAAA<span class="pp-region">AAAC</span>TTTT<br>
    AAAA<span class="pp-region">AAAG</span>TTTT<br>
    AAAA<span class="pp-region">AAAT</span>TTTT<br>
    <span class="pp-ellipsis">... (256 total)</span>
    </div>

Replace with a small explicit set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supply :func:`~poolparty.from_seqs` to substitute only specific sequences.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    inserts = pp.from_seqs(["AAA", "TTT", "CCC"])
    library = pp.replace_region(wt, inserts, region_name="cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (3 sequences)</em>
    AAAA<span class="pp-region">AAA</span>TTTT<br>
    AAAA<span class="pp-region">TTT</span>TTTT<br>
    AAAA<span class="pp-region">CCC</span>TTTT
    </div>

Replace a zero-length point tag (pure insertion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the region is a zero-length point tag, ``replace_region`` inserts
without deleting any bases.

.. code-block:: python

    wt      = pp.from_seq("AAAA<ins/>TTTT")
    inserts = pp.from_seqs(["GC", "AT"])
    library = pp.replace_region(wt, inserts, region_name="ins")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (2 sequences &mdash; dinucleotide inserted at point tag)</em>
    AAAA<span class="pp-ins">GC</span>TTTT<br>
    AAAA<span class="pp-ins">AT</span>TTTT
    </div>

Insert reverse-complemented content (rc=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rc=True`` reverse-complements each content sequence before insertion.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    inserts = pp.from_seqs(["GCGC", "ATAT"], mode="sequential")
    library = pp.replace_region(wt, inserts, region_name="cre", rc=True)
    # GCGC -> GCGC (palindrome); ATAT -> ATAT (palindrome); shown as-is

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (2 sequences &mdash; content reverse-complemented before insertion)</em>
    AAAA<span class="pp-region">GCGC</span>TTTT<br>
    AAAA<span class="pp-region">ATAT</span>TTTT
    </div>

See :func:`~poolparty.replace_region`.
