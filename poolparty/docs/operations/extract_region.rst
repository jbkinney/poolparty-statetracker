extract_region
==============

Extract the content of a named region as a new pool. The flanking sequence
is discarded and the result contains only the bases inside the region tags.

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
     - Input pool or sequence string containing the named region.
   * - ``region_name``
     - ``str``
     - *(required)*
     - Name of the region whose content will be extracted.
   * - ``rc``
     - ``bool``
     - ``False``
     - If ``True``, reverse-complement the extracted content.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Iteration priority for downstream multi-pool iteration.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.

----

Examples
--------

Extract a tagged region
~~~~~~~~~~~~~~~~~~~~~~~

Pull the content of the ``cre`` region out of a flanked sequence.

.. code-block:: python

    bg      = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    content = pp.extract_region(bg, "cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; content of <em>cre</em>)</em>
    ATCGATCG
    </div>

Extract and reverse-complement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``rc=True`` to get the reverse complement of the region content — useful
for antisense constructs.

.. code-block:: python

    bg         = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    content_rc = pp.extract_region(bg, "cre", rc=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; reverse complement of <em>cre</em>)</em>
    CGATCGAT
    </div>

Extract from a multi-state pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the parent pool has multiple states, each state's region content is
extracted independently.

.. code-block:: python

    bg      = pp.from_seq("AAAA<ins>NNNN</ins>TTTT")
    filled  = pp.replace_region(bg, pp.from_iupac("NNNN", mode="sequential"),
                                region_name="ins")
    content = pp.extract_region(filled, "ins")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (256 sequences &mdash; extracted <em>ins</em> content from each state)</em>
    AAAA<br>
    AAAC<br>
    AAAG<br>
    <span class="pp-ellipsis">... (256 total &mdash; flanks discarded)</span>
    </div>

See :func:`~poolparty.extract_region`.
