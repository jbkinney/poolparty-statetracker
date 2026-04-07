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
   :widths: auto
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
     - Enumeration order when combined with other pools.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.extract_region` in the
   :doc:`API Reference </api>`.

Examples
--------

Extract a tagged region
~~~~~~~~~~~~~~~~~~~~~~~

Pull the content of the ``cre`` region out of a flanked sequence.

.. code-block:: python

    bg      = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    content = pp.extract_region(bg, "cre")
    content.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">content: seq_length=8, num_states=1</em>
    ATCGATCG
    </div>

Extract and reverse-complement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``rc=True`` to get the reverse complement of the region content — useful
for antisense constructs.

.. code-block:: python

    bg         = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    content_rc = pp.extract_region(bg, "cre", rc=True)
    content_rc.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">content_rc: seq_length=8, num_states=1</em>
    CGATCGAT
    </div>

Extract from a multi-state pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the parent pool has multiple states, each state's region content is
extracted independently.

.. code-block:: python

    bg      = pp.from_seq("AAAA<ins>NNNN</ins>TTTT")
    filled  = pp.replace_region(bg, pp.from_iupac("NNNN", mode="sequential"),
                                region_name="ins", sync=False, keep_tags=True)
    content = pp.extract_region(filled, "ins")
    content.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">content: seq_length=4, num_states=256</em>
    AAAA<br>
    AAAC<br>
    AAAG<br>
    AAAT<br>
    AACA<br>
    <span class="pp-ellipsis">... (256 total)</span>
    </div>

See :func:`~poolparty.extract_region`.
