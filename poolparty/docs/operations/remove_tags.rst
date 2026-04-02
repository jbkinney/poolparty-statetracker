remove_tags
===========

Remove XML-style region tags from sequences, either keeping or discarding
the enclosed bases. After removal the region is no longer tracked by the pool.

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
     - The Pool to remove tags from. Can also be a plain sequence string.
   * - ``region_name``
     - ``str``
     - *(required)*
     - Name of the region whose tags should be removed.
   * - ``keep_content``
     - ``bool``
     - ``True``
     - When ``True`` the enclosed bases are kept and only the tag markup
       is stripped. When ``False`` both the tags and the enclosed bases
       are deleted, shortening the sequence.
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

Keep content (default)
~~~~~~~~~~~~~~~~~~~~~~~

Strip the ``cre`` tags but leave the four enclosed bases in place.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    cleaned = pp.remove_tags(wt, "cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; tags stripped, 4 bases kept)</em>
    AAAATCGTTTT
    </div>

Drop content (keep_content=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Delete both the tags and the enclosed bases, shortening the sequence.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    dropped = pp.remove_tags(wt, "cre", keep_content=False)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; tags and 4 enclosed bases deleted)</em>
    AAAATTTT
    </div>

Strip scan tags while keeping another region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After :func:`~poolparty.region_scan` the scanning tag remains in every
sequence. Use ``remove_tags`` to strip it while leaving the ``cre`` region
tag intact.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    scan = pp.region_scan(wt, region="win", region_length=2,
                          region_constraint="cre", mode="sequential")
    out  = pp.remove_tags(scan, "win")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (7 sequences &mdash; <em>win</em> scan tag removed; <em>cre</em> tag still present)</em>
    AAAA<span class="pp-region">ATCGATCG</span>TTTT<br>
    AAAA<span class="pp-region">ATCGATCG</span>TTTT<br>
    AAAA<span class="pp-region">ATCGATCG</span>TTTT<br>
    <span class="pp-ellipsis">... (7 total; one per window position)</span>
    </div>

Remove two regions sequentially
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call ``remove_tags`` once per region name to clear multiple tags.

.. code-block:: python

    wt    = pp.from_seq("AAAA<left>ATCG</left>GGGG<right>CCCT</right>TTT")
    step1 = pp.remove_tags(wt,    "left")
    clean = pp.remove_tags(step1, "right")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; both region tags stripped, all bases kept)</em>
    AAAAATCGGGGGCCCTTTT
    </div>

See :func:`~poolparty.remove_tags`.
