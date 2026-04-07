replace_region
==============

Replace the entire content of a named region with sequences drawn from a
content pool. By default, background and content states are paired 1:1
(``sync=True``) and region tags are preserved (``keep_tags=True``).
Set ``sync=False`` for a Cartesian product and ``keep_tags=False`` to
strip the region tags.

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
     - The background Pool or sequence string whose named region will be
       replaced.
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
     - ``True``
     - When ``True``, pair background and content states 1:1 (lockstep
       iteration) instead of taking the Cartesian product.
   * - ``keep_tags``
     - ``bool``
     - ``True``
     - When ``True``, preserve the region tags around the replaced content
       so the region can still be referenced by downstream operations.
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
   parameter list, see :func:`~poolparty.replace_region` in the
   :doc:`API Reference </api>`.

Examples
--------

Pair each background with its own insert
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When both pools have the same number of states, the default ``sync=True``
pairs them 1:1. Tags are preserved (``keep_tags=True``) so the region can
be referenced by later operations.

.. code-block:: python

    backgrounds = pp.from_seqs(
        ["AAAA<bc/>TTTT", "CCCC<bc/>GGGG", "GGGG<bc/>CCCC"],
        mode="sequential",
    )
    inserts = pp.from_seqs(["XX", "YY", "ZZ"], mode="sequential")
    library = pp.replace_region(backgrounds, inserts, region_name="bc")
    library.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">library: seq_length=None, num_states=3</em>
    AAAA<span class="pp-xtag-cre">&lt;bc&gt;</span>XX<span class="pp-xtag-cre">&lt;/bc&gt;</span>TTTT<br>
    CCCC<span class="pp-xtag-cre">&lt;bc&gt;</span>YY<span class="pp-xtag-cre">&lt;/bc&gt;</span>GGGG<br>
    GGGG<span class="pp-xtag-cre">&lt;bc&gt;</span>ZZ<span class="pp-xtag-cre">&lt;/bc&gt;</span>CCCC
    </div>

Replace a region with multiple variants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single background with multiple inserts requires ``sync=False`` (different
state counts cannot be synced). Here the 4-base ``cre`` region is replaced
with three shorter variants.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    inserts = pp.from_seqs(["AAA", "TTT", "CCC"], mode="sequential")
    library = pp.replace_region(wt, inserts, region_name="cre", sync=False, keep_tags=False)
    library.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">library: seq_length=11, num_states=3</em>
    AAAAAAATTTT<br>
    AAAATTTTTTT<br>
    AAAACCCTTTT
    </div>

Replace a zero-length point tag (pure insertion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the region is a zero-length point tag, ``replace_region`` inserts
without deleting any bases.

.. code-block:: python

    wt      = pp.from_seq("AAAA<ins/>TTTT")
    inserts = pp.from_seqs(["GC", "AT"], mode="sequential")
    library = pp.replace_region(wt, inserts, region_name="ins", sync=False, keep_tags=False)
    library.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">library: seq_length=10, num_states=2</em>
    AAAAGCTTTT<br>
    AAAAATTTTT
    </div>

Cartesian product with sync=False
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting ``sync=False`` takes the Cartesian product of background and content
states. Compare with the first example — same inputs, but 9 states instead
of 3.

.. code-block:: python

    backgrounds = pp.from_seqs(
        ["AAAA<bc/>TTTT", "CCCC<bc/>GGGG", "GGGG<bc/>CCCC"],
        mode="sequential",
    )
    inserts = pp.from_seqs(["XX", "YY", "ZZ"], mode="sequential")
    library = pp.replace_region(backgrounds, inserts, region_name="bc", sync=False, keep_tags=False)
    library.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">library: seq_length=None, num_states=9</em>
    AAAAXXTTTT<br>
    AAAAYYTTTT<br>
    AAAAZZTTTT<br>
    CCCCXXGGGG<br>
    CCCCYYGGGG<br>
    CCCCZZGGGG<br>
    GGGGXXCCCC<br>
    GGGGYYCCCC<br>
    GGGGZZCCCC
    </div>

See :func:`~poolparty.replace_region`.
