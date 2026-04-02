annotate_region
===============

Tag a region by position range and optionally apply a display style in one
step. If the named region already exists in the pool, ``extent`` must be
``None`` (the existing bounds are kept) but a new ``style`` can still be
applied.

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
     - ``Pool``
     - *(required)*
     - The Pool to annotate.
   * - ``region_name``
     - ``str``
     - *(required)*
     - Name for the region (e.g. ``'cre'``, ``'orf'``).
   * - ``extent``
     - ``tuple[int, int] | None``
     - ``None``
     - ``(start, stop)`` tuple (0-based, exclusive stop). When ``None``
       and the region does not yet exist, the entire sequence is
       annotated. Must be ``None`` if the region already exists.
   * - ``style``
     - ``str | None``
     - ``None``
     - Named display style applied to the region (e.g. ``'bold_blue'``).
       ``None`` leaves the display unchanged.
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

Annotate by extent with no style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mark positions 4–12 as the ``cre`` region with default rendering.

.. code-block:: python

    wt        = pp.from_seq("AAAAATCGATCGTTTT")
    annotated = pp.annotate_region(wt, "cre", extent=(4, 12))

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; <em>cre</em> tagged at positions 4&ndash;12)</em>
    AAAA<span class="pp-region">ATCGATCG</span>TTTT
    </div>

Annotate entire sequence (extent omitted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Omit ``extent`` to tag the full sequence as a single named region.

.. code-block:: python

    wt        = pp.from_seq("ATCGATCG")
    annotated = pp.annotate_region(wt, "full")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; entire sequence tagged as <em>full</em>)</em>
    <span class="pp-region">ATCGATCG</span>
    </div>

Annotate with a named style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``style`` inserts the tag and applies a colour/weight in one call.

.. code-block:: python

    wt        = pp.from_seq("AAAAATCGATCGTTTT")
    annotated = pp.annotate_region(wt, "cre", extent=(4, 12),
                                   style="bold_green")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; <em>cre</em> tagged and styled bold green)</em>
    AAAA<span style="color:#15803d;font-weight:bold;">ATCGATCG</span>TTTT
    </div>

Two regions with different styles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chain two calls to give adjacent segments distinct colours.

.. code-block:: python

    wt    = pp.from_seq("AAAAATCGGGGGCCCTTTT")
    step1 = pp.annotate_region(wt,    "left",  extent=(4, 8),   style="bold_blue")
    step2 = pp.annotate_region(step1, "right", extent=(13, 17), style="bold_red")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; <em>left</em> blue, <em>right</em> red)</em>
    AAAA<span style="color:#1d4ed8;font-weight:bold;">ATCG</span>GGGGG<span style="color:#dc2626;font-weight:bold;">CCCT</span>TTT
    </div>

Apply a style to an existing region (extent=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the region already exists, omit ``extent`` and supply only ``style``
to add colour without re-tagging.

.. code-block:: python

    wt     = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    styled = pp.annotate_region(wt, "cre", style="red")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; existing <em>cre</em> region styled red)</em>
    AAAA<span style="color:#dc2626;">ATCGATCG</span>TTTT
    </div>

See :func:`~poolparty.annotate_region`.
