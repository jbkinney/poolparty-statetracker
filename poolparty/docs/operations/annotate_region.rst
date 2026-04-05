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
   :widths: auto

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
     - Named display style applied to the region (e.g. ``'blue bold'``).
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

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.annotate_region` in the
   :doc:`API Reference </api>`.

Examples
--------

Annotate by extent with no style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mark positions 4–12 as the ``cre`` region with default rendering.

.. code-block:: python

    wt        = pp.from_seq("AAAAATCGATCGTTTT")
    annotated = pp.annotate_region(wt, "cre", extent=(4, 12))
    annotated.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">annotated: seq_length=16, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>

Annotate entire sequence (extent omitted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Omit ``extent`` to tag the full sequence as a single named region.

.. code-block:: python

    wt        = pp.from_seq("ATCGATCG")
    annotated = pp.annotate_region(wt, "full")
    annotated.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">annotated: seq_length=8, num_states=1</em>
    <span class="pp-xtag-cre">&lt;full&gt;</span>ATCGATCG<span class="pp-xtag-cre">&lt;/full&gt;</span>
    </div>

Annotate with a named style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``style`` inserts the tag and applies a colour/weight in one call.

.. code-block:: python

    wt        = pp.from_seq("AAAAATCGATCGTTTT")
    annotated = pp.annotate_region(wt, "cre", extent=(4, 12),
                                   style="green bold")
    annotated.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">annotated: seq_length=16, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><strong class="pp-codon-c">ATCGATCG</strong><span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>

Two regions with different styles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chain two calls to give adjacent segments distinct colours.

.. code-block:: python

    wt    = pp.from_seq("AAAAATCGGGGGCCCTTTT")
    step1 = pp.annotate_region(wt,    "left",  extent=(4, 8),   style="blue bold")
    step2 = pp.annotate_region(step1, "right", extent=(13, 17), style="red bold")
    step2.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">step2: seq_length=19, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;left&gt;</span><strong class="pp-codon-a">ATCG</strong><span class="pp-xtag-cre">&lt;/left&gt;</span>GGGGC<span class="pp-xtag-cre">&lt;right&gt;</span><strong class="pp-mut">CCTT</strong><span class="pp-xtag-cre">&lt;/right&gt;</span>TT
    </div>

Apply a style to an existing region (extent=None)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the region already exists, omit ``extent`` and supply only ``style``
to add colour without re-tagging.

.. code-block:: python

    wt     = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    styled = pp.annotate_region(wt, "cre", style="red")
    styled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">styled: seq_length=16, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><span class="pp-mut">ATCGATCG</span><span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>

See :func:`~poolparty.annotate_region`.
