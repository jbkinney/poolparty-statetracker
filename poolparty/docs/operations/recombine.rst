recombine
=========

Produce chimeric sequences by slicing multiple source sequences at breakpoints
and stitching together alternating segments. Requires at least two source pools
of equal sequence length.

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
   * - ``sources``
     - ``list[Pool]``
     - *(required)*
     - Tuple or list of at least two :class:`~poolparty.Pool` objects to
       recombine. All sources must produce sequences of the same length.
   * - ``num_breakpoints``
     - ``int``
     - ``1``
     - Number of crossover breakpoints. A single breakpoint produces 2-segment
       chimeras; N breakpoints produce N+1 segments.
   * - ``positions``
     - ``list[int] | None``
     - ``None``
     - Explicit list of breakpoint positions. ``None`` = random positions each
       draw.
   * - ``styles``
     - ``list[str] | None``
     - ``None``
     - List of display styles, one per segment (length = ``num_breakpoints``
       + 1).
   * - ``style_by``
     - ``str``
     - ``'order'``
     - ``'order'`` styles segments by position in the chimera;
       ``'source'`` styles them by which source pool they came from.
   * - ``region``
     - ``str | None``
     - ``None``
     - Name of a tagged region to restrict recombination to.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'random'`` selects breakpoints randomly; ``'sequential'`` enumerates.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Fix the total number of output states.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.

----

Examples
--------

Two sources, single breakpoint (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each draw picks a random crossover position and returns a two-segment chimera.

.. code-block:: python

    src_a = pp.from_seq("AAAAAAAAAA")
    src_b = pp.from_seq("CCCCCCCCCC")
    rec   = pp.recombine(sources=[src_a, src_b])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; single crossover between src_a and src_b)</em>
    AAAA<span class="pp-mut">CCCCCC</span><br>
    AA<span class="pp-mut">CCCCCCCC</span><br>
    AAAAAAA<span class="pp-mut">CCC</span><br>
    <span class="pp-ellipsis">... (breakpoint position varies each draw)</span>
    </div>

Two breakpoints (num_breakpoints=2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two crossover points divide each chimera into three segments.

.. code-block:: python

    src_a = pp.from_seq("AAAAAAAAAA")
    src_b = pp.from_seq("CCCCCCCCCC")
    rec   = pp.recombine(sources=[src_a, src_b], num_breakpoints=2)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 3-segment chimeras)</em>
    AA<span class="pp-mut">CCCC</span>AAAA<br>
    <span class="pp-mut">CCCC</span>AAAA<span class="pp-mut">CC</span><br>
    AAA<span class="pp-mut">CCC</span>AAAA<br>
    <span class="pp-ellipsis">... (two random breakpoints chosen each draw)</span>
    </div>

Fixed breakpoint position (positions parameter)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``positions=[5]`` restricts the breakpoint to position 5, so chimeras always
cross over after index 5.

.. code-block:: python

    src_a = pp.from_seq("AAAAAAAAAA")
    src_b = pp.from_seq("TTTTTTTTTT")
    rec   = pp.recombine(sources=[src_a, src_b], positions=[5], mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (2 sequences &mdash; breakpoint fixed after position 5)</em>
    AAAAAA<span class="pp-mut">TTTT</span><br>
    <span class="pp-mut">TTTTTT</span>AAAA
    </div>

Color segments by source (style_by="source")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``style_by='source'`` applies the same style to all segments originating from
the same source pool, regardless of their position in the chimera.

.. code-block:: python

    src_a = pp.from_seq("AAAAAAAAAA")
    src_b = pp.from_seq("CCCCCCCCCC")
    rec   = pp.recombine(
        sources=[src_a, src_b],
        num_breakpoints=2,
        styles=["blue", "red"],
        style_by="source",
    )

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; blue = src_a segments, red = src_b segments)</em>
    <span style="color:#1d4ed8;font-weight:bold;">AA</span><span style="color:#dc2626;font-weight:bold;">CCCC</span><span style="color:#1d4ed8;font-weight:bold;">AAAA</span><br>
    <span style="color:#dc2626;font-weight:bold;">CCCC</span><span style="color:#1d4ed8;font-weight:bold;">AAAA</span><span style="color:#dc2626;font-weight:bold;">CC</span><br>
    <span class="pp-ellipsis">... (src_a segments always blue, src_b segments always red)</span>
    </div>

See :func:`~poolparty.recombine`.
