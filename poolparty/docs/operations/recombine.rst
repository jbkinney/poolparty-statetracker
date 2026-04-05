recombine
=========

Produce chimeric sequences by slicing multiple source pools at breakpoints
and stitching together alternating segments. Requires at least two source pools
of equal sequence length. Use ``mode='sequential'`` to enumerate all breakpoint
combinations deterministically, or ``mode='random'`` (default) to draw chimeras
stochastically.

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
     - ``Pool | str | None``
     - ``None``
     - Parent pool for region-based recombination. If provided with
       ``region``, the recombined sequences replace the region content.
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
     - Explicit list of breakpoint positions. ``None`` = positions chosen by
       the current ``mode``.
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
     - ``'random'`` selects breakpoints randomly; ``'sequential'`` enumerates
       all breakpoint positions; ``'fixed'`` uses ``positions`` exactly once.
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

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.recombine` in the
   :doc:`API Reference </api>`.

Examples
--------

Two sources, single breakpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enumerate all single-crossover chimeras between two 10-base sequences. Each
breakpoint position produces two states (one per source ordering), yielding
18 chimeras total.

.. code-block:: python

    src_a = pp.from_seq("AAAAAAAAAA")
    src_b = pp.from_seq("CCCCCCCCCC")
    rec   = pp.recombine(sources=[src_a, src_b], mode="sequential")
    rec.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">rec: seq_length=10, num_states=18</em>
    ACCCCCCCCC<br>
    CAAAAAAAAA<br>
    AACCCCCCCC<br>
    CCAAAAAAAA<br>
    AAACCCCCCC
    <span class="pp-ellipsis">... (18 total)</span>
    </div>

Random chimeras (mode="random")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode='random'`` draws breakpoints stochastically, producing a different
chimera each draw. Use ``num_states`` to sample several chimeras at once.

.. code-block:: python

    src_a = pp.from_seq("AAAAAAAAAA")
    src_b = pp.from_seq("CCCCCCCCCC")
    rec   = pp.recombine(sources=[src_a, src_b], mode="random", num_states=5)
    rec.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">rec: seq_length=10, num_states=5</em>
    AAAAAAAAAC<br>
    CCCCCCCCCA<br>
    AACCCCCCCC<br>
    CCCAAAAAAA<br>
    CCCCCAAAAA
    </div>

Two breakpoints (num_breakpoints=2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two crossover points divide each chimera into three segments. With
``mode='sequential'`` every combination of two breakpoint positions is
enumerated.

.. code-block:: python

    src_a = pp.from_seq("AAAAAAAAAA")
    src_b = pp.from_seq("CCCCCCCCCC")
    rec   = pp.recombine(sources=[src_a, src_b], num_breakpoints=2,
                         mode="sequential")
    rec.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">rec: seq_length=10, num_states=72</em>
    ACAAAAAAAA<br>
    CACCCCCCCC<br>
    ACCAAAAAAA<br>
    CAACCCCCCC<br>
    ACCCAAAAAA
    <span class="pp-ellipsis">... (72 total)</span>
    </div>

Fixed breakpoint position (positions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``positions=[5]`` pins the breakpoint after index 5, so chimeras always split
into a 6-character head and a 4-character tail. With two sources this yields
exactly 2 states.

.. code-block:: python

    src_a = pp.from_seq("AAAAAAAAAA")
    src_b = pp.from_seq("TTTTTTTTTT")
    rec   = pp.recombine(sources=[src_a, src_b], positions=[5],
                         mode="sequential")
    rec.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">rec: seq_length=10, num_states=2</em>
    AAAAAATTTT<br>
    TTTTTTAAAA
    </div>

Color segments by source (style_by="source")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``style_by='source'`` applies the same colour to every segment originating
from the same source pool, regardless of its position in the chimera.

.. code-block:: python

    src_a = pp.from_seq("AAAAAAAAAA")
    src_b = pp.from_seq("CCCCCCCCCC")
    rec   = pp.recombine(
        sources=[src_a, src_b],
        num_breakpoints=2,
        styles=["blue", "red"],
        style_by="source",
        mode="sequential",
    )
    rec.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">rec: seq_length=10, num_states=72</em>
    <span class="pp-codon-a">A</span><span class="pp-codon-b">C</span><span class="pp-codon-a">AAAAAAAA</span><br>
    <span class="pp-codon-b">C</span><span class="pp-codon-a">A</span><span class="pp-codon-b">CCCCCCCC</span><br>
    <span class="pp-codon-a">A</span><span class="pp-codon-b">CC</span><span class="pp-codon-a">AAAAAAA</span><br>
    <span class="pp-codon-b">C</span><span class="pp-codon-a">AA</span><span class="pp-codon-b">CCCCCCC</span><br>
    <span class="pp-codon-a">A</span><span class="pp-codon-b">CCC</span><span class="pp-codon-a">AAAAAA</span>
    <span class="pp-ellipsis">... (72 total)</span>
    </div>

See :func:`~poolparty.recombine`.
