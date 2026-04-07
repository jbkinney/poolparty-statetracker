from_iupac
==========

Create a pool from an IUPAC ambiguity-code string. Every ambiguous position
expands to its corresponding base set; the pool enumerates all possible
combinations. By default the pool samples uniformly at random; pass
``mode='sequential'`` to enumerate in lexicographic order.

IUPAC codes: R = {A,G}, Y = {C,T}, S = {C,G}, W = {A,T}, K = {G,T},
M = {A,C}, B = {C,G,T}, D = {A,G,T}, H = {A,C,T}, V = {A,C,G},
N = {A,C,G,T}.

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
   * - ``iupac_seq``
     - ``str``
     - *(required)*
     - IUPAC sequence string (case-insensitive). Unambiguous bases pass
       through unchanged.
   * - ``pool``
     - ``Pool | str | None``
     - ``None``
     - Background pool or sequence string. When provided with ``region``,
       each generated sequence replaces the content of that region.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Region to replace in ``pool``: a marker name or ``[start, stop]``
       interval. Required when ``pool`` is provided.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` enumerates every combination in lexicographic
       order; ``'random'`` samples uniformly at random.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Number of output states. ``None`` enumerates all combinations in
       sequential mode or defaults to 1 in random mode.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Enumeration order when combined with other pools.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to every generated sequence.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.from_iupac` in the
   :doc:`API Reference </api>`.

Examples
--------

Random sampling from an ambiguous code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``W`` = {A,T}, ``N`` = {A,C,G,T}: 2 × 4 = 8 total combinations.
Random mode draws one sequence per call.

.. code-block:: python

    pool = pp.from_iupac("WN", mode="random", num_states=4)
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=2, num_states=4</em>
    TG<br>
    TT<br>
    AC<br>
    TC
    </div>

Sequential enumeration
~~~~~~~~~~~~~~~~~~~~~~

``mode='sequential'`` iterates all combinations in lexicographic order,
giving exactly one draw per combination.

.. code-block:: python

    pool = pp.from_iupac("RY", mode="sequential")
    pool.print_library()
    # R={A,G}, Y={C,T}: 2x2=4 combinations

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=2, num_states=4</em>
    AC<br>
    AT<br>
    GC<br>
    GT
    </div>

Capping with ``num_states``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine ``mode='sequential'`` with ``num_states`` to take only the first N
combinations — useful for prototyping with a large degenerate sequence.

.. code-block:: python

    pool = pp.from_iupac("NNNN", mode="sequential", num_states=8)
    pool.print_library()
    # NNNN has 256 total; take first 8

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=4, num_states=8</em>
    AAAA<br>
    AAAC<br>
    AAAG<br>
    AAAT<br>
    AACA<br>
    AACC<br>
    AACG<br>
    AACT
    </div>

Inserting into a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide ``pool`` and ``region`` to enumerate all IUPAC combinations into a
fixed flanking context.

.. code-block:: python

    bg   = pp.from_seq("AAAA<cre>XXX</cre>TTTT")
    pool = pp.from_iupac("NNN", mode="sequential", pool=bg, region="cre")
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=11, num_states=64</em>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>AAA<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>AAC<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>AAG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>AAT<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ACA<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT
    <span class="pp-ellipsis">... (64 total)</span>
    </div>

Pool method shorthand
~~~~~~~~~~~~~~~~~~~~~

When inserting into a region, the same operation is available as a method
on any ``DnaPool``.  The call ``bg.insert_from_iupac(...)`` is equivalent
to ``pp.from_iupac(..., pool=bg)`` — it simply passes ``self`` as the
background pool.

.. code-block:: python

    bg   = pp.from_seq("AAAA<cre>XXX</cre>TTTT")
    pool = bg.insert_from_iupac("NNN", region="cre", mode="sequential")
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=11, num_states=64</em>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>AAA<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>AAC<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>AAG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>AAT<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ACA<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT
    <span class="pp-ellipsis">... (64 total)</span>
    </div>

See :func:`~poolparty.from_iupac` and :meth:`~poolparty.DnaPool.insert_from_iupac`.
