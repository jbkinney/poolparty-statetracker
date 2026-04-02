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
   :widths: 20 18 12 50

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
     - ``Pool | None``
     - ``None``
     - Background pool. When provided with ``region``, each generated
       sequence replaces the content of that region.
   * - ``region``
     - ``str | None``
     - ``None``
     - Region to replace in ``pool``. Required when ``pool`` is provided.
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
     - Cap on total states. With ``mode='sequential'`` takes the first N;
       with ``mode='random'`` draws N independently.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to every generated sequence.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

----

Examples
--------

Random sampling from an ambiguous code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``W`` = {A,T}, ``N`` = {A,C,G,T}: 2 × 4 = 8 total combinations.
Random mode draws one sequence per call.

.. code-block:: python

    pool = pp.from_iupac("WN", num_states=4)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (8 possible sequences, random &mdash; W={A,T} N={A,C,G,T})</em>
    AA<br>TG<br>AC<br>TC<br>
    <span class="pp-ellipsis">... (stochastic &mdash; draws sample uniformly from all 8 combinations)</span>
    </div>

Sequential enumeration
~~~~~~~~~~~~~~~~~~~~~~

``mode='sequential'`` iterates all combinations in lexicographic order,
giving exactly one draw per combination.

.. code-block:: python

    pool = pp.from_iupac("RY", mode="sequential")
    # R={A,G}, Y={C,T}: 2x2=4 combinations

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (4 sequences, sequential &mdash; R={A,G} Y={C,T})</em>
    AC<br>AT<br>GC<br>GT
    </div>

Capping with ``num_states``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine ``mode='sequential'`` with ``num_states`` to take only the first N
combinations — useful for prototyping with a large degenerate sequence.

.. code-block:: python

    pool = pp.from_iupac("NNNN", mode="sequential")
    # NNNN has 256 total; take first 8

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (8 of 256 sequences &mdash; first 8 in lexicographic order)</em>
    AAAA<br>AAAC<br>AAAG<br>AAAT<br>AACA<br>AACC<br>AACG<br>AACT
    </div>

Inserting into a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide ``pool`` and ``region`` to enumerate all IUPAC combinations into a
fixed flanking context.

.. code-block:: python

    bg   = pp.from_seq("AAAA<cre>XXX</cre>TTTT")
    pool = pp.from_iupac("NNN", mode="sequential", pool=bg, region="cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (64 sequences &mdash; all 3-mers in <em>cre</em> region)</em>
    AAAA<span class="pp-region">AAA</span>TTTT<br>
    AAAA<span class="pp-region">AAC</span>TTTT<br>
    AAAA<span class="pp-region">AAG</span>TTTT<br>
    AAAA<span class="pp-region">AAT</span>TTTT<br>
    <span class="pp-ellipsis">... (64 total)</span>
    </div>

See :func:`~poolparty.from_iupac`.
