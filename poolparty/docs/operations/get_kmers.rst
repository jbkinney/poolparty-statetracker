get_kmers
=========

Enumerate every k-mer of a given length over the DNA alphabet (A, C, G, T).
By default the pool samples uniformly at random; pass ``mode='sequential'``
to iterate through all 4\ :sup:`k` k-mers in lexicographic order.

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
   * - ``length``
     - ``int``
     - *(required)*
     - k-mer length. Total possible k-mers = 4\ :sup:`length`.
   * - ``pool``
     - ``Pool | None``
     - ``None``
     - Background pool. When provided with ``region``, each k-mer replaces
       the content of that region.
   * - ``region``
     - ``str | None``
     - ``None``
     - Region to replace in ``pool``. Required when ``pool`` is provided.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to every k-mer.
   * - ``case``
     - ``str``
     - ``'upper'``
     - ``'upper'`` (default) or ``'lower'`` output case.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` iterates all 4\ :sup:`length` k-mers in
       lexicographic order; ``'random'`` samples uniformly at random.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Cap on total states. With ``mode='sequential'`` takes the first N.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

----

Examples
--------

All dinucleotides (length=2, sequential)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode='sequential'`` enumerates all 16 dinucleotides in lexicographic order.

.. code-block:: python

    pool = pp.get_kmers(length=2, mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (16 sequences &mdash; all dinucleotides in order)</em>
    AA<br>AC<br>AG<br>AT<br>CA<br>CC<br>CG<br>CT<br>GA<br>GC<br>GG<br>GT<br>TA<br>TC<br>TG<br>TT
    </div>

Random subset of 4-mers with ``num_states``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cap a large k-mer space using ``num_states`` in random mode to draw a
representative subset without enumerating all 256 4-mers.

.. code-block:: python

    pool = pp.get_kmers(length=4, mode="random", num_states=8)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (8 of 256 4-mers, random &mdash; stochastic example draw)</em>
    ACGT<br>TTAA<br>GCGC<br>AACC<br>TGCA<br>CGTA<br>GATC<br>ATAT
    </div>

Lowercase k-mers
~~~~~~~~~~~~~~~~

``case='lower'`` produces lowercase output, useful for visual distinction
when k-mers are joined with uppercase flanking sequences.

.. code-block:: python

    pool = pp.get_kmers(length=2, mode="sequential", case="lower")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (16 sequences &mdash; all dinucleotides, lowercase)</em>
    aa<br>ac<br>ag<br>at<br>ca<br>cc<br>cg<br>ct<br>ga<br>gc<br>gg<br>gt<br>ta<br>tc<br>tg<br>tt
    </div>

Inserting k-mers into a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide ``pool`` and ``region`` to place every k-mer inside a fixed context,
creating a combinatorial library in one step.

.. code-block:: python

    bg   = pp.from_seq("GCGC<insert>XX</insert>GCGC")
    pool = pp.get_kmers(length=2, mode="sequential", pool=bg, region="insert")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (16 sequences &mdash; all dinucleotides in <em>insert</em> region)</em>
    GCGC<span class="pp-region">AA</span>GCGC<br>
    GCGC<span class="pp-region">AC</span>GCGC<br>
    GCGC<span class="pp-region">AG</span>GCGC<br>
    GCGC<span class="pp-region">AT</span>GCGC<br>
    <span class="pp-ellipsis">... (16 total)</span>
    </div>

See :func:`~poolparty.get_kmers`.
