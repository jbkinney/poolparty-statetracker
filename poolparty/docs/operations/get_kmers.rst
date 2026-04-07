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
   :widths: auto

   * - Parameter
     - Type
     - Default
     - Description
   * - ``length``
     - ``int``
     - *(required)*
     - k-mer length. Total possible k-mers = 4\ :sup:`length`.
   * - ``pool``
     - ``Pool | str | None``
     - ``None``
     - Background pool or sequence string. When provided with ``region``,
       each k-mer replaces the content of that region.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Region to replace in ``pool``: a marker name or ``[start, stop]``
       interval. Required when ``pool`` is provided.
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
     - Number of output states. ``None`` enumerates all k-mers in
       sequential mode or defaults to 1 in random mode.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.get_kmers` in the
   :doc:`API Reference </api>`.

Examples
--------

All dinucleotides (length=2, sequential)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode='sequential'`` enumerates all 16 dinucleotides in lexicographic order.

.. code-block:: python

    pool = pp.get_kmers(length=2, mode="sequential")
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=2, num_states=16</em>
    AA<br>
    AC<br>
    AG<br>
    AT<br>
    CA
    <span class="pp-ellipsis">... (16 total)</span>
    </div>

Random subset of 4-mers with ``num_states``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cap a large k-mer space using ``num_states`` in random mode to draw a
representative subset without enumerating all 256 4-mers.

.. code-block:: python

    pool = pp.get_kmers(length=4, mode="random", num_states=8)
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=4, num_states=8</em>
    TGGC<br>
    TCAC<br>
    AGCC<br>
    GTTC<br>
    ATTC<br>
    TTAA<br>
    GGAG<br>
    TAAG
    </div>

Lowercase k-mers
~~~~~~~~~~~~~~~~

``case='lower'`` produces lowercase output, useful for visual distinction
when k-mers are joined with uppercase flanking sequences.

.. code-block:: python

    pool = pp.get_kmers(length=2, mode="sequential", case="lower")
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=2, num_states=16</em>
    aa<br>
    ac<br>
    ag<br>
    at<br>
    ca
    <span class="pp-ellipsis">... (16 total)</span>
    </div>

Inserting k-mers into a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide ``pool`` and ``region`` to place every k-mer inside a fixed context,
creating a combinatorial library in one step.

.. code-block:: python

    bg   = pp.from_seq("GCGC<insert>XX</insert>GCGC")
    pool = pp.get_kmers(length=2, mode="sequential", pool=bg, region="insert")
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=10, num_states=16</em>
    GCGC<span class="pp-xtag-cre">&lt;insert&gt;</span>AA<span class="pp-xtag-cre">&lt;/insert&gt;</span>GCGC<br>
    GCGC<span class="pp-xtag-cre">&lt;insert&gt;</span>AC<span class="pp-xtag-cre">&lt;/insert&gt;</span>GCGC<br>
    GCGC<span class="pp-xtag-cre">&lt;insert&gt;</span>AG<span class="pp-xtag-cre">&lt;/insert&gt;</span>GCGC<br>
    GCGC<span class="pp-xtag-cre">&lt;insert&gt;</span>AT<span class="pp-xtag-cre">&lt;/insert&gt;</span>GCGC<br>
    GCGC<span class="pp-xtag-cre">&lt;insert&gt;</span>CA<span class="pp-xtag-cre">&lt;/insert&gt;</span>GCGC
    <span class="pp-ellipsis">... (16 total)</span>
    </div>

Pool method shorthand
~~~~~~~~~~~~~~~~~~~~~

When inserting into a region, the same operation is available as a method
on any ``DnaPool``.  The call ``bg.insert_kmers(...)`` is equivalent to
``pp.get_kmers(..., pool=bg)`` — it simply passes ``self`` as the
background pool.

.. code-block:: python

    bg   = pp.from_seq("GCGC<insert>XX</insert>GCGC")
    pool = bg.insert_kmers(length=2, region="insert", mode="sequential")
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=10, num_states=16</em>
    GCGC<span class="pp-xtag-cre">&lt;insert&gt;</span>AA<span class="pp-xtag-cre">&lt;/insert&gt;</span>GCGC<br>
    GCGC<span class="pp-xtag-cre">&lt;insert&gt;</span>AC<span class="pp-xtag-cre">&lt;/insert&gt;</span>GCGC<br>
    GCGC<span class="pp-xtag-cre">&lt;insert&gt;</span>AG<span class="pp-xtag-cre">&lt;/insert&gt;</span>GCGC<br>
    GCGC<span class="pp-xtag-cre">&lt;insert&gt;</span>AT<span class="pp-xtag-cre">&lt;/insert&gt;</span>GCGC<br>
    GCGC<span class="pp-xtag-cre">&lt;insert&gt;</span>CA<span class="pp-xtag-cre">&lt;/insert&gt;</span>GCGC
    <span class="pp-ellipsis">... (16 total)</span>
    </div>

See :func:`~poolparty.get_kmers` and :meth:`~poolparty.DnaPool.insert_kmers`.
