stack
=====

Combine multiple pools into one by stacking their state spaces as a disjoint
union — each state in the resulting pool comes from exactly one of the input
pools, enumerated in order.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Parameters
----------

.. list-table::
   :widths: 20 18 12 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pools``
     - ``list[Pool]``
     - *(required)*
     - List of pools to stack. States are concatenated in order.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Iteration priority for downstream multi-pool iteration.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output. Available card
       key: ``"active_parent"`` (index of which input pool produced each
       state).

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.stack` in the
   :doc:`API Reference </api>`.

Examples
--------

Stack Three Fixed-Sequence Pools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Merge three single-sequence pools into one pool that contains all three
sequences.

.. code-block:: python

    a = pp.from_seq("AAAA")
    b = pp.from_seq("CCCC")
    c = pp.from_seq("GGGG")
    combined = pp.stack([a, b, c])
    combined.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">combined: seq_length=4, num_states=3</em>
    AAAA<br>
    CCCC<br>
    GGGG
    </div>

Stack Pools of Different Sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stack a four-sequence pool and a two-sequence pool to produce a single pool
with six states total.

.. code-block:: python

    pool_a = pp.from_seqs(["AAAA", "CCCC", "GGGG", "TTTT"], mode="sequential")
    pool_b = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
    combined = pp.stack([pool_a, pool_b])
    combined.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">combined: seq_length=4, num_states=6</em>
    AAAA<br>
    CCCC<br>
    GGGG<br>
    TTTT<br>
    ACGT<br>
    TGCA
    </div>

Stack the Results of Two Scan Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine deletion scans over two different sequences into one pooled library
covering both targets.

.. code-block:: python

    wt_a  = pp.from_seq("AAAACCCC")
    wt_b  = pp.from_seq("GGGGTTTT")
    dels_a = wt_a.deletion_scan(deletion_length=2, mode="sequential")
    dels_b = wt_b.deletion_scan(deletion_length=2, mode="sequential")
    merged = pp.stack([dels_a, dels_b])
    merged.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">merged: seq_length=8, num_states=14</em>
    --AACCCC<br>
    A--ACCCC<br>
    AA--CCCC<br>
    AAA--CCC<br>
    AAAA--CC<br>
    <span class="pp-ellipsis">... (14 total)</span>
    </div>

Stack combined state space
~~~~~~~~~~~~~~~~~~~~~~~~~~~

After stacking, the combined pool enumerates every state from the first
input, then every state from the second, and so on.

.. code-block:: python

    pool_a = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
    pool_b = pp.from_seqs(["GGGG", "TTTT"], mode="sequential")
    combined = pp.stack([pool_a, pool_b])
    combined.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">combined: seq_length=4, num_states=4</em>
    AAAA<br>
    CCCC<br>
    GGGG<br>
    TTTT
    </div>

Operator shorthand (``+``)
~~~~~~~~~~~~~~~~~~~~~~~~~~

``pool_a + pool_b`` is equivalent to ``pp.stack([pool_a, pool_b])`` — it
creates a disjoint union of both pools' states so draws can come from either
pool. Chaining ``+`` appends additional pools.

.. code-block:: python

    wt   = pp.from_seq("ATCG")
    muts = pp.mutagenize(wt, num_mutations=1, mode="sequential")
    ctrl = pp.from_seqs(["AAAA", "TTTT"], mode="sequential")
    lib  = muts + ctrl   # all single-point mutants + 2 controls
    lib.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">lib: seq_length=4, num_states=14</em>
    CTCG<br>
    GTCG<br>
    TTCG<br>
    AACG<br>
    ACCG
    <span class="pp-ellipsis">... (14 total)</span>
    </div>

.. code-block:: python

    a = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
    b = pp.from_seqs(["GGGG"], mode="sequential")
    c = pp.from_seqs(["TTTT", "ACGT"], mode="sequential")
    lib = a + b + c
    lib.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">lib: seq_length=4, num_states=5</em>
    AAAA<br>
    CCCC<br>
    GGGG<br>
    TTTT<br>
    ACGT
    </div>

See :func:`~poolparty.stack`.
