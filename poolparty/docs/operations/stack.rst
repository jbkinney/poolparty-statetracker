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

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (3 sequences &mdash; disjoint union of a, b, and c)</em>
    AAAA<br>CCCC<br>GGGG
    </div>

Stack Pools of Different Sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stack a four-sequence pool and a two-sequence pool to produce a single pool
with six states total.

.. code-block:: python

    pool_a = pp.from_seqs(["AAAA", "CCCC", "GGGG", "TTTT"])
    pool_b = pp.from_seqs(["ACGT", "TGCA"])
    combined = pp.stack([pool_a, pool_b])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (6 sequences &mdash; 4 from pool_a followed by 2 from pool_b)</em>
    AAAA<br>CCCC<br>GGGG<br>TTTT<br>ACGT<br>TGCA
    </div>

Stack the Results of Two Scan Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine deletion scans over two different sequences into one pooled library
covering both targets.

.. code-block:: python

    wt_a  = pp.from_seq("AAAACCCC")
    wt_b  = pp.from_seq("GGGGTTTT")
    dels_a = wt_a.deletion_scan(deletion_length=2)
    dels_b = wt_b.deletion_scan(deletion_length=2)
    merged = pp.stack([dels_a, dels_b])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (14 sequences &mdash; 7 deletions from wt_a then 7 from wt_b)</em>
    <span class="pp-del">--</span>AACCCC<br>
    AA<span class="pp-del">--</span>CCCC<br>
    AAAA<span class="pp-del">--</span>CC<br>
    AAAAC<span class="pp-del">--</span>C<br>
    <span class="pp-ellipsis">... (7 total from wt_a)</span><br>
    <span class="pp-del">--</span>GGTTTT<br>
    GG<span class="pp-del">--</span>TTTT<br>
    GGGG<span class="pp-del">--</span>TT<br>
    <span class="pp-ellipsis">... (7 total from wt_b)</span>
    </div>

Stack Then Generate a Library to Show Interleaved Draws
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After stacking, ``generate_library`` draws from the combined state space,
cycling through all states of all constituent pools in a single pass.

.. code-block:: python

    pool_a = pp.from_seqs(["AAAA", "CCCC"])
    pool_b = pp.from_seqs(["GGGG", "TTTT"])
    combined = pp.stack([pool_a, pool_b])
    df = combined.generate_library()
    print(df["seq"].tolist())
    # ['AAAA', 'CCCC', 'GGGG', 'TTTT']

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Library (4 sequences &mdash; all states from pool_a then pool_b, one pass)</em>
    AAAA<br>CCCC<br>GGGG<br>TTTT
    </div>

See :func:`~poolparty.stack`.