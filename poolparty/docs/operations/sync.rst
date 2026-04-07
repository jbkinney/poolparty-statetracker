sync
====

Synchronize multiple pools so they iterate in lockstep — state *i* of every
synced pool is always drawn together. This is an **in-place** operation
(it modifies the pools and returns ``None``).

.. code-block:: python

    import poolparty as pp
    pp.init()

.. note::

   All synced pools must have the **same number of states**, and no pool
   may be an ancestor of another in the DAG (which would create a circular
   dependency).

----

Parameters
----------

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pools``
     - ``list[Pool]``
     - *(required)*
     - Pools to synchronize. Must all have the same ``num_states``.

Returns ``None`` — pools are modified in-place.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.sync` in the
   :doc:`API Reference </api>`.

Examples
--------

Without sync: Cartesian product
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, joining two pools produces every combination of their states.
Two 3-state pools yield 3 × 3 = 9 sequences.

.. code-block:: python

    a = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")
    b = pp.from_seqs(["TTT", "AAA", "CCC"], mode="sequential")

    cross = pp.join([a, b])
    cross.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">cross: seq_length=6, num_states=9</em>
    AAATTT<br>
    AAAAAA<br>
    AAACCC<br>
    CCCTTT<br>
    CCCAAA<br>
    CCCCCC<br>
    GGGTTT<br>
    GGGAAA<br>
    GGGCCC
    </div>

With sync: matched pairs
~~~~~~~~~~~~~~~~~~~~~~~~~

After syncing, the same join produces only the 3 matched pairs. State 0
of ``left`` is always drawn together with state 0 of ``right``, and so on.

.. code-block:: python

    left  = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")
    right = pp.from_seqs(["TTT", "AAA", "CCC"], mode="sequential")

    pp.sync([left, right])
    paired = pp.join([left, right])
    paired.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">paired: seq_length=6, num_states=3</em>
    AAATTT<br>
    CCCAAA<br>
    GGGCCC
    </div>

Pair variants with identifiers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common use case is to assign each variant a unique identifier sequence
so that the pairing is fixed across the library.

.. code-block:: python

    variants = pp.from_seqs(["ATCG", "GCTA", "TTAA", "CCGG"],
                            mode="sequential")
    tags     = pp.from_seqs(["AAAA", "CCCC", "GGGG", "TTTT"],
                            mode="sequential")

    pp.sync([variants, tags])
    library = pp.join([variants, tags])
    library.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">library: seq_length=8, num_states=4</em>
    ATCGAAAA<br>
    GCTACCCC<br>
    TTAAGGGG<br>
    CCGGTTTT
    </div>

Sync three or more pools
~~~~~~~~~~~~~~~~~~~~~~~~~

``sync`` accepts any number of pools. Here three 3-state pools are
synchronized, reducing the join from 27 combinations to 3 matched triples.

.. code-block:: python

    a = pp.from_seqs(["AAAA", "CCCC", "GGGG"], mode="sequential")
    b = pp.from_seqs(["TTTT", "AAAT", "CCCT"], mode="sequential")
    c = pp.from_seqs(["ACGT", "TGCA", "GATC"], mode="sequential")

    pp.sync([a, b, c])
    combined = pp.join([a, b, c])
    combined.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">combined: seq_length=12, num_states=3</em>
    AAAATTTTACGT<br>
    CCCCAAATTGCA<br>
    GGGGCCCTGATC
    </div>

See :func:`~poolparty.sync`.
