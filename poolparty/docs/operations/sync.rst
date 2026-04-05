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

Pair two pools 1:1
~~~~~~~~~~~~~~~~~~

Without ``sync``, joining two 3-state pools produces 9 sequences (Cartesian
product). After syncing, only the 3 matched pairs are drawn.

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

Sync scan results for matched comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Synchronize a deletion scan and a mutagenize scan so that each deletion
position is paired with a specific mutation draw.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    dels = wt.deletion_scan(deletion_length=1, mode="sequential")
    muts = pp.sample(wt.mutagenize(num_mutations=1, mode="random"), num_seqs=8, seed=0)

    pp.sync([dels, muts])
    combined = pp.join([dels, muts])
    combined.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">combined: seq_length=16, num_states=8</em>
    -CGTACGTACGAACGT<br>
    A-GTACGTACGTAGGT<br>
    AC-TACGTATGTACGT<br>
    ACG-ACGTACGTACTT<br>
    ACGT-CGTACTTACGT<br>
    ACGTA-GTGCGTACGT<br>
    ACGTAC-TACGTCCGT<br>
    ACGTACG-ACGAACGT
    </div>

See :func:`~poolparty.sync`.
