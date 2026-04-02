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
   :widths: 20 18 12 50
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

Examples
--------

Pair two pools 1:1
~~~~~~~~~~~~~~~~~~

Without ``sync``, joining two 3-state pools produces 9 sequences (Cartesian
product). After syncing, only the 3 matched pairs are drawn.

.. code-block:: python

    left  = pp.from_seqs(["AAA", "CCC", "GGG"])
    right = pp.from_seqs(["TTT", "AAA", "CCC"])

    pp.sync([left, right])
    paired = pp.join([left, right])
    df     = paired.generate_library()
    # 3 rows: AAATTT, CCCAAA, GGGCCC

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (3 sequences &mdash; synced left + right, 1:1 pairing)</em>
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
    dels = wt.deletion_scan(deletion_length=1, mode="sequential")  # 8 states
    muts = pp.sample(wt.mutagenize(num_mutations=1), num_seqs=8, seed=0)

    pp.sync([dels, muts])
    combined = pp.join([dels, muts])
    df       = combined.generate_library()
    # 8 rows — each deletion variant paired with a specific mutant

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (8 sequences &mdash; synced deletion + mutation, matched by state)</em>
    <span class="pp-del">-</span>CGTACGT&nbsp;A<span class="pp-mut">G</span>GTACGT<br>
    A<span class="pp-del">-</span>GTACGT&nbsp;ACGT<span class="pp-mut">T</span>CGT<br>
    AC<span class="pp-del">-</span>TACGT&nbsp;<span class="pp-mut">G</span>CGTACGT<br>
    <span class="pp-ellipsis">... (8 total &mdash; each pair drawn together)</span>
    </div>

See :func:`~poolparty.sync`.
