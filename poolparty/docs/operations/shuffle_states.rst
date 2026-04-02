shuffle\_states
===============

Randomly permute the order of a pool's state space, optionally with a fixed
seed for deterministic shuffling, while leaving the sequences themselves
unchanged.

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
   * - ``pool``
     - ``Pool``
     - *(required)*
     - Input pool whose state order will be shuffled.
   * - ``seed``
     - ``int | None``
     - ``None``
     - Random seed for deterministic shuffling. ``None`` uses a random
       seed.
   * - ``permutation``
     - ``list[int] | None``
     - ``None``
     - Explicit permutation of state indices. Overrides ``seed`` when
       provided.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Iteration priority for downstream multi-pool iteration.

----

Examples
--------

Shuffle All 16 2-mer States with a Fixed Seed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reorder the 16 2-mers produced by ``get_kmers`` using ``seed=0`` so that the
permutation is deterministic and reproducible.

.. code-block:: python

    kmers    = pp.get_kmers(length=2, alphabet="ACGT")  # 16 states in order
    shuffled = pp.state_shuffle(kmers, seed=0)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (16 sequences &mdash; all 2-mers in shuffled order, seed=0)</em>
    TG<br>AT<br>GA<br>CT<br>CG<br>TA<br>GC<br>CA<br>TC<br>GT<br>GG<br>CC<br>AA<br>TT<br>AC<br>AG
    </div>

Shuffle Then Slice to Get a Random Subset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine ``state_shuffle`` with ``state_slice`` to draw a reproducible random
subset without replacement — equivalent to a seeded random sample.

.. code-block:: python

    kmers   = pp.get_kmers(length=2, alphabet="ACGT")
    shuffled = pp.state_shuffle(kmers, seed=0)
    subset  = pp.state_slice(shuffled, slice(0, 6))  # first 6 of the shuffle

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (6 sequences &mdash; first 6 states of seed=0 shuffle, no replacement)</em>
    TG<br>AT<br>GA<br>CT<br>CG<br>TA
    </div>

Two Shuffles with Different Seeds Produce Different Orders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changing the seed produces an independent permutation, demonstrating that the
output order is fully determined by the seed value.

.. code-block:: python

    kmers      = pp.get_kmers(length=2, alphabet="ACGT")
    shuffled_0 = pp.state_shuffle(kmers, seed=0)
    shuffled_1 = pp.state_shuffle(kmers, seed=1)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">shuffled_0 &mdash; seed=0 (first 6 states shown)</em>
    TG<br>AT<br>GA<br>CT<br>CG<br>TA<br>
    <span class="pp-ellipsis">... (16 total)</span>
    </div>

    <div class="pp-pool">
    <em class="pp-header">shuffled_1 &mdash; seed=1 (first 6 states shown, different order)</em>
    AC<br>TT<br>GG<br>CA<br>AG<br>TC<br>
    <span class="pp-ellipsis">... (16 total, independent permutation)</span>
    </div>

See :func:`~poolparty.state_shuffle`.
