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
   :widths: auto
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
     - Enumeration order when combined with other pools.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.state_shuffle` in the
   :doc:`API Reference </api>`.

Examples
--------

Shuffle All 16 2-mer States with a Fixed Seed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reorder the 16 2-mers produced by ``get_kmers`` using ``seed=0`` so that the
permutation is deterministic and reproducible.

.. code-block:: python

    kmers    = pp.get_kmers(length=2, mode="sequential")
    shuffled = pp.state_shuffle(kmers, seed=0)
    shuffled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">shuffled: seq_length=2, num_states=16</em>
    GG<br>
    TG<br>
    CC<br>
    AC<br>
    GC
    <span class="pp-ellipsis">... (16 total)</span>
    </div>

Shuffle Then Slice to Get a Random Subset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine ``state_shuffle`` with ``state_slice`` to draw a reproducible random
subset without replacement — equivalent to a seeded random sample.

.. code-block:: python

    kmers   = pp.get_kmers(length=2, mode="sequential")
    shuffled = pp.state_shuffle(kmers, seed=0)
    subset  = pp.state_slice(shuffled, slice(0, 6))
    subset.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">subset: seq_length=2, num_states=6</em>
    GG<br>
    TG<br>
    CC<br>
    AC<br>
    GC<br>
    AG
    </div>

Two Shuffles with Different Seeds Produce Different Orders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changing the seed produces an independent permutation, demonstrating that the
output order is fully determined by the seed value.

.. code-block:: python

    kmers      = pp.get_kmers(length=2, mode="sequential")
    shuffled_0 = pp.state_shuffle(kmers, seed=0)
    shuffled_0.print_library()
    shuffled_1 = pp.state_shuffle(kmers, seed=1)
    shuffled_1.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">shuffled_0: seq_length=2, num_states=16</em>
    GG<br>
    TG<br>
    CC<br>
    AC<br>
    GC
    <span class="pp-ellipsis">... (16 total)</span>
    </div>

    <div class="pp-pool">
    <em class="pp-header">shuffled_1: seq_length=2, num_states=16</em>
    AG<br>
    GG<br>
    AA<br>
    TG<br>
    CG
    <span class="pp-ellipsis">... (16 total)</span>
    </div>

See :func:`~poolparty.state_shuffle`.
