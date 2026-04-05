repeat
======

Repeat the state space of a pool a specified number of times, producing a
new pool whose total number of states is ``times`` &times; the input pool's
state count — each input state appears ``times`` times consecutively.

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
     - Input pool whose states will be repeated.
   * - ``times``
     - ``int``
     - *(required)*
     - Number of times to repeat each state.
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
     - Design card columns to include in library output.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.repeat` in the
   :doc:`API Reference </api>`.

Examples
--------

Repeat a Two-Sequence Pool Three Times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expand a two-sequence pool to six draws by repeating each state three times
in order.

.. code-block:: python

    pool    = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
    tripled = pp.repeat(pool, times=3)
    tripled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">tripled: seq_length=4, num_states=6</em>
    AAAA<br>
    AAAA<br>
    AAAA<br>
    CCCC<br>
    CCCC<br>
    CCCC
    </div>

Repeat a Scan Result to Get More Coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Repeat a deletion-scan pool to obtain additional draws of every variant,
useful when downstream counting requires a minimum number of observations per
sequence.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    dels    = wt.deletion_scan(deletion_length=2, mode="sequential")
    doubled = pp.repeat(dels, times=2)
    doubled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">doubled: seq_length=8, num_states=14</em>
    --CGATCG<br>
    --CGATCG<br>
    A--GATCG<br>
    A--GATCG<br>
    AT--ATCG<br>
    <span class="pp-ellipsis">... (14 total)</span>
    </div>

Operator shorthand (``*``)
~~~~~~~~~~~~~~~~~~~~~~~~~~

``pool * N`` is equivalent to ``pp.repeat(pool, times=N)`` — it creates a
new pool with ``N`` copies of the input pool's states. ``N * pool`` is
identical to ``pool * N``. This is useful for generating multiple replicates
of a stochastic draw.

.. code-block:: python

    wt       = pp.from_seq("ATCGATCG")
    shuffled = pp.shuffle_seq(wt, mode="random")
    rep      = shuffled * 5
    rep.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">rep: seq_length=8, num_states=5</em>
    TGGCAACT<br>
    AGTACGTC<br>
    GTTAAGCC<br>
    CACTTGGA<br>
    TCATGACG
    </div>

.. code-block:: python

    wt  = pp.from_seq("ATCGATCG")
    rep = 3 * pp.shuffle_seq(wt, mode="random")
    rep.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">rep: seq_length=8, num_states=3</em>
    TGGCAACT<br>
    AGTACGTC<br>
    GTTAAGCC
    </div>

See :func:`~poolparty.repeat`.
