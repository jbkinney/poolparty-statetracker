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

Examples
--------

Repeat a Two-Sequence Pool Three Times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expand a two-sequence pool to six draws by repeating each state three times
in order.

.. code-block:: python

    pool    = pp.from_seqs(["AAAA", "CCCC"])
    tripled = pp.repeat(pool, times=3)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (6 sequences &mdash; each of 2 states repeated 3&times;)</em>
    AAAA<br>AAAA<br>AAAA<br>CCCC<br>CCCC<br>CCCC
    </div>

Repeat a Scan Result to Get More Coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Repeat a deletion-scan pool to obtain additional draws of every variant,
useful when downstream counting requires a minimum number of observations per
sequence.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    dels    = wt.deletion_scan(deletion_length=2)   # 7 states
    doubled = pp.repeat(dels, times=2)              # 14 states

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (14 sequences &mdash; 7-state deletion scan repeated 2&times;)</em>
    <span class="pp-del">--</span>CGATCG<br>
    A<span class="pp-del">--</span>GATCG<br>
    AT<span class="pp-del">--</span>ATCG<br>
    ATC<span class="pp-del">--</span>TCG<br>
    ATCG<span class="pp-del">--</span>CG<br>
    ATCGA<span class="pp-del">--</span>G<br>
    ATCGAT<span class="pp-del">--</span><br>
    <span class="pp-del">--</span>CGATCG<br>
    A<span class="pp-del">--</span>GATCG<br>
    <span class="pp-ellipsis">... (14 total &mdash; all 7 deletion variants repeated twice)</span>
    </div>

Difference Between ``repeat`` and the ``*`` Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``repeat(pool, times=N)`` repeats *states* — each sequence is drawn N times
as a separate library entry. The ``*`` operator concatenates the *sequence*
with itself N times end-to-end, producing a longer sequence in a single state.

.. code-block:: python

    base = pp.from_seqs(["AC", "GT"])

    # repeat: 4 states, each original sequence drawn twice
    state_repeated = pp.repeat(base, times=2)

    # * operator: 2 states, each sequence concatenated with itself twice
    seq_repeated = base * 2

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">state_repeated &mdash; repeat(base, times=2): 4 states, 2-nt sequences</em>
    AC<br>AC<br>GT<br>GT
    </div>

    <div class="pp-pool">
    <em class="pp-header">seq_repeated &mdash; base * 2: 2 states, 4-nt sequences (concatenated)</em>
    ACAC<br>GTGT
    </div>

See :func:`~poolparty.repeat`.