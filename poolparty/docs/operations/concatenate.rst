State Operators (``+`` and ``*``)
==================================

PoolParty overloads Python's ``+`` and ``*`` operators on pools as
shorthand for two **state-level** operations:

- ``pool_a + pool_b`` is equivalent to ``pp.stack([pool_a, pool_b])`` —
  it creates a disjoint union of both pools' states so draws can come
  from either pool.
- ``pool * N`` is equivalent to ``pp.repeat(pool, N)`` — it creates a
  new pool with ``N`` states, each drawn independently from ``pool``.
- ``N * pool`` is identical to ``pool * N``.

These operators act on the *state dimension*, not on individual
sequences. To concatenate sequences end-to-end, use :func:`~poolparty.join`.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Examples
--------

Merge two libraries with ``+``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``+`` stacks the state lists of two pools into a single pool whose
library contains all sequences from both inputs. If ``pool_a`` has M
sequences and ``pool_b`` has N sequences, the result has M + N sequences.

.. code-block:: python

    wt   = pp.from_seq("ATCG")
    muts = pp.mutagenize(wt, num_mutations=1, mode="sequential")
    ctrl = pp.from_seqs(["AAAA", "TTTT"], mode="sequential")
    lib  = muts + ctrl   # all single-point mutants + 2 controls

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">muts (12 sequences &mdash; all single-point variants of ATCGATCG)</em>
    <span class="pp-mut">C</span>TCG<br>
    <span class="pp-mut">G</span>TCG<br>
    <span class="pp-ellipsis">... (9 total)</span>
    </div>
    <div class="pp-pool">
    <em class="pp-header">lib = muts + ctrl (14 sequences &mdash; 12 mutants then 2 controls)</em>
    <span class="pp-mut">C</span>TCG<br>
    <span class="pp-mut">G</span>TCG<br>
    <span class="pp-ellipsis">... (12 mutants, then)</span><br>
    AAAA<br>TTTT
    </div>

Chain multiple pools with repeated ``+``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each ``+`` appends another pool's states to the combined library.

.. code-block:: python

    a = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
    b = pp.from_seqs(["GGGG"], mode="sequential")
    c = pp.from_seqs(["TTTT", "ACGT"], mode="sequential")
    lib = a + b + c

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">lib (5 sequences &mdash; states from a, then b, then c)</em>
    AAAA<br>CCCC<br>GGGG<br>TTTT<br>ACGT
    </div>

Repeat a pool's states with ``pool * N``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pool * N`` creates a new pool with ``N`` states, each independently
sampled from ``pool``. Useful for generating multiple replicates of a
stochastic draw.

.. code-block:: python

    wt       = pp.from_seq("ATCGATCG")
    shuffled = pp.shuffle_seq(wt)        # stochastic: one permutation per draw
    rep      = shuffled * 5              # draw 5 independent shuffles

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">rep (5 states &mdash; 5 independent draws from shuffle_seq)</em>
    CTAGATCG<br>
    GATCATCG<br>
    TCGAATCG<br>
    AGCTTCGA<br>
    CGATTCAG
    </div>

Left-multiplication ``N * pool``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``N * pool`` is identical to ``pool * N``.

.. code-block:: python

    wt  = pp.from_seq("ATCGATCG")
    rep = 3 * pp.shuffle_seq(wt)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">rep (3 states &mdash; same as shuffle_seq(wt) * 3)</em>
    TCGAATCG<br>
    GATCTCGA<br>
    CTAGATCG
    </div>

See :func:`~poolparty.stack`, :func:`~poolparty.repeat`.
