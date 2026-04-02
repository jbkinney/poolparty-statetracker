sample
======

Draw a fixed number of sequences from a pool's state space, optionally with a
random seed for reproducibility or with cycling when more sequences are
requested than the pool contains.

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
     - Input pool to sample from.
   * - ``num_seqs``
     - ``int | None``
     - ``None``
     - Number of sequences to draw. Provide either ``num_seqs`` or
       ``seq_states``.
   * - ``seq_states``
     - ``list[int] | None``
     - ``None``
     - Explicit list of state indices to select. Overrides ``num_seqs``.
   * - ``seed``
     - ``int | None``
     - ``None``
     - Random seed for reproducible sampling.
   * - ``with_replacement``
     - ``bool``
     - ``True``
     - If ``True``, states may be drawn more than once when ``num_seqs``
       exceeds the pool's state count.
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

Sample 5 Sequences from a 256-Sequence Pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Draw a small random subset from all 256 4-mers to obtain a manageable
representative sample.

.. code-block:: python

    kmers  = pp.get_kmers(length=4, alphabet="ACGT")  # 256 states
    subset = pp.sample(kmers, num_seqs=5)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (5 sequences &mdash; random sample from 256 4-mers, no fixed seed)</em>
    GCTA<br>TTAC<br>CAGG<br>AGCT<br>TGCA
    </div>

Sample with a Fixed Seed for Reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide a seed to guarantee the same subset is selected every time the
pipeline is evaluated.

.. code-block:: python

    kmers  = pp.get_kmers(length=4, alphabet="ACGT")
    subset = pp.sample(kmers, num_seqs=5, seed=42)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (5 sequences &mdash; deterministic sample, seed=42)</em>
    CGTA<br>AGTC<br>TTGA<br>GCAC<br>ATCG
    </div>

Sample More Sequences Than the Pool Has States (Cycling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``num_seqs`` exceeds the pool's state count and ``with_replacement=True``
(the default), states are resampled with replacement so the requested count is
always honoured.

.. code-block:: python

    small  = pp.from_seqs(["AAAA", "CCCC", "GGGG"])  # 3 states
    large  = pp.sample(small, num_seqs=9, seed=0)     # 9 draws with replacement

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (9 sequences &mdash; 3-state pool sampled 9&times; with replacement, seed=0)</em>
    AAAA<br>GGGG<br>AAAA<br>CCCC<br>GGGG<br>AAAA<br>CCCC<br>GGGG<br>CCCC
    </div>

Sample from a Stochastic Pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``sample`` on a mutagenized pool to select a reproducible subset of
stochastic draws, combining random mutation with deterministic sampling.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    mutants = wt.mutagenize(num_mutations=1)        # stochastic pool
    sampled = pp.sample(mutants, num_seqs=4, seed=7)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (4 sequences &mdash; 4 reproducible draws from mutagenize, seed=7)</em>
    A<span class="pp-mut">G</span>CGATCG<br>
    ATCG<span class="pp-mut">C</span>TCG<br>
    ATCGAT<span class="pp-mut">A</span>G<br>
    <span class="pp-mut">C</span>TCGATCG
    </div>

See :func:`~poolparty.sample`.
