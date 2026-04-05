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
   :widths: auto
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

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.sample` in the
   :doc:`API Reference </api>`.

Examples
--------

Sample 5 Sequences from a 256-Sequence Pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Draw a small random subset from all 256 4-mers to obtain a manageable
representative sample. Without a seed, which sequences appear changes on each
evaluation.

.. code-block:: python

    import poolparty as pp
    pp.init()

    kmers  = pp.get_kmers(length=4, mode="sequential")
    subset = pp.sample(kmers, num_seqs=5)
    subset.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">subset: seq_length=4, num_states=5</em>
    AGTA<br>
    TTGA<br>
    ATCG<br>
    GGTA<br>
    AGGC
    </div>

Sample with a Fixed Seed for Reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide a seed to guarantee the same subset is selected every time the
pipeline is evaluated.

.. code-block:: python

    import poolparty as pp
    pp.init()

    kmers  = pp.get_kmers(length=4, mode="sequential")
    subset = pp.sample(kmers, num_seqs=5, seed=42)
    subset.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">subset: seq_length=4, num_states=5</em>
    GGAT<br>
    AACG<br>
    CACG<br>
    ATGC<br>
    GTTA
    </div>

Sample More Sequences Than the Pool Has States (Cycling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``num_seqs`` exceeds the pool's state count and ``with_replacement=True``
(the default), states are resampled with replacement so the requested count is
always honoured.

.. code-block:: python

    import poolparty as pp
    pp.init()

    small  = pp.from_seqs(["AAAA", "CCCC", "GGGG"], mode="sequential")
    large  = pp.sample(small, num_seqs=9, seed=0)
    large.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">large: seq_length=4, num_states=9</em>
    GGGG<br>
    GGGG<br>
    CCCC<br>
    AAAA<br>
    CCCC<br>
    CCCC<br>
    GGGG<br>
    AAAA<br>
    CCCC
    </div>

Sample from a Stochastic Pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``sample`` on a mutagenized pool to select a reproducible subset of
stochastic draws, combining random mutation with deterministic sampling. The
mutagenized sequences themselves still vary between runs unless the upstream
stochastic pool is seeded.

.. code-block:: python

    import poolparty as pp
    pp.init()

    wt      = pp.from_seq("ATCGATCG")
    mutants = wt.mutagenize(num_mutations=1)
    sampled = pp.sample(mutants, num_seqs=4, seed=7)
    sampled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">sampled: seq_length=8, num_states=4</em>
    ATCGACCG<br>
    ATCGTTCG<br>
    ATGGATCG<br>
    GTCGATCG
    </div>

See :func:`~poolparty.sample`.
