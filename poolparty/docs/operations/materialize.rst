materialize
===========

Eagerly generate sequences from a pool and cache them in a new, standalone
pool whose state space is exactly the set of stored sequences. The resulting
pool has no parent references (severed DAG), so it can be used as a cheap
starting point for any number of independent downstream pipelines.

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
     - Input pool to materialize.
   * - ``num_seqs``
     - ``int | None``
     - ``None``
     - Number of sequences to generate and cache. Provide either
       ``num_seqs`` or ``num_cycles``.
   * - ``num_cycles``
     - ``int | None``
     - ``None``
     - Number of complete cycles through the state space.
   * - ``seed``
     - ``int | None``
     - ``None``
     - Random seed for reproducible generation.
   * - ``discard_null_seqs``
     - ``bool``
     - ``True``
     - If ``True``, skip filtered-out (``NullSeq``) sequences.
   * - ``max_iterations``
     - ``int | None``
     - ``None``
     - Maximum iterations before stopping (useful with filters that reject
       most draws).
   * - ``min_acceptance_rate``
     - ``float | None``
     - ``None``
     - If the acceptance rate drops below this threshold, generation stops
       early.
   * - ``attempts_per_rate_assessment``
     - ``int``
     - ``100``
     - Number of draws between acceptance-rate checks.
   * - ``name``
     - ``str | None``
     - ``None``
     - Name for the materialized pool.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.materialize` in the
   :doc:`API Reference </api>`.

Examples
--------

Materialize before applying downstream scans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-compute an expensive mutagenize result once and reuse it across multiple
scan operations without re-running the mutation logic each time.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    mutants = pp.mutagenize(wt, num_mutations=1)

    # Freeze 20 mutants into a standalone pool
    cached  = pp.materialize(mutants, num_seqs=20, seed=42)

    # Apply different downstream scans to the same cached pool
    scan_a  = pp.deletion_scan(cached, deletion_length=2)
    scan_b  = pp.mutagenize(cached, num_mutations=1)

    df_a    = pp.generate_library(scan_a, num_seqs=6)
    df_b    = pp.generate_library(scan_b, num_seqs=6)

    cached.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">cached: seq_length=8, num_states=20</em>
    ATCGAACG<br>
    ACCGATCG<br>
    ATCGATCT<br>
    ATCGACCG<br>
    ATCGAGCG<br>
    <span class="pp-ellipsis">... (20 total)</span>
    </div>

Reproducible caching with ``seed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``seed=`` so that re-running the same script produces the same
materialized pool every time.

.. code-block:: python

    wt     = pp.from_seq("ATCGATCG")
    pool   = pp.mutagenize(wt, num_mutations=1, mode="random")
    cached = pp.materialize(pool, num_seqs=5, seed=0)
    cached.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">cached: seq_length=8, num_states=5</em>
    ATCGGTCG<br>
    ATCGAACG<br>
    ATCGCTCG<br>
    GTCGATCG<br>
    ACCGATCG
    </div>

Materialize then apply a deletion scan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because ``materialize`` returns a standalone pool with no upstream parents,
chaining a deletion scan is just as efficient as starting from a plain
``from_seqs`` pool.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    mutants = pp.mutagenize(wt, num_mutations=1)

    # Snapshot 8 mutants once; subsequent operations cost nothing extra
    cached  = pp.materialize(mutants, num_seqs=8, seed=1)

    # Systematically delete 2-base windows across every cached sequence
    scan    = pp.deletion_scan(cached, deletion_length=2)
    df      = pp.generate_library(scan, num_seqs=8)

    cached.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">cached: seq_length=8, num_states=8</em>
    ATCGGTCG<br>
    ATGGATCG<br>
    ATCGATCA<br>
    ATCGATCT<br>
    ATCGATCC<br>
    <span class="pp-ellipsis">... (8 total)</span>
    </div>

See :func:`~poolparty.materialize` or :meth:`~poolparty.Pool.materialize`.
