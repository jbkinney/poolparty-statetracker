Pools
=====

A **Pool** represents a designed collection of DNA sequences. A Pool
can represent the final library you wish to generate, or an intermediate
set of sequences used to construct it. Pools are *lazy*: they record
which operations to apply and to what inputs, forming a directed acyclic
graph (DAG), but no sequences are generated until you explicitly request
them. This means you can explore and test multiple design options
without triggering expensive computations.

Pools are also *immutable*: every operation returns a new Pool, leaving
the original unchanged. You can branch a pipeline at any point and apply
different operations to each branch without interference.

The final Pool in a pipeline -- the one from which you generate
sequences -- is called the *root Pool*. The DAG rooted at this Pool
describes the high-level logic used to generate your library; PoolParty
handles the procedural details and bookkeeping internally.

----

Context management
------------------

Pools must be created inside an active context. Call ``pp.init()`` before
each independent library design to initialize a fresh context:

.. code-block:: python

    import poolparty as pp
    pp.init()

If you design multiple libraries in one script, call ``pp.init()`` again
before starting each new design. For scoped contexts (e.g., inside a
reusable function), use ``with pp.Party():`` instead -- the context is
automatically cleaned up when the block exits:

.. code-block:: python

    with pp.Party():
        pool = pp.from_seq("ACGT")
        # ... build and export library ...
    # context is released here

All remaining examples on this page assume the ``import`` and
``pp.init()`` calls above have been run.

----

Properties
----------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``name``
     - ``str``
     - Human-readable name for this pool. Settable. Defaults to ``"pool[N]"``.
   * - ``num_states``
     - ``int``
     - Number of states this pool has (the total across the entire pipeline).
       This is an upper bound on the number of *distinct* sequences, not the
       number itself -- see the note below. Use :ref:`stats <pool-stats>` to
       count them.
   * - ``seq_length``
     - ``int | None``
     - Fixed sequence length, or ``None`` for variable-length pools.
   * - ``regions``
     - ``set[Region]``
     - Set of :class:`~poolparty.Region` objects present in this pool's sequences.
       See :doc:`regions` for details.
   * - ``parents``
     - ``list[Pool]``
     - Input pools that this pool's operation reads from.
   * - ``operation``
     - ``Operation``
     - The operation that created this pool. Exposes ``operation.mode``,
       ``operation.num_states``, and ``operation.natural_num_states``.

Internally, each sequence is identified by a *state* -- an integer that,
together with a random seed, uniquely determines the sequence content.
``pool.num_states`` is the total number of distinct states the pool can produce.

Distinct states do not always mean distinct sequences. Two states can yield the
same sequence, ``repeat`` asks for copies deliberately, and ``filter`` leaves a
``NullSeq`` behind rather than removing a state. A random operation built
*without* ``num_states`` draws afresh for every sequence, so it contributes one
state and ``pool.num_states`` becomes a floor rather than a total. In every case
:ref:`stats <pool-stats>` reports what the pool actually produced.

Note that ``pool.num_states`` and ``pool.operation.num_states`` are different
values. The pool's ``num_states`` is the total across the entire pipeline,
while the operation's ``num_states`` is just that operation's contribution
(see :doc:`operations/modes` and :doc:`operations/library_size`):

.. code-block:: python

    seqs = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")
    mut  = seqs.mutagenize(num_mutations=1, mode="sequential")

    mut.num_states                    # 27 (3 inputs × 9 mutants)
    mut.operation.num_states          # 9  (mutagenize alone)
    mut.operation.natural_num_states  # 9  (before any num_states override)

----

Naming pools
------------

``named(name)``
~~~~~~~~~~~~~~~

Set the pool's name and return ``self``, allowing in-line renaming without
breaking a chain:

.. code-block:: python

    wt = pp.from_seq("ACGT").named("wildtype")
    # wt.name == "wildtype"

    scored = (
        pp.from_iupac("NNNN", mode="sequential")
          .mutagenize(num_mutations=1)
          .named("single_mut")
    )

Pool names appear in ``print_library`` headers and ``print_dag`` output.
This is distinct from ``prefix``, which labels individual *sequence names*
in the output DataFrame (see :doc:`metadata/naming`).

----

Previewing sequences — ``print_library(...)``
----------------------------------------------

Print a formatted preview of the pool's sequences to stdout. Returns ``self``
so it can be used mid-pipeline.

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``num_seqs``
     - ``int | None``
     - ``None``
     - Number of sequences to show.
   * - ``show_header``
     - ``bool``
     - ``True``
     - Print a summary header line before the sequences.
   * - ``show_name``
     - ``bool``
     - ``True``
     - Include the sequence name column.
   * - ``show_state``
     - ``bool``
     - ``False``
     - Include the state index column.
   * - ``seed``
     - ``int | None``
     - ``None``
     - Random seed for reproducible previews.

See :class:`~poolparty.Pool` in the :doc:`api` for the full parameter list.

.. code-block:: python

    pp.from_iupac("NNNNN", mode="sequential").print_library(num_seqs=6)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool[0]: seq_length=5, num_states=1024</em>
    pool[0].0  AAAAA<br>
    pool[0].1  AAAAC<br>
    pool[0].2  AAAAG<br>
    pool[0].3  AAAAT<br>
    pool[0].4  AAACA<br>
    pool[0].5  AAACC<br>
    </div>

----

Generating libraries — ``generate_library(...)``
-------------------------------------------------

Generate all sequences from this pool and return them as a
:class:`pandas.DataFrame`. Best for small to medium pools; for libraries above ~10k
sequences, use ``to_df`` which streams in chunks.

.. code-block:: python

    pool = pp.from_iupac("NNNN", mode="sequential")
    df   = pool.generate_library()
    # df has columns: name, seq  (plus any design card columns)

See :doc:`operations/generate_library` for full documentation.

----

Exporting to a DataFrame — ``to_df(...)``
-----------------------------------------

Generate sequences and collect them into a :class:`pandas.DataFrame` using
chunked streaming. Prefer ``to_df`` over ``generate_library`` for large
libraries (above ~10k sequences). It processes sequences in batches, keeping
peak memory proportional to ``chunk_size`` rather than the full library.

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``num_seqs``
     - ``int | None``
     - ``None``
     - Total sequences to generate. Required when ``num_cycles`` is not given.
   * - ``num_cycles``
     - ``int | None``
     - ``None``
     - Number of complete passes through the pool's ``num_states`` sequences.
       One cycle produces ``num_states`` sequences.
   * - ``chunk_size``
     - ``int``
     - ``1000``
     - Sequences generated per internal batch. Larger values may be faster
       but use more memory.
   * - ``write_tags``
     - ``bool``
     - ``False``
     - If ``True``, include region tags (e.g. ``<region>…</region>``) in
       the ``seq`` column.
   * - ``seed``
     - ``int | None``
     - ``None``
     - Random seed for reproducibility.
   * - ``discard_null_seqs``
     - ``bool``
     - ``True``
     - Skip sequences removed by a ``filter`` operation (``NullSeq``).
   * - ``columns``
     - ``list[str] | None``
     - ``None``
     - Columns to keep. Defaults to all columns (``name``, ``seq``, plus
       any design card columns). Pass ``["name", "seq"]`` to drop cards.
   * - ``show_progress``
     - ``bool``
     - ``True``
     - Display a ``tqdm`` progress bar during generation.

See :class:`~poolparty.Pool` in the :doc:`api` for the full parameter list.

.. rubric:: Basic usage

.. code-block:: python

    pool = pp.from_iupac("NNNNNNNN", mode="sequential")
    df   = pool.to_df(num_cycles=1)
    # 65536 rows, columns: name, seq

.. rubric:: Large library with chunked streaming

.. code-block:: python

    pool = pp.from_iupac("NNNNNNNNNN")
    df   = pool.to_df(num_seqs=500_000, chunk_size=10_000, seed=42) # Random sample of 500k sequences from ~1M possible sequences

.. rubric:: Keep only name and seq (drop design cards)

.. code-block:: python

    scored = pool.score(pp.calc_gc, card_key="gc", cards={"gc": "gc"})
    df     = scored.to_df(num_cycles=1, columns=["name", "seq"])
    # "gc" column is excluded

----

Exporting to file — ``to_file(...)``
-------------------------------------

Stream sequences directly to disk without ever holding the full library in
memory. Supports CSV, TSV, FASTA, and JSONL formats, including gzip
compression.

.. list-table::
   :widths: 25 18 15 42
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``path``
     - ``str | Path``
     - *(required)*
     - Output file path. Use a ``.gz`` suffix for transparent gzip
       compression (e.g. ``library.csv.gz``).
   * - ``file_type``
     - ``str | None``
     - ``None``
     - ``"csv"``, ``"tsv"``, ``"fasta"``, or ``"jsonl"``. Auto-detected
       from the file extension when ``None``.
   * - ``num_seqs``
     - ``int | None``
     - ``None``
     - Total sequences to write.
   * - ``num_cycles``
     - ``int | None``
     - ``None``
     - Number of complete passes through the pool's ``num_states`` sequences.
       One cycle produces ``num_states`` sequences.
   * - ``chunk_size``
     - ``int``
     - ``1000``
     - Sequences written per internal batch.
   * - ``write_tags``
     - ``bool``
     - ``False``
     - Include region tags in output sequences.
   * - ``seed``
     - ``int | None``
     - ``None``
     - Random seed for reproducibility.
   * - ``discard_null_seqs``
     - ``bool``
     - ``True``
     - Skip sequences removed by a ``filter`` operation (``NullSeq``).
   * - ``columns``
     - ``list[str] | None``
     - ``None``
     - Columns to write (CSV/TSV only).
   * - ``line_width``
     - ``int | None``
     - ``60``
     - FASTA only: wrap sequence lines at this width. ``None`` for no
       wrapping.
   * - ``description``
     - ``str | callable | None``
     - ``None``
     - FASTA only: additional description text after the sequence name.
       A string is treated as a format template (e.g. ``"GC={gc:.2f}"``);
       a callable receives the row dict and should return a string.
   * - ``show_progress``
     - ``bool``
     - ``True``
     - Show a ``tqdm`` progress bar.

Returns the number of sequences written. See :class:`~poolparty.Pool` in the
:doc:`api` for the full parameter list.

.. rubric:: Export to CSV

.. code-block:: python

    pool = pp.from_iupac("NNNNNNNN")
    n    = pool.to_file("library.csv", num_seqs=100_000)
    # n == 100000

.. code-block:: text

    name,seq
    pool[0].0,AAAAAAAA
    pool[0].1,AAAAAAAC
    pool[0].2,AAAAAAAG
    pool[0].3,AAAAAAAT
    pool[0].4,AAAAAACA
    ...

.. rubric:: Export to gzip-compressed CSV

.. code-block:: python

    n = pool.to_file("library.csv.gz", num_seqs=1_000_000, chunk_size=50_000)

.. rubric:: Export to FASTA

.. code-block:: python

    n = pool.to_file("library.fasta", num_seqs=10_000)

.. code-block:: text

    >pool[0].0
    AAAAAAAA
    >pool[0].1
    AAAAAAAC
    >pool[0].2
    AAAAAAAG
    ...

.. rubric:: FASTA with a custom description line

.. code-block:: python

    scored = pool.score(pp.calc_gc, card_key="gc", cards={"gc": "gc"})
    n = scored.to_file(
        "library.fasta",
        num_seqs=1000,
        description=lambda row: f"GC={row['gc']:.3f}",
    )

.. code-block:: text

    >pool[0].0 GC=0.000
    AAAAAAAA
    >pool[0].1 GC=0.125
    AAAAAAAC
    >pool[0].2 GC=0.125
    AAAAAAAG
    ...

----

.. _pool-stats:

Summarising a library — ``stats(...)``
--------------------------------------

Generate the library and report on it: how many sequences are unique and how
many are duplicates, how far apart they are, and how often they carry features
that complicate synthesis. Nothing about the pool or its library changes.

.. code-block:: python

    seqs = ["AAAATTTT", "GGGGCCCC", "AATTAATT", "GGCCGGCC", "ACGTACGT"]
    pool = pp.from_seqs(seqs, mode="sequential").filter_gc(max_gc=0.5)
    print(pool.stats())

.. code-block:: text

    pool.stats()  -  5 of 5 sequences in the design

    Composition
      design size (num_states)     5
      generated                    5
      filtered out                 2
      unique sequences             3
      duplicate sequences          0   (0.0%)
      most-repeated sequence       1 copy

    Length
      min / max                    8 / 8

    GC content
      min / mean / max             0.000 / 0.167 / 0.500

    Homopolymer runs
      longest run                  4
      sequences with a long run    0.0%

    Repetitiveness (DUST)
      mean / max                   0.33 / 0.33

    Pairwise distance (Hamming)
      min / mean / max             4 / 4.7 / 6

The result is a ``dict``, so individual numbers can be read out, saved as JSON,
or collected into a table:

.. code-block:: python

    pool.stats()["frac_duplicate_seqs"]
    pd.DataFrame([left.stats(), right.stats()])

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``num_seqs``
     - ``int | None``
     - ``None``
     - Generate exactly this many sequences.
   * - ``num_cycles``
     - ``int | None``
     - ``None``
     - Generate this many complete passes through the state space.
       ``num_cycles=1`` means the whole design.
   * - ``seed``
     - ``int | None``
     - ``None``
     - Random seed. Fixes both the sequences a randomly-sampling pool produces
       and the subsample used for pairwise distances.
   * - ``max_hamming_seqs``
     - ``int | None``
     - ``2000``
     - Compare at most this many sequences pairwise. ``None`` skips the
       distance statistics.
   * - ``max_homopolymer_run``
     - ``int | None``
     - ``6``
     - Longest single-base run to tolerate. ``None`` omits
       ``frac_seqs_with_long_homopolymer``.
   * - ``enzymes``
     - ``list[str] | None``
     - ``None``
     - Restriction enzyme names, or preset names such as ``'golden_gate'``.
   * - ``sites``
     - ``list[str] | None``
     - ``None``
     - Recognition sequences to look for, IUPAC codes allowed.
   * - ``show_progress``
     - ``bool``
     - ``True``
     - Show a progress bar while generating.

----

How much of the design is measured
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pool records how to build a library rather than the library itself, and
duplicates cannot be counted in a recipe, so ``stats`` generates sequences
before measuring anything. How many it generates depends on the design:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Design
     - ``pool.stats()``
   * - Fixed size, at most 1,000,000 sequences
     - Measures all of it.
   * - Fixed size, more than 1,000,000 sequences
     - Raises, so an accidental call cannot start an enormous job. Pass
       ``num_seqs=`` to measure part of it, or ``num_cycles=1`` to ask for all
       of it deliberately.
   * - No fixed size (a random operation without ``num_states``)
     - Raises: there is no "all of it" to measure. Pass ``num_seqs=``.

A count given explicitly is always honoured and never capped.

----

Reading the numbers
~~~~~~~~~~~~~~~~~~~

Everything except the pairwise distances is exact and costs one pass over the
sequences. Comparing every pair costs time quadratic in the number of
sequences, so above ``max_hamming_seqs`` a random subsample is compared instead
and ``hamming_exact`` is ``False``. A subsample estimates the **mean** very
well, but it sees only a small fraction of the pairs, so the reported
**minimum is an upper bound** on the true minimum and the maximum a lower
bound. The exact answer to "are any two sequences identical?" is
``num_duplicate_seqs``, which is always exact.

Sequences are measured as they would be exported: region tags stripped and
characters uppercased. Gap characters left by a deletion operation count as
characters.

Statistics that were not computed are absent from the result rather than
present as ``None`` -- there are no restriction-site keys unless ``enzymes`` or
``sites`` was given, and no distance keys when the sequences differ in length.

``stats`` is available on ``DnaPool``. Several of its statistics are
DNA-specific, so ``ProteinPool`` is not supported.

.. seealso::

   Every number in the report has a matching operation that acts on it:
   :doc:`operations/filter` for GC content, homopolymer runs, repetitiveness
   and restriction sites.

----

Visualising the DAG — ``print_dag(...)``
-----------------------------------------

Print an ASCII tree of the computation graph rooted at this pool. Returns
``self`` so it can be used mid-pipeline.

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``style``
     - ``str``
     - ``"clean"``
     - Tree drawing style. ``"clean"`` uses Unicode box-drawing characters;
       ``"ascii"`` uses only ASCII.
   * - ``show_pools``
     - ``bool``
     - ``True``
     - Show pool nodes in addition to operation nodes.

.. code-block:: python

    wt       = pp.from_seq("ACG")
    mut      = wt.mutagenize(num_mutations=1, mode="sequential")
    repeated = mut * 2
    repeated.print_dag()

.. code-block:: text

    pool[2] (pool, n=18)
    └── op[2]:repeat [mode=sequential, n=2]
        └── pool[1] (pool, n=9)
            └── op[1]:mutagenize [mode=sequential, n=9]
                └── pool[0] (pool, n=1)
                    └── op[0]:from_seq [mode=fixed, n=1]

----

Advanced
--------

Operator shortcuts
~~~~~~~~~~~~~~~~~~

Pools support three Python operators as shorthand for common operations:

``pool_a + pool_b``
    Equivalent to ``pp.stack([pool_a, pool_b])``. See :doc:`operations/stack`.

``pool * N``
    Equivalent to ``pp.repeat(pool, times=N)``. See :doc:`operations/repeat`.

``pool[start:stop]``
    Equivalent to ``pp.slice_states(pool, start=start, stop=stop)``. See
    :doc:`operations/slice_states`.

.. code-block:: python

    a = pp.from_seqs(["AAA", "CCC"], mode="sequential")
    b = pp.from_seqs(["GGG", "TTT"], mode="sequential")

    combined = a + b          # 4 states (2 + 2)
    repeated = a * 3          # 6 states (2 × 3)
    sliced   = combined[:3]   # 3 states (first 3 of 4)

``copy()`` and ``deepcopy()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``copy()`` creates a new pool that shares the same input pools -- useful for
branching a design at a specific point without re-running earlier operations.

``deepcopy()`` creates a fully independent copy of the entire upstream DAG
-- nothing is shared with the original. In most cases ``copy()`` is sufficient.
Use ``deepcopy()`` when the two branches must be fully independent and share
no input pools.

.. code-block:: python

    base = pp.from_iupac("NNNN", mode="sequential")
    branch_a = base.mutagenize(num_mutations=1).named("branch_a")
    branch_b = base.copy().mutagenize(num_mutations=2).named("branch_b")
    # branch_a and branch_b share the same "base" input pool
