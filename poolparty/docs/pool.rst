Pools
=====

Every PoolParty operation returns a **Pool**. A Pool represents a designed
sequence library: it records which operation was applied and to what inputs,
forming a directed acyclic graph (DAG) of operations. PoolParty walks this
graph to generate sequences on demand when you call ``generate_library()``,
``print_library()``, ``to_df()``, or ``to_file()``.

Every operation returns a **new** Pool — the original is never modified. This
means you can branch a pipeline at any point and apply different operations to
each branch without interference.

Each pool carries a reference to the operation that created it. You can inspect
it via ``pool.operation`` to check settings like ``operation.mode`` and
``operation.num_states`` at any point in a pipeline. See :doc:`operations/modes`
for details.

Pools must be created inside an active context. Call ``pp.init()`` once at the
top of a notebook, or use ``with pp.Party():`` for automatic cleanup when the
block exits. See :doc:`quickstart` for details.

All examples assume:

.. code-block:: python

    import poolparty as pp
    pp.init()

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
     - Number of distinct sequences this pool produces.
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

Naming and copying
------------------

``named(name)``
~~~~~~~~~~~~~~~

Set the pool's name and return ``self``, allowing in-line renaming without
breaking a chain.

.. code-block:: python

    wt = pp.from_seq("ACGT").named("wildtype")
    # wt.name == "wildtype"

    scored = (
        pp.from_iupac("NNNN", mode="sequential")
          .mutagenize(num_mutations=1)
          .named("single_mut")
    )

``copy()`` and ``deepcopy()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``copy()`` creates a new pool that shares the same input pools — useful for
branching a design at a specific point without re-running earlier operations.

``deepcopy()`` creates a fully independent copy of the entire upstream DAG
— nothing is shared with the original. In most cases ``copy()`` is sufficient.
Use ``deepcopy()`` when the two branches must be fully independent and share
no input pools.

.. code-block:: python

    base = pp.from_iupac("NNNN", mode="sequential")
    branch_a = base.mutagenize(num_mutations=1).named("branch_a")
    branch_b = base.copy().mutagenize(num_mutations=2).named("branch_b")
    # branch_a and branch_b share the same "base" input pool

----

Operator shortcuts
------------------

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

----

Generating sequences
--------------------

``generate_library(...)``
~~~~~~~~~~~~~~~~~~~~~~~~~

Generate all sequences from this pool and return them as a
:class:`pandas.DataFrame`. Best for small to medium pools; for libraries above ~10k
sequences, use ``to_df`` which streams in chunks. See
:doc:`operations/generate_library` for full documentation.

.. code-block:: python

    pool = pp.from_iupac("NNNN", mode="sequential")
    df   = pool.generate_library()
    # df has columns: name, seq  (plus any design card columns)

``print_library(...)``
~~~~~~~~~~~~~~~~~~~~~~

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
   * - ``num_cycles``
     - ``int | None``
     - ``1``
     - Number of complete passes through the pool's ``num_states`` sequences
       (used when ``num_seqs`` is not given). One cycle produces
       ``num_states`` sequences.
   * - ``show_header``
     - ``bool``
     - ``True``
     - Print a summary header line before the sequences.
   * - ``show_name``
     - ``bool``
     - ``True``
     - Include the sequence name column.
   * - ``show_seq``
     - ``bool``
     - ``True``
     - Include the sequence column.
   * - ``show_state``
     - ``bool``
     - ``False``
     - Include the state index column.
   * - ``pad_names``
     - ``bool``
     - ``True``
     - Align sequences by padding names to the same width.
   * - ``seed``
     - ``int | None``
     - ``None``
     - Random seed for reproducible previews.
   * - ``discard_null_seqs``
     - ``bool``
     - ``False``
     - Skip sequences removed by a ``filter`` operation (``NullSeq``).

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
