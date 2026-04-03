Pools
===============

Every PoolParty operation returns a :class:`~poolparty.Pool`. This page covers
the attributes available on every pool and the methods for generating and
exporting sequences.

.. seealso::

   Pools must be created inside an active context. Call ``pp.init()`` once at
   the top of a notebook, or use ``with pp.Party():`` for scoped isolation.
   See :doc:`quickstart` for setup, context management, and configuration.

All examples assume::

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
     - Number of distinct states (draws) this pool produces.
   * - ``seq_length``
     - ``int | None``
     - Fixed sequence length, or ``None`` for variable-length pools.
   * - ``iter_order``
     - ``float``
     - Iteration priority. Pools with higher values iterate fastest in a
       joined or stacked pool.
   * - ``regions``
     - ``set[Region]``
     - Set of :class:`~poolparty.Region` objects present in this pool's sequences.
   * - ``parents``
     - ``list[Pool]``
     - Upstream pools that feed into this pool's operation.

----

Naming and Copying
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

``copy()`` creates a parallel branch in the DAG that shares the same
upstream pools — useful for branching a design at a specific point without
re-running upstream operations.

``deepcopy()`` creates a fully independent copy of the entire upstream DAG
— nothing is shared with the original.

.. code-block:: python

    base = pp.from_iupac("NNNN", mode="sequential")
    branch_a = base.mutagenize(num_mutations=1).named("branch_a")
    branch_b = base.copy().mutagenize(num_mutations=2).named("branch_b")
    # branch_a and branch_b share the same "base" upstream pool

----

Generating Sequences
--------------------

``generate_library(...)``
~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate the pool DAG and return a :class:`pandas.DataFrame`. This is the
primary way to draw sequences. See :doc:`operations/generate_library` for full
documentation.

.. code-block:: python

    pool = pp.from_iupac("NNNN", mode="sequential")
    df   = pool.generate_library()
    # df has columns: name, seq  (plus any design card columns)

``print_library(...)``
~~~~~~~~~~~~~~~~~~~~~~

Print a formatted preview of the pool's sequences to stdout and return
``self`` for chaining.

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
     - Complete cycles through the state space (used when ``num_seqs`` is
       not given).
   * - ``show_header``
     - ``bool``
     - ``True``
     - Print a summary header line before the sequences.
   * - ``show_name``
     - ``bool``
     - ``True``
     - Include the sequence name column.
   * - ``pad_names``
     - ``bool``
     - ``True``
     - Align sequences by padding names to the same width.
   * - ``seed``
     - ``int | None``
     - ``None``
     - Random seed for reproducible previews.

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
libraries (> ~10 k sequences) because it processes sequences in batches and
avoids building the full DataFrame in a single allocation.

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
     - Complete cycles through the state space.
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
     - Skip sequences that were filtered out (``NullSeq``).
   * - ``columns``
     - ``list[str] | None``
     - ``None``
     - Columns to keep. Defaults to all columns (``name``, ``seq``, plus
       any design card columns). Pass ``["name", "seq"]`` to drop cards.
   * - ``show_progress``
     - ``bool``
     - ``True``
     - Display a ``tqdm`` progress bar during generation.

Basic usage
~~~~~~~~~~~

.. code-block:: python

    pool = pp.from_iupac("NNNNNNNN", mode="sequential")
    df   = pool.to_df(num_cycles=1)
    # 65536 rows, columns: name, seq

Large library with chunked streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    pool = pp.from_iupac("NNNNNNNNNN")
    df   = pool.to_df(num_seqs=500_000, chunk_size=10_000, seed=42) # Random sample of 500k sequences from ~1M possibe sequences

Keep only ``name`` and ``seq`` (drop design cards)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    scored = pp.score(pool, pp.calc_gc, card_key="gc", cards={"gc": "gc"})
    df     = scored.to_df(num_cycles=1, columns=["name", "seq"])
    # "gc" column is excluded

----

Exporting to File — ``to_file(...)``
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
     - Complete cycles through the state space.
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
     - Skip filtered-out (``NullSeq``) sequences.
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

Returns the number of sequences written.

Export to CSV
~~~~~~~~~~~~~

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

Export to gzip-compressed CSV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    n = pool.to_file("library.csv.gz", num_seqs=1_000_000, chunk_size=50_000)

Export to FASTA
~~~~~~~~~~~~~~~

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

FASTA with a custom description line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    scored = pp.score(pool, pp.calc_gc, card_key="gc", cards={"gc": "gc"})
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

Print an ASCII tree of the computation graph rooted at this pool and return
``self`` for chaining.

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

    wt  = pp.from_seq("ACGTACGT")
    mut = pp.mutagenize(wt, num_mutations=2)
    scored = pp.score(mut, pp.calc_gc, card_key="gc", cards={"gc": "gc"})
    scored.print_dag()

.. code-block:: text

    scored
    └── mutagenize
        └── from_seq
