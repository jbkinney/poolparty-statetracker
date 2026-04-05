Sequence Names
==============

Every sequence produced by ``generate_library()`` has a **name** — a
dot-separated string built from segments contributed by each operation.
Names let you trace exactly how a sequence was constructed.

All examples assume:

.. code-block:: python

    import poolparty as pp
    pp.init()

----

How names are built
-------------------

Each operation can contribute a name segment via its ``prefix`` parameter.
Segments are collected from source to downstream and joined with dots:

.. code-block:: text

    name = "prefix_A.prefix_B.prefix_C"

If an operation has ``prefix=None`` (the default), it contributes nothing
to the name. If no operation in the pipeline sets a prefix, the ``name``
column is ``None``.

----

The ``prefix`` parameter
------------------------

Most operations accept a ``prefix`` parameter. How the prefix is formatted
depends on the operation's mode:

**Fixed mode** (single deterministic output):
    Contributes the prefix string as-is.

    .. code-block:: python

        pool = pp.from_seq("ACGT", prefix="wt")
        df   = pool.generate_library()

    .. raw:: html

        <div class="pp-pool">
        <em class="pp-header">df — 1 row × 2 columns</em>
        <table class="pp-df">
        <tr><th>name</th><th>seq</th></tr>
        <tr><td>wt</td><td>ACGT</td></tr>
        </table>
        </div>

**Sequential mode** (one state per variant):
    Appends a zero-padded state index. The width adjusts to the number of
    states so names sort correctly.

    .. code-block:: python

        pool = pp.from_seqs(["AAAA", "CCCC", "GGGG"], mode="sequential", prefix="var")
        df   = pool.generate_library()

    .. raw:: html

        <div class="pp-pool">
        <em class="pp-header">df — 3 rows × 2 columns</em>
        <table class="pp-df">
        <tr><th>name</th><th>seq</th></tr>
        <tr><td>var_0</td><td>AAAA</td></tr>
        <tr><td>var_1</td><td>CCCC</td></tr>
        <tr><td>var_2</td><td>GGGG</td></tr>
        </table>
        </div>

    With more states the padding grows:

    .. code-block:: python

        pool = pp.from_iupac("NNNN", mode="sequential", prefix="seq")
        df   = pool.generate_library()
        # names: "seq_000", "seq_001", ..., "seq_255"

**Random mode** (random draws):
    Appends a zero-padded global draw index, based on how many sequences
    were requested.

    .. code-block:: python

        wt   = pp.from_seq("ATCGATCG")
        pool = wt.mutagenize(num_mutations=1, prefix="mut")
        df   = pool.generate_library(num_seqs=50)
        # names: "mut_00", "mut_01", ..., "mut_49"

----

Chaining operations
-------------------

When multiple operations in a pipeline set ``prefix``, each contributes a
segment and they are joined with dots:

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG", prefix="bg")
    muts = wt.mutagenize(num_mutations=1, num_states=3, prefix="mut")
    df   = muts.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 3 rows × 2 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th></tr>
    <tr><td>bg.mut_0</td><td>CTCGATCG</td></tr>
    <tr><td>bg.mut_1</td><td>GTCGATCG</td></tr>
    <tr><td>bg.mut_2</td><td>TTCGATCG</td></tr>
    </table>
    </div>

Add more segments with ``add_prefix``:

.. code-block:: python

    tagged = muts.add_prefix("final")
    df     = tagged.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 3 rows × 2 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th></tr>
    <tr><td>bg.mut_0.final</td><td>CTCGATCG</td></tr>
    <tr><td>bg.mut_1.final</td><td>GTCGATCG</td></tr>
    <tr><td>bg.mut_2.final</td><td>TTCGATCG</td></tr>
    </table>
    </div>

----

Custom sequence names with ``from_seqs``
----------------------------------------

``from_seqs`` accepts a ``seq_names`` parameter for explicit names that
override the prefix logic:

.. code-block:: python

    pool = pp.from_seqs(
        ["ATCG", "ATAG", "AACG"],
        seq_names=["wt", "mut_A", "mut_B"],
        mode="sequential",
    )
    df = pool.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 3 rows × 2 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th></tr>
    <tr><td>wt</td><td>ATCG</td></tr>
    <tr><td>mut_A</td><td>ATAG</td></tr>
    <tr><td>mut_B</td><td>AACG</td></tr>
    </table>
    </div>

----

Scan operation names
--------------------

Scan operations can contribute **compound names** with separate segments
for the position index and the variant index. These are controlled by
additional prefix parameters:

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    alt  = pp.from_seqs(["A", "C", "G", "T"], mode="sequential", prefix="base")
    scan = wt.replacement_scan(replacement_pool=alt, mode="sequential",
                               prefix="scan", prefix_position="pos",
                               prefix_insert="ins")
    df   = scan.generate_library(num_seqs=8)
    # names: "scan_00.pos_0.base_0", "scan_01.pos_0.base_1", ...

----

``Pool.named()`` vs ``prefix``
-------------------------------

These are different things:

- ``pool.named("my_pool")`` sets the **pool's metadata name** — used for
  display, DAG visualization, and internal tracking. It does **not**
  affect the ``name`` column in the output.

- ``prefix="label"`` on an operation affects the **sequence names** in the
  generated DataFrame.

.. code-block:: python

    pool = pp.from_seq("ACGT", prefix="bg").named("my_pool")
    print(pool.name)   # "my_pool" (pool metadata)
    df = pool.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 1 row × 2 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th></tr>
    <tr><td>bg</td><td>ACGT</td></tr>
    </table>
    </div>

The pool is called ``"my_pool"`` (used in DAG display), but the sequence's
name in the output is ``"bg"`` (from the ``prefix`` parameter).
