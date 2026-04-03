Sequence Names
==============

Every sequence produced by ``generate_library()`` has a **name** — a
dot-separated string assembled from contributions made by each operation
in the DAG. Names let you trace exactly how a sequence was generated.

All examples assume::

    import poolparty as pp
    pp.init()

----

How Names Are Built
-------------------

Each operation can contribute a name segment via its ``prefix`` parameter.
Segments are collected in topological order (source operations first,
downstream last) and joined with dots:

.. code-block:: text

    name = "prefix_A.prefix_B.prefix_C"

If an operation has ``prefix=None`` (the default), it contributes nothing
to the name. If no operation in the pipeline sets a prefix, the ``name``
column is ``None``.

----

The ``prefix`` Parameter
------------------------

Most operations accept a ``prefix`` parameter. How the prefix is formatted
depends on the operation's mode:

**Fixed mode** (single deterministic output):
    Contributes the prefix string as-is.

    .. code-block:: python

        pool = pp.from_seq("ACGT", prefix="wt")
        df   = pool.generate_library()
        # name: "wt"

**Sequential mode** (one state per variant):
    Appends a zero-padded state index. The width adjusts to the number of
    states so names sort correctly.

    .. code-block:: python

        pool = pp.from_seqs(["AAAA", "CCCC", "GGGG"], mode="sequential", prefix="var")
        df   = pool.generate_library()
        # names: "var_0", "var_1", "var_2"

    With more states the padding grows:

    .. code-block:: python

        pool = pp.from_iupac("NNNN", mode="sequential", prefix="seq")
        df   = pool.generate_library()
        # names: "seq_000", "seq_001", ..., "seq_255"

**Random mode** (stochastic draws):
    Appends a zero-padded global draw index, based on how many sequences
    were requested.

    .. code-block:: python

        pool = pp.mutagenize(wt, num_mutations=1, prefix="mut")
        df   = pool.generate_library(num_seqs=50)
        # names: "mut_00", "mut_01", ..., "mut_49"

----

Chaining Operations
-------------------

When multiple operations in a pipeline set ``prefix``, each contributes a
segment and they are joined with dots:

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG", prefix="bg")
    muts = pp.mutagenize(wt, num_mutations=1, num_states=3, prefix="mut")
    df   = muts.generate_library()
    # names: "bg.mut_0", "bg.mut_1", "bg.mut_2"

Add more segments with ``pp.add_prefix``:

.. code-block:: python

    tagged = pp.add_prefix(muts, "final")
    df     = tagged.generate_library()
    # names: "bg.mut_0.final", "bg.mut_1.final", "bg.mut_2.final"

----

Custom Sequence Names with ``from_seqs``
-----------------------------------------

``from_seqs`` accepts a ``seq_names`` parameter for explicit names that
override the prefix logic:

.. code-block:: python

    pool = pp.from_seqs(
        ["ATCG", "ATAG", "AACG"],
        seq_names=["wt", "mut_A", "mut_B"],
        mode="sequential",
    )
    df = pool.generate_library()
    # names: "wt", "mut_A", "mut_B"

----

Scan Operation Names
--------------------

Scan operations can contribute **compound names** with separate segments
for the position index and the variant index. These are controlled by
additional prefix parameters:

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    alt  = pp.from_seqs(["A", "C", "G", "T"], mode="sequential", prefix="base")
    scan = pp.replacement_scan(wt, ins_pool=alt, mode="sequential",
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
    # pool.name == "my_pool"  (metadata)
    df = pool.generate_library()
    # df["name"] == "bg"      (sequence name from prefix)
