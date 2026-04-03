Quickstart Guide
================

This guide introduces the core concepts of PoolParty through practical examples.

Installation
------------

.. code-block:: bash

    pip install poolparty

Basic Concepts
--------------

PoolParty uses **Pools** to represent collections of DNA sequences. Pools are:

- **Lazy**: Sequences are generated on-demand, not stored in memory
- **Composable**: Pools can be combined using operations like ``join``, ``+``, and ``*``
- **Stateful**: Each pool tracks its position in a combinatorial space via StateTracker

Getting Started
---------------

First, import PoolParty and initialize a session:

.. code-block:: python

    import poolparty as pp

    # Initialize PoolParty (creates a default Party context)
    pp.init()

Creating Pools
--------------

From a Single Sequence
~~~~~~~~~~~~~~~~~~~~~~

Create a pool containing a single sequence:

.. code-block:: python

    # Create a pool from a single sequence
    wt = pp.from_seq("ATCGATCGATCG")

    # Generate and display
    df = wt.generate_library()
    print(df[["seq", "seq_name"]])

From Multiple Sequences
~~~~~~~~~~~~~~~~~~~~~~~

Create a pool that selects from multiple sequences:

.. code-block:: python

    # Create a pool from multiple sequences
    variants = pp.from_seqs(["AAAA", "CCCC", "GGGG", "TTTT"])

    df = variants.generate_library()
    print(df[["seq", "seq_name"]])

K-mer Pools
~~~~~~~~~~~

Generate all k-mers of a given length:

.. code-block:: python

    # Generate all 3-mers (64 sequences)
    kmers = pp.get_kmers(length=3, alphabet="ACGT")

    df = kmers.generate_library()
    print(f"Generated {len(df)} sequences")
    print(df[["seq"]].head(10))

Combining Pools
---------------

Concatenation with ``join``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Join pools to create composite sequences:

.. code-block:: python

    # Create components
    pp.init()  # Reset to fresh state

    promoter = pp.from_seq("ATCG")
    barcode = pp.get_kmers(length=4, alphabet="ACGT")

    # Join them together
    library = pp.join([promoter, barcode])

    df = library.generate_library()
    print(f"Generated {len(df)} sequences")
    print(df[["seq"]].head(5))

Using the ``+`` Operator
~~~~~~~~~~~~~~~~~~~~~~~~

Pools can also be concatenated with ``+``:

.. code-block:: python

    pp.init()

    left = pp.from_seq("AAA")
    middle = pp.from_seqs(["G", "C"])
    right = pp.from_seq("TTT")

    combined = left + middle + right

    df = combined.generate_library()
    print(df[["seq"]])

Mutagenesis
-----------

Random Mutations
~~~~~~~~~~~~~~~~

Apply random mutations to a sequence:

.. code-block:: python

    pp.init()

    # Start with a wild-type sequence
    wt = pp.from_seq("ATCGATCGATCG")

    # Create single-mutation variants
    mutants = wt.mutagenize(alphabet="ACGT", k=1)

    df = mutants.generate_library()
    print(f"Generated {len(df)} single mutants")
    print(df[["seq", "seq_name"]].head(10))

Scan Operations
---------------

Scan operations tile across sequence positions.

Replacement Scan
~~~~~~~~~~~~~~~~

Replace each position with alternative bases:

.. code-block:: python

    pp.init()

    wt = pp.from_seq("ATCG")

    # Replace each position with all 4 bases
    scan = wt.replacement_scan(
        replacement_seqs=["A", "C", "G", "T"],
        length=1
    )

    df = scan.generate_library()
    print(df[["seq", "seq_name"]])

Deletion Scan
~~~~~~~~~~~~~

Systematically delete portions of a sequence:

.. code-block:: python

    pp.init()

    wt = pp.from_seq("ATCGATCG")

    # Delete 2-nt windows across the sequence
    deletions = wt.deletion_scan(length=2, step=1)

    df = deletions.generate_library()
    print(df[["seq", "seq_name"]])

Working with Regions
--------------------

PoolParty supports XML-like region tagging for targeting specific parts of sequences.

Tagging Regions
~~~~~~~~~~~~~~~

.. code-block:: python

    pp.init()

    # Define a sequence with a tagged region
    seq = "AAAA<cre>ATCGATCG</cre>TTTT"
    wt = pp.from_seq(seq)

    # Apply mutations only to the CRE region
    mutants = wt.mutagenize(alphabet="ACGT", k=1, region="cre")

    df = mutants.generate_library()
    print(f"Generated {len(df)} CRE mutants")
    print(df[["seq"]].head(5))

Generating Libraries
--------------------

The ``generate_library()`` method produces a pandas DataFrame with sequence information:

.. code-block:: python

    pp.init()

    # Create a simple library
    promoter = pp.from_seq("ATCG", region="promoter")
    barcode = pp.get_kmers(length=3, alphabet="AC", region="barcode")
    library = pp.join([promoter, barcode])

    # Generate with full metadata
    df = library.generate_library()

    print("Columns available:")
    print(df.columns.tolist())
    print()
    print(df[["seq", "seq_name"]].head())

Initialisation and Context Management
--------------------------------------

``pp.init()`` — persistent context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pp.init()`` creates a long-lived Party context that stays active for the
rest of the session. This is the recommended approach for notebooks and
interactive scripts.

.. code-block:: python

    import poolparty as pp
    pp.init()

    pool = pp.from_seq("ACGT")
    df   = pool.generate_library()

.. note::

   If you need a clean slate — for example, at the top of a new notebook cell
   block or after an experiment — call ``pp.init()`` again. This tears down the
   previous context and starts fresh: **all prior pools and operations are
   discarded**.

``pp.init()`` accepts:

- ``genetic_code`` (``str | dict``, default ``"standard"``) — genetic code for
  ORF operations.
- ``log_level`` (``str | None``, default ``None``) — if set, configures logging
  (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, etc.).

``with pp.Party()`` — scoped context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For isolation — running independent experiments, writing reusable functions,
or testing — use ``with pp.Party()``. The context is cleaned up automatically
when the block exits.

.. code-block:: python

    with pp.Party() as party:
        pool = pp.from_seq("ACGT")
        df   = pool.generate_library()
    # context closed

Contexts nest automatically:

.. code-block:: python

    with pp.Party() as outer:
        wt = pp.from_seq("ACGT")
        with pp.Party() as inner:
            other = pp.from_seq("TTTT")  # inner is active
        # outer is active again; wt is still usable

.. list-table::
   :widths: 50 25 25
   :header-rows: 1

   * - Scenario
     - ``pp.init()``
     - ``with pp.Party()``
   * - Interactive notebook or REPL
     - Recommended
     -
   * - Multiple independent experiments
     -
     - Recommended
   * - Inside a reusable function
     -
     - Recommended
   * - Quick reset (discard all pools)
     - Call again
     - Start a new ``with`` block

Configuration
~~~~~~~~~~~~~

These functions apply to whichever Party is currently active:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - ``pp.clear_pools()``
     - Discard all pools and operations without resetting configuration.
   * - ``pp.toggle_styles(on=True)``
     - Enable or disable inline sequence styling.
   * - ``pp.toggle_cards(on=True)``
     - Enable or disable design card computation.
   * - ``pp.set_text_progress(on=True)``
     - Use text-based progress bars instead of notebook widgets.
   * - ``pp.configure_logging(level)``
     - Set the logging level for ``poolparty`` and ``statetracker``.
   * - ``pp.set_genetic_code(genetic_code)``
     - Change the genetic code (affects ORF operations).

.. code-block:: python

    # Disable cards and styles for a performance-sensitive run
    pp.init()
    pp.toggle_cards(on=False)
    pp.toggle_styles(on=False)

    pool = pp.from_iupac("NNNNNNNN", mode="sequential")
    df   = pool.to_df(num_cycles=1)  # no card columns, no style overhead

Next Steps
----------

- Browse the :doc:`operations/index` for the full list of composable operations
- See :doc:`pool` for Pool properties and export methods (``to_df``, ``to_file``)
- Check out `StateTracker <https://statetracker.readthedocs.io>`_ for understanding the underlying state algebra
