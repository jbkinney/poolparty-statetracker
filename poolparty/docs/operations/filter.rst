filter
======

Retain only the sequences for which a predicate function returns ``True``; all
other sequences are replaced with a ``NullSeq`` sentinel.  Upstream pools are
often built with ``mode="sequential"`` (deterministic enumeration) or
``mode="random"`` (stochastic draws), depending on whether you need a fixed
walk through states or sampled variants.

.. code-block:: python

    import poolparty as pp
    pp.init()

.. note::

   Rejected sequences are **not removed** from the state space — they
   become ``NullSeq`` values that propagate silently through every
   downstream operation.  By default ``generate_library`` still includes
   ``NullSeq`` rows (as empty values).  Pass ``discard_null_seqs=True`` to
   exclude them from the output.

   The predicate receives the **tag-free** sequence string (region tags are
   stripped before evaluation).

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
     - ``Pool | DnaPool | ProteinPool``
     - *(required)*
     - Input pool to filter.
   * - ``predicate``
     - ``Callable[[str], bool]``
     - *(required)*
     - Function taking the clean (tag-free) sequence string; return
       ``True`` to keep the sequence.
   * - ``name``
     - ``str | None``
     - ``None``
     - Optional name for the filter operation.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for sequence names in the resulting pool.
   * - ``cards``
     - ``list[str] | dict[str, str] | None``
     - ``None``
     - Design card keys to include.  Available keys: ``'passed'``.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.filter` in the
   :doc:`API Reference </api>`.

Examples
--------

Filter 6-mers by GC content
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep only 6-mers whose GC count is at least 3 (GC content &ge; 50 %).

.. code-block:: python

    pool    = pp.get_kmers(6, mode="sequential")
    high_gc = pp.filter(pool, lambda s: s.count("G") + s.count("C") >= 3)
    high_gc.print_library()
    df      = pp.generate_library(high_gc, num_seqs=6, discard_null_seqs=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">high_gc: seq_length=6, num_states=4096</em>
    None<br>
    None<br>
    None<br>
    None<br>
    None
    <span class="pp-ellipsis">... (4096 total)</span>
    </div>

Filter by sequence length
~~~~~~~~~~~~~~~~~~~~~~~~~~

When a pool may contain sequences of varying length, keep only those that are
exactly 8 bases long.

.. code-block:: python

    seqs    = pp.from_seqs(
        ["ATCG", "ATCGATCG", "GGCC", "TTTTAAAA", "ACG"],
        mode="sequential",
    )
    trimmed = pp.filter(seqs, lambda s: len(s) == 8)
    trimmed.print_library()
    df      = pp.generate_library(trimmed, discard_null_seqs=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">trimmed: seq_length=None, num_states=5</em>
    None<br>
    ATCGATCG<br>
    None<br>
    TTTTAAAA<br>
    None
    </div>

Exclude sequences containing a restriction site
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove any 8-mer that contains the EcoRI recognition site ``GAATTC``.  (Here
``get_kmers`` uses ``mode="sequential"`` with length 8; length 12 is too large
for sequential enumeration under the default state limit.)

.. code-block:: python

    pool     = pp.get_kmers(8, mode="sequential")
    no_ecori = pp.filter(pool, lambda s: "GAATTC" not in s)
    no_ecori.print_library()
    df       = pp.generate_library(no_ecori, num_seqs=6, discard_null_seqs=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">no_ecori: seq_length=8, num_states=65536</em>
    AAAAAAAC<br>
    AAAAAAAG<br>
    AAAAAAAT<br>
    AAAAAACA<br>
    AAAAAACC
    <span class="pp-ellipsis">... (65536 total)</span>
    </div>

Chain: mutagenize then filter for single-mutant sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build single-point mutants of a wild-type sequence, then keep only those that
differ from the wild type at exactly one position.

.. code-block:: python

    wt       = pp.from_seq("ATCGATCG")
    mutants  = pp.mutagenize(wt, num_mutations=1, mode="random")
    singles  = pp.filter(
        mutants,
        lambda s: sum(a != b for a, b in zip(s, "ATCGATCG")) == 1,
    )
    singles.print_library()
    df       = pp.generate_library(singles, num_seqs=5, discard_null_seqs=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">singles: seq_length=8, num_states=1</em>
    ATCGGTCG
    </div>

See :func:`~poolparty.filter`.
