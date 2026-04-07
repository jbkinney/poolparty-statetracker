filter
======

Retain only the sequences for which a predicate function returns ``True``; all
other sequences are replaced with a ``NullSeq`` sentinel.

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

Filter by GC content
~~~~~~~~~~~~~~~~~~~~~

Keep only sequences whose GC count is at least 3 (GC content &ge; 50 %).
Sequences that fail the predicate become ``None`` (a ``NullSeq`` sentinel);
pass ``discard_null_seqs=True`` to ``generate_library`` to exclude them from
the final DataFrame.

.. code-block:: python

    seqs    = pp.from_seqs(
        ["AAAAAA", "GCGCGC", "AAACCC", "TTTTTT", "GGCCAA"],
        mode="sequential",
    )
    high_gc = pp.filter(seqs, lambda s: s.count("G") + s.count("C") >= 3)
    high_gc.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">high_gc: seq_length=6, num_states=5</em>
    None<br>
    GCGCGC<br>
    AAACCC<br>
    None<br>
    GGCCAA
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
    AAAAAAAA<br>
    AAAAAAAC<br>
    AAAAAAAG<br>
    AAAAAAAT<br>
    AAAAAACA
    <span class="pp-ellipsis">... (65536 total)</span>
    </div>

Chain: mutagenize then filter by GC content
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate all single-nucleotide mutants, then keep only those whose GC
content is at least 5 out of 8 bases.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    mutants = pp.mutagenize(wt, num_mutations=1, mode="sequential")
    high_gc = pp.filter(mutants, lambda s: s.count("G") + s.count("C") >= 5)
    high_gc.print_library(num_seqs=8)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">high_gc: seq_length=8, num_states=24</em>
    CTCGATCG<br>
    GTCGATCG<br>
    None<br>
    None<br>
    ACCGATCG<br>
    AGCGATCG<br>
    None<br>
    None
    <span class="pp-ellipsis">... (24 total)</span>
    </div>

See :func:`~poolparty.filter`.
