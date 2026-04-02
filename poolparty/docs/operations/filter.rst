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

Examples
--------

Filter 6-mers by GC content
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep only 6-mers whose GC count is at least 3 (GC content &ge; 50 %).

.. code-block:: python

    pool    = pp.get_kmers(6)
    high_gc = pp.filter(pool, lambda s: s.count("G") + s.count("C") >= 3)
    df      = pp.generate_library(high_gc, num_seqs=6, discard_null_seqs=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (passing sequences only &mdash; GC &ge; 3 out of 6 bases)</em>
    AACGCG<br>
    ACCCGT<br>
    AGCCAT<br>
    CACCGT<br>
    CGCTAT<br>
    GCGAAT<br>
    <span class="pp-ellipsis">... (many more sequences pass; failing sequences are excluded)</span>
    </div>

Filter by sequence length
~~~~~~~~~~~~~~~~~~~~~~~~~~

When a pool may contain sequences of varying length, keep only those that are
exactly 8 bases long.

.. code-block:: python

    seqs    = pp.from_seqs(["ATCG", "ATCGATCG", "GGCC", "TTTTAAAA", "ACG"])
    trimmed = pp.filter(seqs, lambda s: len(s) == 8)
    df      = pp.generate_library(trimmed, discard_null_seqs=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (length == 8 only &mdash; shorter sequences are excluded)</em>
    ATCGATCG<br>
    TTTTAAAA<br>
    <span class="pp-ellipsis">... (sequences of other lengths are silently dropped)</span>
    </div>

Exclude sequences containing a restriction site
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove any sequence that contains the EcoRI recognition site ``GAATTC``.

.. code-block:: python

    pool     = pp.get_kmers(12)
    no_ecori = pp.filter(pool, lambda s: "GAATTC" not in s)
    df       = pp.generate_library(no_ecori, num_seqs=6, discard_null_seqs=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (no EcoRI site &mdash; sequences containing GAATTC are excluded)</em>
    AAAACGTTCCGT<br>
    AACGTTAGCCTA<br>
    ACGTATCGCCAT<br>
    CGATCGATCGAT<br>
    GCTAGCTAGCTA<br>
    TTACGCTAGCCA<br>
    <span class="pp-ellipsis">... (any 12-mer containing GAATTC is silently dropped)</span>
    </div>

Chain: mutagenize then filter for single-mutant sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build all single-point mutants of a wild-type sequence, then keep only those
that differ from the wild type at exactly one position.

.. code-block:: python

    wt       = pp.from_seq("ATCGATCG")
    mutants  = pp.mutagenize(wt, num_mutations=1)
    singles  = pp.filter(
        mutants,
        lambda s: sum(a != b for a, b in zip(s, "ATCGATCG")) == 1,
    )
    df       = pp.generate_library(singles, num_seqs=5, discard_null_seqs=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; sequences differing from wild type at exactly 1 position)</em>
    <span class="pp-mut">G</span>TCGATCG<br>
    A<span class="pp-mut">G</span>CGATCG<br>
    AT<span class="pp-mut">A</span>GATCG<br>
    ATCG<span class="pp-mut">C</span>TCG<br>
    ATCGAT<span class="pp-mut">T</span>G<br>
    <span class="pp-ellipsis">... (each draw is a unique single-substitution variant of ATCGATCG)</span>
    </div>

See :func:`~poolparty.filter`.
