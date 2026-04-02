clear_annotation
================

Strip all XML region tags and non-molecular characters from sequences, then
uppercase the result. This produces clean, undecorated sequences suitable for
export (e.g., writing to FASTA) or for operations that do not accept tagged
input. When a ``region=`` is specified, only the content inside that tagged
segment is cleaned; the outer tags themselves are removed from the output
according to the ``remove_tags`` setting.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Examples
--------

Strip tags from a region-tagged sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove the ``cre`` open/close tags and uppercase every base, producing a
plain sequence with no markup.

.. code-block:: python

    wt    = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    plain = pp.clear_annotation(wt)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; region tags stripped, sequence uppercased)</em>
    AAAATCGTTTT
    </div>

Strip tags from the result of a region_scan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``region_scan`` output carries nested region tags around each mutated
position. Pipe through ``clear_annotation`` to yield bare sequences ready
for counting or export.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    alt  = pp.from_seqs(["A", "C", "G", "T"])
    scan = pp.region_scan(wt, "cre", ins_pool=alt)
    bare = pp.clear_annotation(scan)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (16 sequences &mdash; all tags removed; plain uppercase sequences)</em>
    AAAAATCGTTTT<br>
    AAAACTCGTTTT<br>
    AAAAGTCGTTTT<br>
    AAAATCGTTTT<br>
    <span class="pp-ellipsis">... (16 total, one per replacement)</span>
    </div>

Clear annotation before saving to FASTA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chain ``clear_annotation`` before ``generate_library`` to ensure the
sequences written to a FASTA file contain no markup characters.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    mutants = pp.mutagenize(wt, num_mutations=1, region="cre")
    clean   = pp.clear_annotation(mutants)
    df      = clean.generate_library(num_seqs=50, seed=0)
    # df["seq"] now contains plain uppercase sequences with no tags

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; mutagenized sequences with all tags cleared)</em>
    AAAAGCGATCGTTTT<br>
    AAAATCGTTTTTTTT<br>
    AAAATCGATTGTTTT<br>
    <span class="pp-ellipsis">... (50 total; each sequence is plain uppercase, no markup)</span>
    </div>

See :func:`~poolparty.clear_annotation`.
