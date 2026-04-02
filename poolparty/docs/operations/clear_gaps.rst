clear_gaps
==========

Remove all gap and non-molecular characters (``-``, ``.``, spaces, and any
other characters outside the DNA alphabet) from sequences. XML region tags are
preserved intact; only characters between tags are filtered. Because the output
length varies with the number of gaps removed, the resulting pool does not
carry a fixed ``seq_length``.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Examples
--------

Remove gap markers from a deletion_scan result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``deletion_scan`` replaces deleted bases with ``-`` markers. Pipe the result
through ``clear_gaps`` to produce gapless sequences of varying length.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    dels = pp.deletion_scan(wt, deletion_length=2)
    clean = pp.clear_gaps(dels)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (7 sequences &mdash; gap markers removed; each sequence is 6 nt)</em>
    CGATCG<br>
    AGATCG<br>
    ATATCG<br>
    ATCTCG<br>
    ATCGCG<br>
    ATCGAG<br>
    ATCGAT
    </div>

Clear gaps from a manually gapped sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strip dash characters from a sequence that was constructed with explicit
alignment gaps.

.. code-block:: python

    wt    = pp.from_seq("AT--CG--AT")
    clean = pp.clear_gaps(wt)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; four gap characters removed)</em>
    ATCGAT
    </div>

Chain clear_gaps with another operation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove gaps first, then apply ``rc`` to produce gapless reverse-complement
sequences ready for downstream analysis.

.. code-block:: python

    wt    = pp.from_seq("AT--CG")
    clean = pp.clear_gaps(wt)
    rev   = pp.rc(clean)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; gaps removed, then reverse-complemented)</em>
    CGAT
    </div>

See :func:`~poolparty.clear_gaps`.
