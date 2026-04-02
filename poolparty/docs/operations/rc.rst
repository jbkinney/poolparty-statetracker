rc
==

Return the reverse complement of a sequence or a named region. By default the
entire sequence is reverse-complemented and region tags are stripped; pass
``region=`` to restrict the operation to a single tagged segment.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Examples
--------

Reverse complement a simple sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Take the reverse complement of the 4-mer ``ATCG``; the result is ``CGAT``.

.. code-block:: python

    wt = pp.from_seq("ATCG")
    r  = pp.rc(wt)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; ATCG &rarr; reverse complement)</em>
    CGAT
    </div>

Reverse complement a longer tagged sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reverse-complement the full sequence ``AAAA<cre>ATCGATCG</cre>TTTT``
(tags are stripped before the operation and not carried into the output).

.. code-block:: python

    wt = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    r  = pp.rc(wt)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; full 16-mer reverse-complemented; tags removed)</em>
    AAAACGATCGATTTTT
    </div>

Reverse complement only a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``region=`` to reverse-complement only the content of the ``cre``
segment; flanking sequences are returned unchanged.

.. code-block:: python

    wt = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    r  = pp.rc(wt, region="cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; only the <em>cre</em> region reverse-complemented)</em>
    AAAA<span class="pp-region">CGATCGAT</span>TTTT
    </div>

See :func:`~poolparty.rc`.
