Case Operations (``upper``, ``lower``, ``swapcase``)
====================================================

Three deterministic operations change the case of sequence characters without
altering the underlying bases. XML region tags are preserved intact; only
molecular characters are transformed. All three accept an optional ``region=``
parameter to restrict the transformation to a named segment.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Examples
--------

upper: uppercase a full sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert every base of a lowercase sequence to uppercase.

.. code-block:: python

    wt = pp.from_seq("atcgatcg")
    up = pp.upper(wt)
    up.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">up: seq_length=8, num_states=1</em>
    ATCGATCG
    </div>

upper: uppercase a named region only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only the content of the ``cre`` region is uppercased; the flanking lowercase
bases remain unchanged.

.. code-block:: python

    wt = pp.from_seq("aaaa<cre>atcgatcg</cre>tttt")
    up = pp.upper(wt, region="cre")
    up.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">up: seq_length=8, num_states=1</em>
    aaaa<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>tttt
    </div>

lower: lowercase a full sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert every base of an uppercase sequence to lowercase.

.. code-block:: python

    wt = pp.from_seq("ATCGATCG")
    lo = pp.lower(wt)
    lo.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">lo: seq_length=8, num_states=1</em>
    atcgatcg
    </div>

lower: lowercase a named region only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only the content of the ``cre`` region is lowercased; the uppercase flanks
are returned unchanged.

.. code-block:: python

    wt = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    lo = pp.lower(wt, region="cre")
    lo.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">lo: seq_length=8, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>atcgatcg<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>

swapcase: swap a mixed-case sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Invert the case of every base: uppercase becomes lowercase and vice versa.

.. code-block:: python

    wt = pp.from_seq("ATCGatcg")
    sw = pp.swapcase(wt)
    sw.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">sw: seq_length=8, num_states=1</em>
    atcgATCG
    </div>

swapcase: visually distinguish a region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply ``lower`` to the full sequence, then ``swapcase`` restricted to the
``cre`` region to make that segment stand out in uppercase.

.. code-block:: python

    wt = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    lo = pp.lower(wt)
    hi = pp.swapcase(lo, region="cre")
    hi.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">hi: seq_length=8, num_states=1</em>
    aaaa<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>tttt
    </div>

See :func:`~poolparty.swapcase`.
