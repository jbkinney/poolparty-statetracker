insertion_scan
==============

Insert sequences from ``insertion_pool`` at every position along the background
sequence (or within a named region). Unlike :func:`~poolparty.replacement_scan`,
no background bases are removed, so output sequences are longer than the input.
Set ``replace=True`` to replace a window of ``ins_length`` bases at each
position rather than inserting without deletion; output length stays equal
to the background length. This is equivalent to
:func:`~poolparty.replacement_scan`.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pool``
     - ``Pool | str``
     - *(required)*
     - The background Pool to scan. Can also be a plain sequence string.
   * - ``insertion_pool``
     - ``Pool | str``
     - *(required)*
     - Pool or sequence string whose content is inserted at each scanned
       position. An *L*-mer has *L* + 1 valid insertion sites (before
       each base and after the last).
   * - ``positions``
     - ``list[int] | None``
     - ``None``
     - Explicit list of insertion positions. ``None`` = all valid positions.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Restrict insertions to a named region or ``[start, stop]`` interval.
       Flanking sequences are never modified.
   * - ``replace``
     - ``bool``
     - ``False``
     - When ``True``, a window of ``ins_length`` bases is replaced at each
       position (equivalent to :func:`~poolparty.replacement_scan`). Valid
       positions = background length − insert length + 1; output length =
       background length.
   * - ``style``
     - ``str | None``
     - ``None``
     - Named display style applied to inserted bases.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` iterates positions then inserts in order; ``'random'``
       shuffles the (position × insert) product.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Number of output states. ``None`` auto-computes in sequential mode
       or defaults to 1 in random mode.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Enumeration order when combined with other pools.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.insertion_scan` in the
   :doc:`API Reference </api>`.

Examples
--------

Single-base insertions at every position
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An 8-mer has 9 insertion sites. 9 sites × 4 bases = 36 sequences, each of
length 9.

.. code-block:: python

    wt    = pp.from_seq("ACGTACGT")
    bases = pp.from_seqs(["A", "C", "G", "T"], mode="sequential")
    scan  = wt.insertion_scan(insertion_pool=bases, mode="sequential", style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=9, num_states=36</em>
    <span class="pp-mut">A</span>ACGTACGT<br>
    A<span class="pp-mut">A</span>CGTACGT<br>
    AC<span class="pp-mut">A</span>GTACGT<br>
    ACG<span class="pp-mut">A</span>TACGT<br>
    ACGT<span class="pp-mut">A</span>ACGT
    <span class="pp-ellipsis">... (36 total)</span>
    </div>

All-dinucleotide insertions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``from_iupac("NN")`` to enumerate all 16 dinucleotide inserts.
9 sites × 16 inserts = 144 sequences, each of length 10.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    nn   = pp.from_iupac("NN", mode="sequential")
    scan = wt.insertion_scan(insertion_pool=nn, mode="sequential", style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=10, num_states=144</em>
    <span class="pp-mut">AA</span>ACGTACGT<br>
    A<span class="pp-mut">AA</span>CGTACGT<br>
    AC<span class="pp-mut">AA</span>GTACGT<br>
    ACG<span class="pp-mut">AA</span>TACGT<br>
    ACGT<span class="pp-mut">AA</span>ACGT
    <span class="pp-ellipsis">... (144 total)</span>
    </div>

Insert-and-replace mode (replace=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``replace=True`` replaces a window equal in width to the insert (here 2
bases) at each position. For an 8-mer with a 2-base insert: 8 − 2 + 1 = 7
valid positions; output length stays 8. This is equivalent to calling
:func:`~poolparty.replacement_scan`.

.. code-block:: python

    wt    = pp.from_seq("ACGTACGT")
    bases = pp.from_seqs(["AA", "CC", "GG", "TT"], mode="sequential")
    scan  = wt.insertion_scan(insertion_pool=bases, replace=True, mode="sequential",
                              style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=8, num_states=28</em>
    <span class="pp-mut">AA</span>GTACGT<br>
    A<span class="pp-mut">AA</span>TACGT<br>
    AC<span class="pp-mut">AA</span>ACGT<br>
    ACG<span class="pp-mut">AA</span>CGT<br>
    ACGT<span class="pp-mut">AA</span>GT
    <span class="pp-ellipsis">... (28 total)</span>
    </div>

Insertion scan within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict insertion sites to the ``cre`` region. The 8-base region has 9
valid insertion sites; flanks are never altered.

.. code-block:: python

    wt    = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    bases = pp.from_seqs(["A", "C", "G", "T"], mode="sequential")
    scan  = wt.insertion_scan(insertion_pool=bases, region="cre", mode="sequential",
                              style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=17, num_states=36</em>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span><span class="pp-mut">A</span>ATCGATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>A<span class="pp-mut">A</span>TCGATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>AT<span class="pp-mut">A</span>CGATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ATC<span class="pp-mut">A</span>GATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ATCG<span class="pp-mut">A</span>ATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT
    <span class="pp-ellipsis">... (36 total)</span>
    </div>

Explicit position list
~~~~~~~~~~~~~~~~~~~~~~~

Limit the scan to specific insertion sites.

.. code-block:: python

    wt    = pp.from_seq("ACGTACGT")
    bases = pp.from_seqs(["A", "C", "G", "T"], mode="sequential")
    scan  = wt.insertion_scan(insertion_pool=bases, positions=[0, 4, 8],
                              mode="sequential", style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=9, num_states=12</em>
    <span class="pp-mut">A</span>ACGTACGT<br>
    ACGT<span class="pp-mut">A</span>ACGT<br>
    ACGTACGT<span class="pp-mut">A</span><br>
    <span class="pp-mut">C</span>ACGTACGT<br>
    ACGT<span class="pp-mut">C</span>ACGT
    <span class="pp-ellipsis">... (12 total)</span>
    </div>

Random motif insertion (mode="random")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode='random'`` draws insertion positions stochastically. Here a degenerate
6-base IUPAC motif (``R`` = A|G, ``Y`` = C|T) is inserted at random
positions along a 12-mer.

.. code-block:: python

    wt    = pp.from_seq("ACGTACGTACGT")
    motif = pp.from_iupac("RRYYYY")
    scan  = wt.insertion_scan(insertion_pool=motif, mode="random", num_states=5,
                              style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=18, num_states=5</em>
    ACGTACGTACGT<span class="pp-mut">GACCCT</span><br>
    ACGTACGTACGT<span class="pp-mut">GATCTT</span><br>
    AC<span class="pp-mut">GACCTT</span>GTACGTACGT<br>
    ACG<span class="pp-mut">AACTTT</span>TACGTACGT<br>
    ACGTAC<span class="pp-mut">AATTCC</span>GTACGT
    </div>

See :func:`~poolparty.insertion_scan`.
