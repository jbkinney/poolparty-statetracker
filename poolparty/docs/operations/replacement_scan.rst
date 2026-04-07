replacement_scan
================

Slide a window across the sequence and, at each position, replace the window
with every sequence drawn from ``replacement_pool``. The output length equals the
background sequence length.

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
     - The background Pool or sequence string that is scanned.
   * - ``replacement_pool``
     - ``Pool | str``
     - *(required)*
     - Pool supplying replacement content. Each of this pool's sequences is
       substituted into the window at every scanned position. The pool's
       ``seq_length`` determines the window width.
   * - ``positions``
     - ``list[int] | None``
     - ``None``
     - Explicit list of window start positions. ``None`` = all valid positions.
   * - ``region``
     - ``str | None``
     - ``None``
     - Name of a tagged region to restrict the scan to. Positions outside
       the region are never modified.
   * - ``style``
     - ``str | None``
     - ``None``
     - Style to apply to inserted content.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` iterates positions then replacements in order;
       ``'random'`` shuffles the (position × replacement) product.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Number of output states. ``None`` auto-computes in sequential mode
       or defaults to 1 in random mode.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``prefix_position``
     - ``str | None``
     - ``None``
     - Prefix for position index (e.g., ``'pos_'`` produces ``'pos_0'``, ``'pos_1'``, ...).
   * - ``prefix_insert``
     - ``str | None``
     - ``None``
     - Prefix for insert index (e.g., ``'ins_'`` produces ``'ins_0'``, ``'ins_1'``, ...).
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Enumeration order when combined with other pools.


----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.replacement_scan` in the
   :doc:`API Reference </api>`.

Examples
--------

Single-base replacement at every position
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace each base with every member of a four-base pool — 8 positions × 4
substitutions = 32 sequences.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    alt  = pp.from_seqs(["A", "C", "G", "T"], mode="sequential")
    scan = wt.replacement_scan(replacement_pool=alt, mode="sequential", style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=8, num_states=32</em>
    <span class="pp-mut">A</span>CGTACGT<br>
    A<span class="pp-mut">A</span>GTACGT<br>
    AC<span class="pp-mut">A</span>TACGT<br>
    ACG<span class="pp-mut">A</span>ACGT<br>
    ACGT<span class="pp-mut">A</span>CGT
    <span class="pp-ellipsis">... (32 total)</span>
    </div>

Trinucleotide window scan
~~~~~~~~~~~~~~~~~~~~~~~~~~

The window width is determined by ``replacement_pool`` sequence length. A 3-base
pool scans 6 positions across an 8-mer.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    tri  = pp.from_seqs(["AAA", "CCC", "GGG", "TTT"], mode="sequential")
    scan = wt.replacement_scan(replacement_pool=tri, mode="sequential", style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=8, num_states=24</em>
    <span class="pp-mut">AAA</span>TACGT<br>
    A<span class="pp-mut">AAA</span>ACGT<br>
    AC<span class="pp-mut">AAA</span>CGT<br>
    ACG<span class="pp-mut">AAA</span>GT<br>
    ACGT<span class="pp-mut">AAA</span>T
    <span class="pp-ellipsis">... (24 total)</span>
    </div>

All-dinucleotide replacements via from_iupac
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``from_iupac("NN")`` enumerates all 16 dinucleotides. Seven windows × 16
substitutions = 112 sequences.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    nn   = pp.from_iupac("NN", mode="sequential")
    scan = wt.replacement_scan(replacement_pool=nn, mode="sequential", style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=8, num_states=112</em>
    <span class="pp-mut">AA</span>GTACGT<br>
    A<span class="pp-mut">AA</span>TACGT<br>
    AC<span class="pp-mut">AA</span>ACGT<br>
    ACG<span class="pp-mut">AA</span>CGT<br>
    ACGT<span class="pp-mut">AA</span>GT
    <span class="pp-ellipsis">... (112 total)</span>
    </div>

Scan restricted to a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict the scan to the ``cre`` region; the ``AAAA`` and ``TTTT`` flanks
are never modified.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    alt  = pp.from_seqs(["A", "C", "G", "T"], mode="sequential")
    scan = wt.replacement_scan(replacement_pool=alt, region="cre", mode="sequential",
                               style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=16, num_states=32</em>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span><span class="pp-mut">A</span>TCGATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>A<span class="pp-mut">A</span>CGATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>AT<span class="pp-mut">A</span>GATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ATC<span class="pp-mut">A</span>ATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ATCG<span class="pp-mut">A</span>TCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT
    <span class="pp-ellipsis">... (32 total)</span>
    </div>

Random motif replacement (mode="random")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode='random'`` draws a random window position and replacement each time.
Here a degenerate 6-base IUPAC motif (``R`` = A|G, ``Y`` = C|T) is scanned
across a 16-mer, with 5 random draws.

.. code-block:: python

    wt    = pp.from_seq("ACGTACGTACGTACGT")
    motif = pp.from_iupac("RRYYYY")
    scan  = wt.replacement_scan(replacement_pool=motif, mode="random",
                                num_states=5, style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=16, num_states=5</em>
    ACGTACGTAC<span class="pp-mut">GACCCT</span><br>
    ACGTACGTAC<span class="pp-mut">GATCTT</span><br>
    A<span class="pp-mut">GACCTT</span>TACGTACGT<br>
    AC<span class="pp-mut">AACTTT</span>ACGTACGT<br>
    ACGTA<span class="pp-mut">AATTCC</span>TACGT
    </div>