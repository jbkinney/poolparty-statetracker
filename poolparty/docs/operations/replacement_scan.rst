replacement_scan
================

Slide a window across the sequence and, at each position, replace the window
with every sequence drawn from ``ins_pool``. The output length equals the
background sequence length.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 18 12 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pool``
     - ``Pool | str``
     - *(required)*
     - The background Pool or sequence string that is scanned.
   * - ``ins_pool``
     - ``Pool | str``
     - *(required)*
     - Pool whose sequences replace the window at each scanned position. The
       pool's sequence length determines the window width.
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
     - Fix the total number of output states.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``prefix_position``
     - ``str | None``
     - ``None``
     - Prefix for position index (e.g., 'pos_' produces 'pos_0', 'pos_1', ...).
   * - ``prefix_insert``
     - ``str | None``
     - ``None``
     - Prefix for insert index (e.g., 'ins_' produces 'ins_0', 'ins_1', ...).
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.


----

Examples
--------

Single-base replacement at every position
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace each base with every member of a four-base pool — 8 positions × 4
substitutions = 32 sequences.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    alt  = pp.from_seqs(["A", "C", "G", "T"], mode="sequential")
    scan = wt.replacement_scan(ins_pool=alt, mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (32 sequences &mdash; 8 positions &times; 4 substitutions)</em>
    <span class="pp-mut">A</span>CGTACGT<br>
    <span class="pp-mut">C</span>CGTACGT<br>
    <span class="pp-mut">G</span>CGTACGT<br>
    A<span class="pp-mut">A</span>GTACGT<br>
    A<span class="pp-mut">C</span>GTACGT<br>
    <span class="pp-ellipsis">... (32 total)</span>
    </div>

Trinucleotide window scan
~~~~~~~~~~~~~~~~~~~~~~~~~~

The window width is determined by ``ins_pool`` sequence length. A 3-base
pool scans 6 positions across an 8-mer.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    tri  = pp.from_seqs(["AAA", "CCC", "GGG", "TTT"], mode="sequential")
    scan = wt.replacement_scan(ins_pool=tri, mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (24 sequences &mdash; 6 windows &times; 4 trinucleotide substitutions)</em>
    <span class="pp-mut">AAA</span>TACGT<br>
    <span class="pp-mut">CCC</span>TACGT<br>
    <span class="pp-mut">GGG</span>TACGT<br>
    <span class="pp-mut">TTT</span>TACGT<br>
    A<span class="pp-mut">AAA</span>ACGT<br>
    <span class="pp-ellipsis">... (24 total)</span>
    </div>

All-dinucleotide replacements via from_iupac
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``from_iupac("NN")`` enumerates all 16 dinucleotides. Seven windows × 16
substitutions = 112 sequences.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    nn   = pp.from_iupac("NN", mode="sequential")
    scan = wt.replacement_scan(ins_pool=nn, mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (112 sequences &mdash; 7 windows &times; 16 dinucleotide substitutions)</em>
    <span class="pp-mut">AA</span>GTACGT<br>
    <span class="pp-mut">AC</span>GTACGT<br>
    <span class="pp-mut">AG</span>GTACGT<br>
    <span class="pp-mut">AT</span>GTACGT<br>
    A<span class="pp-mut">AA</span>TACGT<br>
    <span class="pp-ellipsis">... (112 total)</span>
    </div>

Scan restricted to a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict the scan to the ``cre`` region; the ``AAAA`` and ``TTTT`` flanks
are never modified.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    alt  = pp.from_seqs(["A", "C", "G", "T"], mode="sequential")
    scan = wt.replacement_scan(ins_pool=alt, region="cre", mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (32 sequences &mdash; 8 positions within <em>cre</em> &times; 4 substitutions)</em>
    AAAA<span class="pp-region"><span class="pp-mut">A</span>TCGATCG</span>TTTT<br>
    AAAA<span class="pp-region"><span class="pp-mut">C</span>TCGATCG</span>TTTT<br>
    AAAA<span class="pp-region"><span class="pp-mut">G</span>TCGATCG</span>TTTT<br>
    AAAA<span class="pp-region">A<span class="pp-mut">A</span>CGATCG</span>TTTT<br>
    <span class="pp-ellipsis">... (32 total; flanks always unchanged)</span>
    </div>