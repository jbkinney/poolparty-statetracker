insertion_scan
==============

Insert sequences from ``ins_pool`` at every position along the background
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
   :widths: 20 18 12 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pool``
     - ``Pool | str``
     - *(required)*
     - The background Pool to scan. Can also be a plain sequence string.
   * - ``ins_pool``
     - ``Pool``
     - *(required)*
     - Pool whose sequences are inserted at each scanned position. An
       *L*-mer has *L* + 1 valid insertion sites (before each base and
       after the last).
   * - ``positions``
     - ``list[int] | None``
     - ``None``
     - Explicit list of insertion positions. ``None`` = all valid positions.
   * - ``region``
     - ``str | None``
     - ``None``
     - Name of a tagged region to restrict insertions to. Flanking sequences
       are never modified.
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
     - Fix the total number of output states.
   * - ``iter_order``
     - ``list[str] | None``
     - ``None``
     - Controls which axis varies fastest, e.g. ``["insert","position"]``.

----

Examples
--------

Single-base insertions at every position
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An 8-mer has 9 insertion sites. 9 sites × 4 bases = 36 sequences, each of
length 9.

.. code-block:: python

    wt    = pp.from_seq("ACGTACGT")
    bases = pp.from_seqs(["A", "C", "G", "T"], mode="sequential")
    scan  = wt.insertion_scan(ins_pool=bases, mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (36 sequences &mdash; 9 positions &times; 4 single-base inserts; output length 9)</em>
    <span class="pp-ins">A</span>ACGTACGT<br>
    <span class="pp-ins">C</span>ACGTACGT<br>
    <span class="pp-ins">G</span>ACGTACGT<br>
    <span class="pp-ins">T</span>ACGTACGT<br>
    A<span class="pp-ins">A</span>CGTACGT<br>
    <span class="pp-ellipsis">... (36 total)</span>
    </div>

All-dinucleotide insertions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``from_iupac("NN")`` to enumerate all 16 dinucleotide inserts.
9 sites × 16 inserts = 144 sequences, each of length 10.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    nn   = pp.from_iupac("NN", mode="sequential")
    scan = wt.insertion_scan(ins_pool=nn, mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (144 sequences &mdash; 9 positions &times; 16 dinucleotide inserts; length 10)</em>
    <span class="pp-ins">AA</span>ACGTACGT<br>
    <span class="pp-ins">AC</span>ACGTACGT<br>
    <span class="pp-ins">AG</span>ACGTACGT<br>
    <span class="pp-ins">AT</span>ACGTACGT<br>
    A<span class="pp-ins">AA</span>CGTACGT<br>
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
    scan  = wt.insertion_scan(ins_pool=bases, replace=True, mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (28 sequences &mdash; 7 positions &times; 4 inserts; 2-base window replaced; output length 8)</em>
    <span class="pp-ins">AA</span>CGTACGT<br>
    <span class="pp-ins">CC</span>CGTACGT<br>
    <span class="pp-ins">GG</span>CGTACGT<br>
    <span class="pp-ins">TT</span>CGTACGT<br>
    A<span class="pp-ins">AA</span>GTACGT<br>
    <span class="pp-ellipsis">... (28 total)</span>
    </div>

Insertion scan within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict insertion sites to the ``cre`` region. The 8-base region has 9
valid insertion sites; flanks are never altered.

.. code-block:: python

    wt    = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    bases = pp.from_seqs(["A", "C", "G", "T"], mode="sequential")
    scan  = wt.insertion_scan(ins_pool=bases, region="cre", mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (36 sequences &mdash; 9 positions within <em>cre</em> &times; 4 inserts; output length 17)</em>
    AAAA<span class="pp-region"><span class="pp-ins">A</span>ATCGATCG</span>TTTT<br>
    AAAA<span class="pp-region"><span class="pp-ins">C</span>ATCGATCG</span>TTTT<br>
    AAAA<span class="pp-region"><span class="pp-ins">G</span>ATCGATCG</span>TTTT<br>
    AAAA<span class="pp-region">A<span class="pp-ins">A</span>TCGATCG</span>TTTT<br>
    <span class="pp-ellipsis">... (36 total; flanks always unchanged)</span>
    </div>

Explicit position list
~~~~~~~~~~~~~~~~~~~~~~~

Limit the scan to specific insertion sites.

.. code-block:: python

    wt    = pp.from_seq("ACGTACGT")
    bases = pp.from_seqs(["A", "C", "G", "T"], mode="sequential")
    scan  = wt.insertion_scan(ins_pool=bases, positions=[0, 4, 8], mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (12 sequences &mdash; 3 positions &times; 4 inserts)</em>
    <span class="pp-ins">A</span>ACGTACGT<br>
    <span class="pp-ins">C</span>ACGTACGT<br>
    ACGT<span class="pp-ins">A</span>ACGT<br>
    ACGT<span class="pp-ins">C</span>ACGT<br>
    <span class="pp-ellipsis">... (12 total)</span>
    </div>

See :func:`~poolparty.insertion_scan`.
