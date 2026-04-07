MPRA Library for Regulatory Grammar
====================================

This tutorial designs a massively parallel reporter assay (MPRA) library
for probing transcriptional regulatory grammar. The library places three
liver-enriched transcription factor binding sites (TFBSs) at random
positions and orientations within a 100 bp candidate regulatory element
(CRE). Each unique CRE arrangement is paired with three distinct
barcodes for technical replication, yielding 24,000 barcoded sequences
that can be used to test how binding site configuration affects gene
expression.

The TFBS sequences (HNF4A, PPARA, XBP1) come from
Georgakopoulos-Soares et al. (*Nature Communications*, 2023), and the
oligo construct layout follows Melnikov et al. (*Nature Biotechnology*,
2012).

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Reference sequences
--------------------

The construct follows the Melnikov et al. oligo layout: a 5' adaptor, a
100 bp CRE region containing putatively inert background sequence, a
KpnI/XbaI restriction junction, an 8 bp barcode, and a 3' sequencing
adapter. The 100 bp background is drawn from a confirmed-negative
genomic region (Georgakopoulos-Soares et al., Supplementary Table 2).

.. code-block:: python

    BG1_100 = (
        "GCAAGTCTGCCATCGTGTTCAGAAGGGCCAGAAATGCCAAGGACTCAGGGGAGG"
        "AGAATTAAGTCAGAGAGTTTCATTACTGAGTGTTGTTTGACTTTGT"
    )

    MELNIKOV_5P = "ACTGGCCGCTTCACTG"       # 5' adaptor
    MELNIKOV_3P = "AGATCGGAAGAGCGTCG"      # sequencing adapter
    MELNIKOV_JUNCTION = "GGTACCTCTAGA"      # KpnI + XbaI

Build the template
------------------

The template contains two :doc:`tagged regions </regions>`:
``<cre>`` marks the 100 bp element where TFBSs will be placed, and
``<bc>`` marks the barcode placeholder (initially filled with ``N``
characters).

.. code-block:: python

    MPRA_TEMPLATE = (
        MELNIKOV_5P
        + "<cre>" + BG1_100 + "</cre>"
        + MELNIKOV_JUNCTION
        + "<bc>" + "N" * 8 + "</bc>"
        + MELNIKOV_3P
    )

    template = pp.from_seq(MPRA_TEMPLATE)

Create TFBS pools
-----------------

Each TFBS is created as a single-sequence pool, then passed through
:doc:`flip </operations/flip>` to include both forward and reverse-complement orientations.
Color :doc:`styles </metadata/styling>` make TFBSs visually
distinguishable in the output: HNF4A in blue, PPARA in purple, XBP1 in
orange.

.. code-block:: python

    hnf4a = pp.from_seq("GGGGCAAAGGTCA", style="blue").flip(mode="sequential")
    ppara = pp.from_seq("CCGGGTCATTGGGGTCAGG", style="purple").flip(mode="sequential")
    xbp1  = pp.from_seq("GTGATGACGTGTCCCAT", style="orange").flip(mode="sequential")

Each TFBS pool now contains two states (forward and reverse complement).

Insert TFBSs into the CRE region
---------------------------------

:doc:`insertion_multiscan </operations/insertion_multiscan>` places three TFBSs at random positions within
the ``<cre>`` region. The ``replace=True`` flag replaces the underlying
background bases so the total sequence length stays constant.
``insertion_mode="unordered"`` means the three sites can appear in any
order, and ``min_spacing=0`` allows binding sites to sit immediately
adjacent to each other.

.. code-block:: python

    cre_pool = template.insertion_multiscan(
        region="cre",
        insertion_pools=[hnf4a, ppara, xbp1],
        insertion_mode="unordered",
        replace=True,
        min_spacing=0,
        num_insertions=3,
        mode="random",
        num_states=1000,
    ).repeat(times=3)

The ``num_states=1000`` parameter draws 1,000 random position
configurations. Because each of the three TFBSs can appear in forward
or reverse-complement orientation, each configuration expands into
2\ :sup:`3` = 8 orientation combinations, giving 8,000 unique CRE
variants. :doc:`repeat </operations/repeat>` then replicates each
variant three times, yielding 24,000 CRE variants ready for barcode
assignment.

Generate and attach barcodes
----------------------------

Each CRE variant receives a unique 8 bp barcode.
:doc:`get_barcodes </operations/get_barcodes>` generates barcodes with controlled GC
content and minimum edit distance to ensure they are distinguishable by
sequencing.

.. code-block:: python

    barcode_pool = pp.get_barcodes(
        num_barcodes=cre_pool.num_states,
        length=8,
        gc_range=(0.3, 0.6),
        min_edit_distance=1,
        style="bold",
        seed=42,
    )

    mpra_pool = cre_pool.replace_region(
        region_name="bc",
        content_pool=barcode_pool,
    )

:doc:`replace_region </operations/replace_region>` with the default
``sync=True`` pairs each of the 24,000 CRE variants with exactly one
barcode. Because every unique CRE arrangement appears three times (from
``repeat``), each arrangement receives three distinct barcodes for
technical replication.

Inspect the library
-------------------

.. code-block:: python

    mpra_pool.print_library(num_seqs=15, seed=42)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mpra_pool: seq_length=153, num_states=24000</em>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GC<span class="pp-style-orange">GTGATGACGTGTCCCAT</span>CAGAAGGGCCAGAAATGCCAA<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span>TAAGTCAGAGAGTTTCATTACTGAGTG<span class="pp-style-blue">GGGGCAAAGGTCA</span>T<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">TGGAGAAA</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG<br>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GC<span class="pp-style-orange">GTGATGACGTGTCCCAT</span>CAGAAGGGCCAGAAATGCCAA<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span>TAAGTCAGAGAGTTTCATTACTGAGTG<span class="pp-style-blue">GGGGCAAAGGTCA</span>T<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">GCTGTCTT</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG<br>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GC<span class="pp-style-orange">GTGATGACGTGTCCCAT</span>CAGAAGGGCCAGAAATGCCAA<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span>TAAGTCAGAGAGTTTCATTACTGAGTG<span class="pp-style-blue">GGGGCAAAGGTCA</span>T<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">CCCGAATT</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG<br>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GCAAGTCTGCCATCGTGTTCAGA<span class="pp-style-blue">GGGGCAAAGGTCA</span>CCAA<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span>TAAGTCAGAGA<span class="pp-style-orange">GTGATGACGTGTCCCAT</span>TGTTTGACTTTGT<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">AAAGGGTC</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG<br>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GCAAGTCTGCCATCGTGTTCAGA<span class="pp-style-blue">GGGGCAAAGGTCA</span>CCAA<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span>TAAGTCAGAGA<span class="pp-style-orange">GTGATGACGTGTCCCAT</span>TGTTTGACTTTGT<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">ACCCACAA</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG<br>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GCAAGTCTGCCATCGTGTTCAGA<span class="pp-style-blue">GGGGCAAAGGTCA</span>CCAA<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span>TAAGTCAGAGA<span class="pp-style-orange">GTGATGACGTGTCCCAT</span>TGTTTGACTTTGT<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">AAGATCTG</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG<br>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GCAAGTCTGCCA<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span>AAATGCCAAGGACTCAG<span class="pp-style-orange">GTGATGACGTGTCCCAT</span>AGAGAGTTTCATTACT<span class="pp-style-blue">GGGGCAAAGGTCA</span>CTTTGT<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">CTGTTGTT</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG<br>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GCAAGTCTGCCA<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span>AAATGCCAAGGACTCAG<span class="pp-style-orange">GTGATGACGTGTCCCAT</span>AGAGAGTTTCATTACT<span class="pp-style-blue">GGGGCAAAGGTCA</span>CTTTGT<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">AGTCATGG</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG<br>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GCAAGTCTGCCA<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span>AAATGCCAAGGACTCAG<span class="pp-style-orange">GTGATGACGTGTCCCAT</span>AGAGAGTTTCATTACT<span class="pp-style-blue">GGGGCAAAGGTCA</span>CTTTGT<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">AGACTGGT</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG<br>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GCAA<span class="pp-style-blue">GGGGCAAAGGTCA</span>TTCAGAAGGGCCAGAAATGCCAAGGACT<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span><span class="pp-style-orange">GTGATGACGTGTCCCAT</span>GAGTGTTGTTTGACTTTGT<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">GAGGAACT</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG<br>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GCAA<span class="pp-style-blue">GGGGCAAAGGTCA</span>TTCAGAAGGGCCAGAAATGCCAAGGACT<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span><span class="pp-style-orange">GTGATGACGTGTCCCAT</span>GAGTGTTGTTTGACTTTGT<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">ATACAACC</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG<br>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GCAA<span class="pp-style-blue">GGGGCAAAGGTCA</span>TTCAGAAGGGCCAGAAATGCCAAGGACT<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span><span class="pp-style-orange">GTGATGACGTGTCCCAT</span>GAGTGTTGTTTGACTTTGT<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">ACCCAGAA</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG<br>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GCAAGTCTGCCATCGTGTTCAGAA<span class="pp-style-blue">GGGGCAAAGGTCA</span>CAAG<span class="pp-style-orange">GTGATGACGTGTCCCAT</span>TTAAGTCAGAGAGTTT<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span>ACTTTGT<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">GTTGAGCA</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG<br>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GCAAGTCTGCCATCGTGTTCAGAA<span class="pp-style-blue">GGGGCAAAGGTCA</span>CAAG<span class="pp-style-orange">GTGATGACGTGTCCCAT</span>TTAAGTCAGAGAGTTT<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span>ACTTTGT<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">ATCGTCTG</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG<br>
    ACTGGCCGCTTCACTG<span class="pp-xtag-light">&lt;cre&gt;</span>GCAAGTCTGCCATCGTGTTCAGAA<span class="pp-style-blue">GGGGCAAAGGTCA</span>CAAG<span class="pp-style-orange">GTGATGACGTGTCCCAT</span>TTAAGTCAGAGAGTTT<span class="pp-style-purple">CCGGGTCATTGGGGTCAGG</span>ACTTTGT<span class="pp-xtag-light">&lt;/cre&gt;</span>GGTACCTCTAGA<span class="pp-xtag-light">&lt;bc&gt;</span><span class="pp-style-bold">TTATGGGG</span><span class="pp-xtag-light">&lt;/bc&gt;</span>AGATCGGAAGAGCGTCG
    </div>

Each sequence shows the positions and orientations of the three TFBSs
(HNF4A in blue, PPARA in purple, XBP1 in orange) and the barcode in
bold. The ``<cre>`` and ``<bc>`` region tags are preserved so downstream
operations can continue to reference those regions. Notice that the
first three sequences share the same TFBS positions and orientations but
carry different barcodes, reflecting the three technical replicates
produced by ``repeat(times=3)``.

See :doc:`insertion_multiscan </operations/insertion_multiscan>`,
:doc:`flip </operations/flip>`,
:doc:`get_barcodes </operations/get_barcodes>`, and
:doc:`replace_region </operations/replace_region>` for full parameter
details. To export the library as a DataFrame or file, see ``to_df``
and ``to_file`` in :doc:`/pool`.
