Operations
==========

PoolParty provides composable operations for designing DNA sequence libraries.
Each operation returns a new :class:`~poolparty.Pool`, so calls can be chained
into a declarative pipeline. All examples assume:

.. code-block:: python

    import poolparty as pp
    pp.init()

Output blocks show representative pool contents. Colour indicates
**regions** (blue), **mutations** (red), and
**insertions** (green). Stochastic operations show example draws.

.. contents:: On this page
   :local:
   :depth: 1

----

Creating Pools
--------------

from_seq
~~~~~~~~

Create a pool containing a single sequence. Inline region tags delimit named
segments for later targeting.

.. code-block:: python

    wt = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence)</em>
    AAAA<span class="pp-region">ATCGATCG</span>TTTT
    </div>

See :func:`~poolparty.from_seq`.

from_seqs
~~~~~~~~~

Create a pool that draws uniformly from a list of sequences.

.. code-block:: python

    pool = pp.from_seqs(["AAAA", "CCCC", "GGGG", "TTTT"])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (4 sequences)</em>
    AAAA<br>CCCC<br>GGGG<br>GGGG
    </div>

See :func:`~poolparty.from_seqs`.

from_fasta
~~~~~~~~~~

Load sequences directly from a FASTA file.

.. code-block:: python

    pool = pp.from_fasta("sequences.fa")

See :func:`~poolparty.from_fasta`.

from_iupac
~~~~~~~~~~

Create a pool from an IUPAC ambiguity-code sequence, sampling each ambiguous
position according to the corresponding base set.

.. code-block:: python

    # W = A or T; S = C or G; N = any base
    pool = pp.from_iupac("WSNN")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (64 sequences) &mdash; W={A,T} &nbsp;S={C,G} &nbsp;N={A,C,G,T}</em>
    ACAA<br>ACAC<br>ACAG<br>ACAT<br>ACCA<br>
    <span class="pp-ellipsis">... (64 total)</span>
    </div>

See :func:`~poolparty.from_iupac`.

from_motif
~~~~~~~~~~

Sample sequences from a position-probability matrix (motif), supplied as a
pandas DataFrame with base columns (A, C, G, T) and one row per position.

.. code-block:: python

    import pandas as pd
    pfm = pd.DataFrame(
        {"A": [0.7, 0.1], "C": [0.1, 0.7], "G": [0.1, 0.1], "T": [0.1, 0.1]}
    )
    pool = pp.from_motif(pfm)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; each draw sampled from position-probability matrix)</em>
    AC<br>AC<br>AA<br>AC<br>
    <span class="pp-ellipsis">... biased toward A at position 1, C at position 2</span>
    </div>

See :func:`~poolparty.from_motif`.

get_kmers
~~~~~~~~~

Enumerate every k-mer of a given length over the specified alphabet.

.. code-block:: python

    # All 256 4-mers
    kmers = pp.get_kmers(length=4, alphabet="ACGT")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (256 sequences)</em>
    AAAA<br>AAAC<br>AAAG<br>AAAT<br>AACA<br>
    <span class="pp-ellipsis">... (256 total)</span>
    </div>

See :func:`~poolparty.get_kmers`.

----

Combining Pools
---------------

join
~~~~

Concatenate a list of pools end-to-end, producing one composite sequence per
draw.

.. code-block:: python

    promoter = pp.from_seq("GCGCGC")
    insert   = pp.from_seqs(["AAA", "TTT", "CCC"])
    spacer   = pp.from_seq("TTTT")
    library  = pp.join([promoter, insert, spacer])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (3 sequences)</em>
    GCGCGCAAATTTT<br>GCGCGCTTTTTTT<br>GCGCGCCCCTTTT
    </div>

See :func:`~poolparty.join`.

``+`` operator
~~~~~~~~~~~~~~

Shorthand for two-pool concatenation.

.. code-block:: python

    left   = pp.from_seq("AAAA")
    middle = pp.from_seqs(["G", "C"])
    right  = pp.from_seq("TTTT")
    combo  = left + middle + right

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (2 sequences)</em>
    AAAAGTTTT<br>AAAACTTTT
    </div>

``*`` operator
~~~~~~~~~~~~~~

Repeat a pool N times end-to-end (equivalent to ``join([pool] * N)``).

.. code-block:: python

    repeat4 = pp.from_seqs(["A", "C", "G", "T"]) * 4

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (4 sequences &mdash; each element concatenated with itself 4&times;)</em>
    AAAA<br>CCCC<br>GGGG<br>TTTT
    </div>

----

Sequence Mutagenesis
--------------------

mutagenize
~~~~~~~~~~

Introduce random point mutations at a specified rate or count. Pass
``region=`` to restrict mutations to a tagged segment.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    mutants = wt.mutagenize(num_mutations=1, region="cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 1 random point mutation per draw within <em>cre</em>)</em>
    AAAA<span class="pp-region">A<span class="pp-mut">G</span>CGATCG</span>TTTT<br>
    AAAA<span class="pp-region">ATCG<span class="pp-mut">C</span>TCG</span>TTTT<br>
    AAAA<span class="pp-region">ATCGAT<span class="pp-mut">A</span>G</span>TTTT<br>
    <span class="pp-ellipsis">... each draw has a unique single substitution inside cre</span>
    </div>

See :func:`~poolparty.mutagenize`.

shuffle_seq
~~~~~~~~~~~

Randomly permute the bases of a sequence (or of a tagged region).

.. code-block:: python

    wt       = pp.from_seq("ATCGATCGATCG")
    shuffled = wt.shuffle_seq()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; random permutation of bases each draw)</em>
    CTAGCGATCGAT<br>GATCGTAGCATC<br>ATCGATGCCGTA<br>
    <span class="pp-ellipsis">... each draw is a unique shuffle of the 12 bases</span>
    </div>

See :func:`~poolparty.shuffle_seq`.

recombine
~~~~~~~~~

Stochastically recombine the pool with one or more source sequences at random
breakpoints, producing chimeric sequences.

.. code-block:: python

    wt  = pp.from_seq("AAAAAAAAAA")
    alt = pp.from_seq("CCCCCCCCCC")
    rec = wt.recombine(sources=[alt], num_breakpoints=2)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; chimeras between wt and alt at 2 breakpoints)</em>
    AA<span class="pp-ins">CCCCCC</span>AA<br>
    <span class="pp-ins">CCCC</span>AAAA<span class="pp-ins">CC</span><br>
    AAA<span class="pp-ins">CCC</span>AAAA<br>
    <span class="pp-ellipsis">... each draw is a unique chimera</span>
    </div>

See :func:`~poolparty.recombine`.

----

Scan Operations
---------------

Scan operations tile a window across the sequence (or a named region) and
enumerate one variant per window position.

replacement_scan
~~~~~~~~~~~~~~~~

At each position, replace the window with sequences drawn from ``ins_pool``.

.. code-block:: python

    wt       = pp.from_seq("ATCGATCG")
    alt      = pp.from_seqs(["A", "C", "G", "T"])
    scan     = wt.replacement_scan(ins_pool=alt)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (32 sequences &mdash; 8 positions &times; 4 substitutions)</em>
    <span class="pp-mut">A</span>TCGATCG<br>
    <span class="pp-mut">C</span>TCGATCG<br>
    <span class="pp-mut">G</span>TCGATCG<br>
    A<span class="pp-mut">A</span>CGATCG<br>
    A<span class="pp-mut">C</span>CGATCG<br>
    <span class="pp-ellipsis">... (32 total)</span>
    </div>

See :func:`~poolparty.replacement_scan`.

deletion_scan
~~~~~~~~~~~~~

Systematically delete windows of a fixed length across the sequence.
Deleted bases are replaced with the ``deletion_marker`` (default ``"-"``).

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    dels = wt.deletion_scan(deletion_length=2)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (7 sequences &mdash; one 2-base deletion per position)</em>
    <span class="pp-del">--</span>CGATCG<br>
    A<span class="pp-del">--</span>GATCG<br>
    AT<span class="pp-del">--</span>ATCG<br>
    ATC<span class="pp-del">--</span>TCG<br>
    ATCG<span class="pp-del">--</span>CG<br>
    ATCGA<span class="pp-del">--</span>G<br>
    ATCGAT<span class="pp-del">--</span>
    </div>

See :func:`~poolparty.deletion_scan`.

insertion_scan
~~~~~~~~~~~~~~

Insert sequences from ``ins_pool`` at each position.

.. code-block:: python

    wt     = pp.from_seq("ATCGATCG")
    insert = pp.from_iupac("NN")        # all 2-mer inserts
    scan   = wt.insertion_scan(ins_pool=insert)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (144 sequences &mdash; 9 positions &times; 16 inserts)</em>
    <span class="pp-ins">AA</span>ATCGATCG<br>
    <span class="pp-ins">AC</span>ATCGATCG<br>
    A<span class="pp-ins">AA</span>TCGATCG<br>
    A<span class="pp-ins">AC</span>TCGATCG<br>
    <span class="pp-ellipsis">... (144 total)</span>
    </div>

See :func:`~poolparty.insertion_scan`.

shuffle_scan
~~~~~~~~~~~~

Shuffle the bases within a sliding window at each position.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCGATCG")
    scan = wt.shuffle_scan(shuffle_length=3)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 10 window positions, each window independently shuffled)</em>
    <span class="pp-mut">CAT</span>GATCGATCG<br>
    A<span class="pp-mut">CTG</span>ATCGATCG<br>
    AT<span class="pp-mut">GAC</span>TCGATCG<br>
    <span class="pp-ellipsis">... (10 positions, each stochastic)</span>
    </div>

See :func:`~poolparty.shuffle_scan`.

mutagenize_scan
~~~~~~~~~~~~~~~

Apply random mutations within a sliding window, tiling across the sequence.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCGATCG")
    scan = wt.mutagenize_scan(mutagenize_length=4, num_mutations=1)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 9 window positions, 1 mutation per window)</em>
    <span class="pp-mut">G</span>TCGATCGATCG<br>
    AT<span class="pp-mut">A</span>GATCGATCG<br>
    ATCG<span class="pp-mut">T</span>TCGATCG<br>
    <span class="pp-ellipsis">... (9 positions, each stochastic)</span>
    </div>

See :func:`~poolparty.mutagenize_scan`.

----

Region Operations
-----------------

Region operations target named segments delimited by XML-like tags
(``<name>...</name>`` or ``<name/>``).

insert_tags
~~~~~~~~~~~

Programmatically add region tags to a sequence by specifying start/stop
indices.

.. code-block:: python

    wt      = pp.from_seq("AAAAATCGATCGTTTT")
    tagged  = wt.insert_tags("cre", start=4, stop=12)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; cre region tagged at positions 4&ndash;12)</em>
    AAAA<span class="pp-region">ATCGATCG</span>TTTT
    </div>

See :func:`~poolparty.insert_tags`.

remove_tags
~~~~~~~~~~~

Remove region tags while keeping (or discarding) their contents.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    cleaned = wt.remove_tags("cre")           # keeps ATCG
    dropped = wt.remove_tags("cre", keep_content=False)  # drops it

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">cleaned (1 sequence &mdash; tag removed, content kept)</em>
    AAAATCGTTTT<br>
    <em class="pp-header" style="margin-top:6px;">dropped (1 sequence &mdash; tag and content removed)</em>
    AAAATTTT
    </div>

See :func:`~poolparty.remove_tags`.

annotate_region
~~~~~~~~~~~~~~~

Tag a region by position range and optionally apply a display style.

.. code-block:: python

    wt       = pp.from_seq("AAAAATCGATCGTTTT")
    annotated = wt.annotate_region("cre", extent=(4, 12), style="bold_green")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; cre region styled bold green)</em>
    AAAA<span style="color:#15803d;font-weight:bold;">ATCGATCG</span>TTTT
    </div>

See :func:`~poolparty.annotate_region`.

replace_region
~~~~~~~~~~~~~~

Replace the content of a named region with sequences drawn from
``content_pool``.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    inserts = pp.from_iupac("NNNN")
    swapped = wt.replace_region(content_pool=inserts, region_name="cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (256 sequences &mdash; all 4-mers substituted into <em>cre</em>)</em>
    AAAA<span class="pp-region">AAAA</span>TTTT<br>
    AAAA<span class="pp-region">AAAC</span>TTTT<br>
    AAAA<span class="pp-region">AAAG</span>TTTT<br>
    AAAA<span class="pp-region">AAAT</span>TTTT<br>
    AAAA<span class="pp-region">AACA</span>TTTT<br>
    <span class="pp-ellipsis">... (256 total)</span>
    </div>

See :func:`~poolparty.replace_region`.

apply_at_region
~~~~~~~~~~~~~~~

Apply an arbitrary transformation function to the content of a named region.

.. code-block:: python

    wt         = pp.from_seq("AAAA<cre>atcg</cre>TTTT")
    uppercased = wt.apply_at_region("cre", transform_fn=str.upper)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; lowercase cre content uppercased)</em>
    AAAA<span class="pp-region">ATCG</span>TTTT
    </div>

See :func:`~poolparty.apply_at_region`.

region_scan
~~~~~~~~~~~

Run a scan operation confined to a named region.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    alt     = pp.from_seqs(["A", "C", "G", "T"])
    scan    = wt.region_scan("cre", ins_pool=alt)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (32 sequences &mdash; replacement scan within <em>cre</em> only; flanks unchanged)</em>
    AAAA<span class="pp-region"><span class="pp-mut">A</span>TCGATCG</span>TTTT<br>
    AAAA<span class="pp-region"><span class="pp-mut">C</span>TCGATCG</span>TTTT<br>
    AAAA<span class="pp-region"><span class="pp-mut">G</span>TCGATCG</span>TTTT<br>
    AAAA<span class="pp-region">A<span class="pp-mut">A</span>CGATCG</span>TTTT<br>
    <span class="pp-ellipsis">... (32 total)</span>
    </div>

See :func:`~poolparty.region_scan`.

----

Multiscan Operations
--------------------

Multiscan operations simultaneously introduce multiple scan windows, producing
libraries with combinatorial coverage.

replacement_multiscan
~~~~~~~~~~~~~~~~~~~~~

Place ``num_replacements`` non-overlapping replacement windows across the
sequence simultaneously.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCGATCG")
    alt  = pp.from_seqs(["A", "C", "G", "T"])
    scan = wt.replacement_multiscan(num_replacements=2, replacement_pools=alt)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 2 simultaneous non-overlapping substitutions per draw)</em>
    <span class="pp-mut">C</span>TCGA<span class="pp-mut">G</span>CGATCG<br>
    A<span class="pp-mut">A</span>CGAT<span class="pp-mut">T</span>GATCG<br>
    ATCG<span class="pp-mut">C</span>TCG<span class="pp-mut">A</span>TCG<br>
    <span class="pp-ellipsis">... each draw places 2 substitutions at distinct positions</span>
    </div>

See :func:`~poolparty.replacement_multiscan`.

deletion_multiscan
~~~~~~~~~~~~~~~~~~

Apply multiple simultaneous deletion windows.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCGATCG")
    scan = wt.deletion_multiscan(num_deletions=2, deletion_length=2)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 2 simultaneous 2-base deletions per draw)</em>
    <span class="pp-del">--</span>CG<span class="pp-del">--</span>TCGATCG<br>
    AT<span class="pp-del">--</span>ATC<span class="pp-del">--</span>TCGG<br>
    ATCG<span class="pp-del">--</span>CG<span class="pp-del">--</span>CG<br>
    <span class="pp-ellipsis">... each draw places 2 non-overlapping deletions</span>
    </div>

See :func:`~poolparty.deletion_multiscan`.

insertion_multiscan
~~~~~~~~~~~~~~~~~~~

Insert sequences at multiple positions simultaneously.

.. code-block:: python

    wt     = pp.from_seq("ATCGATCGATCG")
    insert = pp.from_iupac("NN")
    scan   = wt.insertion_multiscan(num_insertions=2, ins_pool=insert)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 2 simultaneous insertions per draw)</em>
    <span class="pp-ins">AT</span>ATCG<span class="pp-ins">GC</span>ATCGATCG<br>
    A<span class="pp-ins">CC</span>TCGATCG<span class="pp-ins">TT</span>ATCG<br>
    <span class="pp-ellipsis">... each draw places 2 non-overlapping insertions</span>
    </div>

See :func:`~poolparty.insertion_multiscan`.

----

State Operations
----------------

State operations manipulate the combinatorial state space, controlling how
many sequences are drawn and in what order.

stack
~~~~~

Combine multiple pools into one by stacking their state spaces. Each draw
independently samples from one of the constituent pools.

.. code-block:: python

    a = pp.from_seq("AAAA")
    b = pp.from_seq("CCCC")
    c = pp.from_seq("GGGG")
    combined = pp.stack([a, b, c])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (3 sequences &mdash; union of all three pools)</em>
    AAAA<br>CCCC<br>GGGG
    </div>

See :func:`~poolparty.stack`.

repeat
~~~~~~

Repeat the state space of a pool ``times`` times, producing duplicate draws.

.. code-block:: python

    wt       = pp.from_seqs(["AAAA", "CCCC"])
    tripled  = wt.repeat(times=3)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (6 sequences &mdash; each sequence repeated 3&times;)</em>
    AAAA<br>AAAA<br>AAAA<br>CCCC<br>CCCC<br>CCCC
    </div>

See :func:`~poolparty.repeat`.

sample
~~~~~~

Draw a fixed number of sequences, optionally with a random seed for
reproducibility.

.. code-block:: python

    kmers    = pp.get_kmers(length=6, alphabet="ACGT")
    subset   = kmers.sample(num_seqs=100, seed=42)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (100 sequences &mdash; random subset of 4,096 6-mers, seed=42)</em>
    AACGTT<br>TGCAAC<br>CGTTAA<br>GACTCG<br>TTAACC<br>
    <span class="pp-ellipsis">... (100 total)</span>
    </div>

See :func:`~poolparty.sample`.

slice_states
~~~~~~~~~~~~

Retain a slice of the state space using standard Python slice syntax.

.. code-block:: python

    kmers  = pp.get_kmers(length=4, alphabet="ACGT")
    first8 = kmers.slice_states(slice(0, 8))

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (8 sequences &mdash; first 8 of 256 4-mers)</em>
    AAAA<br>AAAC<br>AAAG<br>AAAT<br>AACA<br>AACC<br>AACG<br>AACT
    </div>

See :func:`~poolparty.state_slice`.

shuffle_states
~~~~~~~~~~~~~~

Randomly reorder the state space, optionally with a fixed seed.

.. code-block:: python

    kmers     = pp.get_kmers(length=3, alphabet="ACGT")
    shuffled  = kmers.shuffle_states(seed=0)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (64 sequences &mdash; 3-mers in shuffled order, seed=0)</em>
    GTA<br>CAT<br>TGC<br>ACG<br>GAT<br>
    <span class="pp-ellipsis">... (64 total)</span>
    </div>

See :func:`~poolparty.state_shuffle`.

----

ORF Operations
--------------

ORF operations are codon-aware and preserve reading-frame integrity.

mutagenize_orf
~~~~~~~~~~~~~~

Introduce missense mutations at the codon level within a coding sequence.

.. code-block:: python

    # ATG AAA TTT GGG CCC = ATGAAATTTGGGCCC
    cds      = pp.from_seq("ATGAAATTTGGGCCC")
    mutants  = cds.mutagenize_orf(num_mutations=1)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 1 random codon substitution per draw)</em>
    ATG<span class="pp-mut">CGT</span>TTTGGGCCC<br>
    ATGAAA<span class="pp-mut">ACT</span>GGGCCC<br>
    ATGAAATTT<span class="pp-mut">CAT</span>CCC<br>
    <span class="pp-ellipsis">... each draw mutates one codon to a different amino acid</span>
    </div>

See :func:`~poolparty.mutagenize_orf`.

translate
~~~~~~~~~

Translate a DNA pool to a protein pool.

.. code-block:: python

    cds     = pp.from_seq("ATGAAATTTGGGCCC")
    protein = cds.translate()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; ATG&thinsp;AAA&thinsp;TTT&thinsp;GGG&thinsp;CCC &rarr; M&thinsp;K&thinsp;F&thinsp;G&thinsp;P)</em>
    MKFGP
    </div>

See :func:`~poolparty.translate`.

annotate_orf
~~~~~~~~~~~~

Tag and optionally style an ORF region within a longer sequence.

.. code-block:: python

    seq  = pp.from_seq("TATAATGAAATTTGGGCCCTAA")
    seq  = seq.annotate_orf("gene", extent=(3, 21))

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; ORF tagged as <em>gene</em>)</em>
    TAT<span class="pp-region">ATGAAATTTGGGCCC</span>TAA
    </div>

See :func:`~poolparty.annotate_orf`.

stylize_orf
~~~~~~~~~~~

Apply codon-aware coloring to visualize reading frames.

.. code-block:: python

    cds    = pp.from_seq("ATGAAATTTGGGCCC")
    styled = cds.stylize_orf()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; alternating codon colours)</em>
    <span class="pp-codon-a">ATG</span><span class="pp-codon-b">AAA</span><span class="pp-codon-a">TTT</span><span class="pp-codon-b">GGG</span><span class="pp-codon-a">CCC</span>
    </div>

See :func:`~poolparty.stylize_orf`.

----

Fixed Operations
----------------

Fixed operations apply deterministic, length-preserving transformations.

rc
~~

Take the reverse complement of a sequence or a named region.

.. code-block:: python

    wt = pp.from_seq("ATCG")
    r  = wt.rc()   # CGAT

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; ATCG &rarr; reverse complement)</em>
    CGAT
    </div>

See :func:`~poolparty.rc`.

upper / lower / swapcase
~~~~~~~~~~~~~~~~~~~~~~~~~

Convert sequence case. Useful for visually distinguishing regions.

.. code-block:: python

    wt  = pp.from_seq("atcg")
    up  = wt.upper()     # ATCG
    lo  = wt.lower()     # atcg
    sw  = wt.swapcase()  # ATCG → atcg

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">up &nbsp;/&nbsp; lo &nbsp;/&nbsp; sw (1 sequence each, applied to <code>atcg</code>)</em>
    ATCG<br>atcg<br>ATCG
    </div>

See :func:`~poolparty.upper`, :func:`~poolparty.lower`.

stylize
~~~~~~~

Apply a named display style to a sequence or region for terminal
visualization.

.. code-block:: python

    wt     = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    styled = wt.stylize(region="cre", style="bold_red")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; cre region styled bold red)</em>
    AAAA<span style="color:#dc2626;font-weight:bold;">ATCG</span>TTTT
    </div>

See :func:`~poolparty.stylize`.

clear_gaps
~~~~~~~~~~

Remove gap characters (``-``) from sequences.

.. code-block:: python

    wt      = pp.from_seq("AT--CG")
    cleaned = wt.clear_gaps()   # ATCG

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; gap markers removed)</em>
    ATCG
    </div>

See :func:`~poolparty.clear_gaps`.

clear_annotation
~~~~~~~~~~~~~~~~

Remove annotation markup while keeping the underlying sequence.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    plain   = wt.clear_annotation()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; region tags stripped)</em>
    AAAATCGTTTT
    </div>

See :func:`~poolparty.clear_annotation`.

----

Filtering
---------

filter
~~~~~~

Retain only sequences that satisfy a predicate. Sequences failing the
predicate are marked as null and excluded from output when
``generate_library()`` is called.

.. code-block:: python

    pool     = pp.get_kmers(length=6, alphabet="ACGT")
    filtered = pool.filter(lambda s: s.count("G") + s.count("C") >= 3)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic subset &mdash; 6-mers with GC&nbsp;&ge;&nbsp;3 pass filter)</em>
    GCGCGC<br>AACGCG<br>CCGATT<br>GGCTAT<br>TTCGGC<br>
    <span class="pp-ellipsis">... (~half of 4,096 6-mers satisfy GC &ge; 3)</span>
    </div>

See :meth:`~poolparty.Pool.filter`.

----

Library Generation
------------------

generate_library
~~~~~~~~~~~~~~~~

Evaluate the pool DAG and return a :class:`~pandas.DataFrame` containing
sequences, names, and optional metadata columns.

.. code-block:: python

    wt       = pp.from_seq("ATCGATCG")
    mutants  = wt.mutagenize(num_mutations=1)
    df       = mutants.generate_library(num_seqs=200, seed=0)
    print(df[["seq", "seq_name"]].head())

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">DataFrame (200 rows)</em>
    <table style="border-collapse:collapse;font-size:0.9em;width:auto;margin-top:4px;">
    <tr><th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">seq</th><th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">seq_name</th></tr>
    <tr><td style="padding:1px 14px 1px 0;">A<span class="pp-mut">G</span>CGATCG</td><td style="padding:1px 14px 1px 0;color:#6b7280;">seq_0001</td></tr>
    <tr><td style="padding:1px 14px 1px 0;">ATCG<span class="pp-mut">C</span>TCG</td><td style="padding:1px 14px 1px 0;color:#6b7280;">seq_0002</td></tr>
    <tr><td style="padding:1px 14px 1px 0;">ATCGA<span class="pp-mut">G</span>CG</td><td style="padding:1px 14px 1px 0;color:#6b7280;">seq_0003</td></tr>
    <tr><td colspan="2" style="padding:1px 0;color:#9ca3af;font-style:italic;">... (200 total)</td></tr>
    </table>
    </div>

Key parameters:

- ``num_seqs`` — how many sequences to generate
- ``num_cycles`` — how many full state-space cycles to run
- ``seed`` — random seed for reproducibility
- ``seqs_only`` — return a plain list instead of a DataFrame

See :func:`~poolparty.generate_library`.

materialize
~~~~~~~~~~~

Eagerly generate sequences from the current pool and store them in a new,
standalone pool. Useful for caching expensive upstream operations.

.. code-block:: python

    wt        = pp.from_seq("ATCGATCGATCG")
    expensive = wt.mutagenize(num_mutations=2)
    cached    = expensive.materialize(num_seqs=500, seed=0)

    # downstream ops are now built on the cached pool
    library = cached.deletion_scan(deletion_length=2)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">cached (500 sequences &mdash; materialized from upstream mutagenize, then fed to deletion_scan)</em>
    ATCG<span class="pp-mut">C</span>T<span class="pp-mut">A</span>GATCG<br>
    <span class="pp-mut">G</span>TCGA<span class="pp-mut">T</span>CGATCG<br>
    ATCGATCG<span class="pp-mut">G</span>T<span class="pp-mut">C</span>G<br>
    <span class="pp-ellipsis">... (500 total)</span>
    </div>

See :meth:`~poolparty.Pool.materialize`.