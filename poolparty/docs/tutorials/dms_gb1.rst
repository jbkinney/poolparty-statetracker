Deep Mutational Scanning: Protein GB1
======================================

This tutorial builds a deep mutational scanning (DMS) library for the
IgG-binding domain of protein G (GB1), a 56-amino-acid protein domain.
This library extends the GB1 DMS study by Olson et al. (*Current Biology*, 2014)
by considering:

- All single amino acid substitutions
- All pairwise amino acid substitutions
- 10,000 random higher-order mutants
- 100 wild-type replicates


.. image:: /_static/images/figure2a.drawio.svg
   :width: 80%
   :align: center
   :alt: DMS library design DAG showing the pipeline from wild-type ORF through single, pairwise, and higher-order mutagenesis to the final stacked library.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Define the wild-type ORF
------------------------

The GB1 coding sequence is 168 bp (56 codons). We load it as a
single-sequence pool with :doc:`from_seq </operations/from_seq>` and target
codons 1 through 55 for mutagenesis, skipping the start codon at
position 0.

.. code-block:: python

    GB1_ORF = (
        "ATGCAGTACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAG"
        "ACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATG"
        "GACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA"
    )

    orf_pool = pp.from_seq(GB1_ORF).named("orf_pool")
    pos = slice(1, 56)

The ``codon_positions=slice(1, 56)`` range used below will target all
55 non-start codons for mutagenesis.

Single amino acid substitutions
-------------------------------

:doc:`mutagenize_orf </operations/mutagenize_orf>` in
:doc:`sequential mode </operations/modes>` with ``num_mutations=1``
generates every possible single amino acid
substitution. Each codon position has 19 possible missense changes (one
per alternative amino acid). Because most amino acids are encoded by
multiple codons, ``missense_only_first`` selects a single codon for each
target amino acid (the first listed in the codon table), avoiding
redundant synonymous alternatives. The ``style="red"`` parameter
highlights mutated codons in the output (see :doc:`/metadata/styling`).

.. code-block:: python

    single_pool = orf_pool.mutagenize_orf(
        num_mutations=1,
        mutation_type="missense_only_first",
        codon_positions=pos,
        prefix="single",
        style="red",
        mode="sequential",
        cards={"codon_positions": "position", "wt_aas": "wt_aa", "mut_aas": "mut_aa"},
    ).named("single_pool")

    single_pool.print_library(num_seqs=5, show_name=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">single_pool: seq_length=168, num_states=1045</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th></tr>
    <tr><td>single_0000</td><td>ATG<span class="pp-mut">TTC</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    <tr><td>single_0001</td><td>ATG<span class="pp-mut">CTG</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    <tr><td>single_0002</td><td>ATG<span class="pp-mut">ATC</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    <tr><td>single_0003</td><td>ATG<span class="pp-mut">ATG</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    <tr><td>single_0004</td><td>ATG<span class="pp-mut">GTG</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    </table>
    <span class="pp-ellipsis">... (1,045 total)</span>
    </div>

This yields 1,045 variants: 55 positions times 19 alternative amino
acids at each position. The first five variants shown above all mutate
codon 1 (Gln in the wild type) to different amino acids, with the
mutated codon highlighted in red.

Pairwise amino acid substitutions
---------------------------------

The same operation with ``num_mutations=2`` enumerates every possible
pair of single amino acid changes.

.. code-block:: python

    double_pool = orf_pool.mutagenize_orf(
        num_mutations=2,
        mutation_type="missense_only_first",
        codon_positions=pos,
        prefix="double",
        style="red",
        mode="sequential",
    ).named("double_pool")

    print(double_pool.num_states)  # 536085

With 55 positions and 19 amino acids each, the number of pairwise
combinations is C(55, 2) x 19\ :sup:`2` = 536,085.

Random higher-order mutants
---------------------------

For variants with three or more mutations, exhaustive enumeration is
impractical. :doc:`Random mode </operations/modes>` samples from this
space instead. Unlike ``num_mutations``, which fixes the exact number
of mutations per sequence, ``mutation_rate`` specifies a per-codon
probability, so each sequence receives a variable number of changes.
Here ``mutation_rate=0.1`` mutates each codon independently with 10%
probability, and ``num_states=10000`` controls how many random draws
to take.

.. code-block:: python

    random_pool = orf_pool.mutagenize_orf(
        mutation_rate=0.1,
        mutation_type="missense_only_first",
        codon_positions=pos,
        prefix="random",
        style="red",
        mode="random",
        num_states=10000,
    ).named("random_pool")

Wild-type replicates
--------------------

Including multiple copies of the wild-type sequence provides internal
controls for experimental normalization.
:doc:`repeat </operations/repeat>` simply duplicates the input a given
number of times.

.. code-block:: python

    wt_pool = orf_pool.repeat(times=100, prefix="wt").named("wt_pool")

Combine into a final library
-----------------------------

:doc:`stack </operations/stack>` merges the four sub-libraries into a single pool. Each
component retains its own naming prefix, so variants can be traced
back to their source.

.. code-block:: python

    dms_pool = pp.stack([single_pool, double_pool, random_pool, wt_pool])

    dms_pool.print_library(num_seqs=10, seed=42, show_name=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">dms_pool: seq_length=168, num_states=547230</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th></tr>
    <tr><td>single_0000</td><td>ATG<span class="pp-mut">TTC</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    <tr><td>single_0001</td><td>ATG<span class="pp-mut">CTG</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    <tr><td>single_0002</td><td>ATG<span class="pp-mut">ATC</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    <tr><td>single_0003</td><td>ATG<span class="pp-mut">ATG</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    <tr><td>single_0004</td><td>ATG<span class="pp-mut">GTG</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    <tr><td>single_0005</td><td>ATG<span class="pp-mut">AGC</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    <tr><td>single_0006</td><td>ATG<span class="pp-mut">CCC</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    <tr><td>single_0007</td><td>ATG<span class="pp-mut">ACC</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    <tr><td>single_0008</td><td>ATG<span class="pp-mut">GCC</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    <tr><td>single_0009</td><td>ATG<span class="pp-mut">TAC</span>TACAAGCTGATCCTGAACGGTAAGACGCTGAAAGGTGAGACGACCACCGAAGCTGTAGACGCTGCTACTGCAGAGAAGGTGTTCAAGCAGTACGCTAACGACAACGGCGTCGACGGTGAATGGACCTACGACGACGCTACCAAAACCTTCACGGTTACCGAA</td></tr>
    </table>
    <span class="pp-ellipsis">... (547,230 total)</span>
    </div>

Because ``stack`` places components in the order they are listed, the
first 1,045 states are all single mutants. The 10 variants shown here
are therefore all single amino acid substitutions at codon position 1
(Gln in the wild type). The mutated codon is highlighted in red, making
it easy to spot changes at a glance.

Translating to amino acid sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`translate </operations/translate>` converts the coding sequence to its amino acid
representation. When ``preserve_codon_styles=True`` (the default), the
red highlighting carries over from the mutated codon to the
corresponding amino acid.

.. code-block:: python

    translated = dms_pool.translate()
    translated.print_library(num_seqs=5, show_name=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">translated: seq_length=56, num_states=547230</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th></tr>
    <tr><td>single_0000</td><td>M<span class="pp-mut">F</span>YKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE</td></tr>
    <tr><td>single_0001</td><td>M<span class="pp-mut">L</span>YKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE</td></tr>
    <tr><td>single_0002</td><td>M<span class="pp-mut">I</span>YKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE</td></tr>
    <tr><td>single_0003</td><td>M<span class="pp-mut">M</span>YKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE</td></tr>
    <tr><td>single_0004</td><td>M<span class="pp-mut">V</span>YKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE</td></tr>
    </table>
    <span class="pp-ellipsis">... (547,230 total)</span>
    </div>

Design cards
~~~~~~~~~~~~

The ``cards`` parameter on ``mutagenize_orf`` records each mutation as
structured :doc:`design card </metadata/design_cards>` columns, so every
variant carries a record of what was changed:

.. code-block:: python

    df = single_pool.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 1,045 rows × 5 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th><th>position</th><th>wt_aa</th><th>mut_aa</th></tr>
    <tr><td>single_0000</td><td>ATG<span class="pp-mut">TTC</span>TAC...GAA</td><td>(1,)</td><td>(Q,)</td><td>(F,)</td></tr>
    <tr><td>single_0001</td><td>ATG<span class="pp-mut">CTG</span>TAC...GAA</td><td>(1,)</td><td>(Q,)</td><td>(L,)</td></tr>
    <tr><td>single_0002</td><td>ATG<span class="pp-mut">ATC</span>TAC...GAA</td><td>(1,)</td><td>(Q,)</td><td>(I,)</td></tr>
    <tr><td>single_0003</td><td>ATG<span class="pp-mut">ATG</span>TAC...GAA</td><td>(1,)</td><td>(Q,)</td><td>(M,)</td></tr>
    <tr><td>single_0004</td><td>ATG<span class="pp-mut">GTG</span>TAC...GAA</td><td>(1,)</td><td>(Q,)</td><td>(V,)</td></tr>
    <tr><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td></tr>
    </table>
    </div>

Each row records the codon position, wild-type amino acid, and
substituted amino acid. These columns are ready for downstream filtering
and analysis without parsing the sequences themselves.

Library composition
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 20

   * - Component
     - Mode
     - States
   * - Single mutants
     - sequential
     - 1,045
   * - Double mutants
     - sequential
     - 536,085
   * - Random mutants
     - random
     - 10,000
   * - Wild-type replicates
     - \—
     - 100
   * - **Total**
     -
     - **547,230**

See :doc:`mutagenize_orf </operations/mutagenize_orf>`,
:doc:`translate </operations/translate>`,
:doc:`repeat </operations/repeat>`,
:doc:`stack </operations/stack>`, and
:doc:`library size </operations/library_size>` for full parameter
details and how operation counts compose. To export the library as a
DataFrame or file, see ``to_df`` and ``to_file`` in :doc:`/pool`.
