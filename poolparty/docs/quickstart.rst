Quickstart Guide
================

The examples below walk through the main ideas behind PoolParty, from
creating a single pool to composing a complete combinatorial library.

Installation
------------

.. code-block:: bash

    pip install poolparty

Getting started
---------------

.. code-block:: python

    import poolparty as pp
    pp.init()

``pp.init()`` initializes a fresh design context. Call it before each
independent library design. For scoped contexts (e.g., inside a function
that builds one library), use ``with pp.Party()`` instead.
See :doc:`pool` for details.

----

Core concepts
-------------

**Pools.** A Pool represents a designed collection of DNA sequences. Pools
are *lazy*: they record the rules for generating sequences but delay actual
generation until the user requests it. Pools are also *immutable*: every
operation returns a new Pool, leaving the original unchanged. This means you
can branch a pipeline at any point without interference.

**Operations.** An Operation takes one or more Pools as input and produces a
new Pool as output. By chaining operations, you build a directed acyclic
graph (DAG) that specifies your library design. PoolParty provides over 20
built-in operations in four categories (source, transformation, composition,
and state). See :doc:`operations/index` for the full catalog.

**Modes.** Most operations accept a ``mode`` parameter that controls how
outputs are produced:

- ``sequential`` -- enumerate every possibility deterministically
- ``random`` -- sample from the design space
- ``fixed`` -- output is uniquely determined by the input (no variation)

See :doc:`operations/modes` for details.

These three ideas underlie every PoolParty pipeline. The sections below
show how they work in practice.

----

Creating a pool
---------------

All pools originate from a *source operation*. Source operations do not
require an existing pool as input. The simplest is ``from_seq``, which wraps
a single DNA sequence:

.. code-block:: python

    wt = pp.from_seq("ATCGATCG")
    wt.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool[0]: seq_length=8, num_states=1</em>
    ATCGATCG
    </div>

This pool has ``num_states=1`` -- the number of distinct sequences the
pool can produce -- because it contains exactly one sequence.
``seq_length`` reports the length of every sequence in the pool.

Other source operations include ``from_seqs`` (multiple sequences),
``from_iupac`` (degenerate IUPAC codes), ``from_fasta`` (FASTA files),
``get_kmers`` (all k-mers of a given length), and ``get_barcodes``
(constrained barcodes). See :doc:`operations/source_operations`.

----

Applying an operation
---------------------

Operations transform pools. Each operation returns a new pool; the original
is unchanged. Here, ``mutagenize`` in sequential mode generates every
single-nucleotide substitution:

.. code-block:: python

    mutants = wt.mutagenize(num_mutations=1, mode="sequential")
    mutants.print_library(num_seqs=6)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool[1]: seq_length=8, num_states=24</em>
    CTCGATCG<br>
    GTCGATCG<br>
    TTCGATCG<br>
    AACGATCG<br>
    ACCGATCG<br>
    AGCGATCG
    <span class="pp-ellipsis">... (24 total)</span>
    </div>

``mode="sequential"`` enumerates all 24 single-nucleotide substitutions:
8 positions times 3 alternative bases. The original ``wt`` pool is still a
single-sequence pool -- ``mutagenize`` returned a new pool.

With ``mode="random"``, the operation would draw a single random mutant
instead; passing ``num_states=N`` in random mode draws N random designs.
See :doc:`operations/modes`.

Operations can be called as standalone functions or as methods on a Pool:
``wt.mutagenize(...)`` and ``pp.mutagenize(wt, ...)`` are equivalent.
See :doc:`operations/index` for the full catalog.

----

Working with pools
------------------

A few API patterns make PoolParty pipelines easier to write and debug.

**Method chaining.** Since every operation returns a new Pool, calls can be
chained left-to-right into a pipeline:

.. code-block:: python

    library = (
        pp.from_seq("ATCGATCG")
        .mutagenize(num_mutations=1, mode="sequential")
        .named("mutants")
        .print_library(num_seqs=3)
        .repeat(times=2)
    )
    print(library.num_states)  # 48

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mutants: seq_length=8, num_states=24</em>
    CTCGATCG<br>
    GTCGATCG<br>
    TTCGATCG
    <span class="pp-ellipsis">... (24 total)</span>
    </div>

- ``.named("mutants")`` labels the pool for display (appears in
  ``print_library`` headers and ``print_dag`` output). This is distinct from
  ``prefix``, which labels *sequence names* in the output DataFrame
  (see :doc:`metadata/naming`).
- ``.print_library(num_seqs=3)`` previews 3 sequences mid-chain, then
  returns the pool unchanged so the chain continues.
- The final pool has ``num_states=48``: 24 mutants times 2 repeats.

**Branching.** Because pools are immutable, you can apply different
operations to the same input without interference:

.. code-block:: python

    branch_a = wt.mutagenize(num_mutations=1, mode="sequential")
    branch_b = wt.deletion_scan(deletion_length=2, mode="sequential")
    # wt is unchanged; branch_a and branch_b are independent

This branching pattern is how you build multi-component libraries (as in
the capstone example below).

**Inspecting a pool.** At any point you can check ``pool.num_states``,
``pool.seq_length``, and ``pool.regions``. Call ``pool.print_dag()`` to
visualize the full pipeline structure (demonstrated in the capstone).

**Reproducibility.** Pass ``seed=42`` to ``print_library``,
``generate_library``, or ``to_df`` for reproducible output across runs.

----

Sequence regions
----------------

You often want to perform different operations on different parts of a
sequence. Regions let you mark specific segments with XML-style tags so
that operations can target them by name:

.. code-block:: python

    template = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    cre_mutants = template.mutagenize(
        num_mutations=1, region="cre", mode="sequential"
    ).named("cre_mutants")
    cre_mutants.print_library(num_seqs=4)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">cre_mutants: seq_length=16, num_states=24</em>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>CTCGATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>GTCGATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>TTCGATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>AACGATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT
    <span class="pp-ellipsis">... (24 total)</span>
    </div>

Only the 8 bases inside ``<cre>`` are mutated; the flanking ``AAAA`` and
``TTTT`` remain unchanged. Tags persist through the DAG, so multiple
operations can target the same region in series. PoolParty also supports
self-closing tags (e.g., ``<ins/>``) for zero-length insertion points.

See :doc:`regions` for full tag syntax, persistence rules, and
programmatic tag insertion.

----

Scanning operations
-------------------

Scanning operations systematically tile a window across a sequence (or a
region), producing one variant per position. They are the workhorse for
saturation-style screens:

.. code-block:: python

    dels = template.deletion_scan(
        deletion_length=3, region="cre", mode="sequential"
    ).named("dels")
    dels.print_library(num_seqs=4)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">dels: seq_length=16, num_states=6</em>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>---GATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>A---ATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>AT---TCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ATC---CG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT
    <span class="pp-ellipsis">... (6 total)</span>
    </div>

``deletion_scan`` slides a 3-bp window across the 8-bp region, yielding
8 - 3 + 1 = 6 variants (one per valid window position). The scan is
restricted to ``<cre>``, so flanking sequences remain intact.

Other scanning operations include ``insertion_scan``, ``replacement_scan``,
``mutagenize_scan``, and their multi-window variants
(``insertion_multiscan``, etc.). See :doc:`operations/scanning`.

----

Combining pools
---------------

Composition operations combine sequences from multiple pools. The two
primary operations are ``stack`` (merge state spaces) and ``join``
(concatenate sequences end-to-end).

``stack`` merges the mutagenesis and deletion pools:

.. code-block:: python

    combined = pp.stack([cre_mutants, dels])
    print(combined.num_states)  # 30 (24 + 6)

``repeat`` duplicates a pool's sequences for replication:

.. code-block:: python

    wt_copies = template.repeat(times=5)
    print(wt_copies.num_states)  # 5

``stack`` produces a pool whose state space is the union of all inputs
(24 + 6 = 30). ``repeat`` produces N copies of each input sequence
(multiplicative). Another key composition operation is ``join``, which
concatenates sequences from different pools end-to-end (Cartesian product
of state spaces).

See :doc:`operations/composition_operations`. For how state counts compose
across different operation types, see :doc:`operations/library_size`.

----

Sequence metadata
-----------------

PoolParty automatically tracks how each sequence was constructed through
three complementary mechanisms:

- **Names.** Each sequence receives a dot-separated name summarizing its
  construction history (e.g., ``mut_03.rep_1``). Users assign labels via the
  ``prefix`` parameter on operations.
- **Design cards.** Structured DataFrame columns that record every design
  choice -- mutation positions, substituted characters, scores, orientations
  -- ready for filtering, grouping, and statistical modeling.
- **Styling.** Per-character color and formatting annotations that highlight
  mutations, deletions, and regions in ``print_library`` output for quick
  visual auditing.

The capstone example below demonstrates all three: names via ``prefix``,
styling via ``style``, and design cards via ``cards``.
See :doc:`metadata/naming`, :doc:`metadata/design_cards`, and
:doc:`metadata/styling`.

----

Generating libraries
--------------------

``print_library()`` previews sequences in the terminal. To produce a
``pandas.DataFrame``, use ``generate_library()``:

.. code-block:: python

    df = combined.generate_library()

The DataFrame contains a ``name`` column, a ``seq`` column, and any design
card columns requested via the ``cards`` parameter. For larger libraries
(above ~10k sequences), use ``to_df`` (chunked streaming) or ``to_file``
(stream directly to CSV, FASTA, or JSONL). See :doc:`pool` for full export
options.

----

Putting it all together
-----------------------

The following example combines every concept from the preceding sections
into a complete pipeline. A template
sequence contains a ``<tag>`` region targeted for both mutagenesis and
deletion scanning. We start a fresh session to build the example from
scratch.

.. image:: /_static/images/figure1c.drawio.svg
   :width: 100%
   :alt: Example PoolParty workflow combining mutagenesis and deletion scanning into a single library.

.. code-block:: python

    pp.init()

    template = pp.from_seq("TCCGACT<tag>GCA</tag>ATTCGGA").named("template")

    mut_pool = template.mutagenize(
        num_mutations=1,
        region="tag",
        style="red bold",
        prefix="mut",
        mode="sequential",
        cards={"positions": "mut_pos", "wt_chars": "wt", "mut_chars": "mut"},
    ).named("mut_pool")

    del_pool = template.deletion_scan(
        deletion_length=1,
        region="tag",
        style="green bold",
        prefix="del",
        mode="sequential",
        cards={"start": "del_start"},
    ).repeat(times=2, prefix="rep",
        cards={"repeat_index": "rep_idx"},
    ).named("del_pool")

    pool_final = pp.stack([mut_pool, del_pool]).named("pool_final")
    pool_final.print_library(show_name=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool_final: seq_length=17, num_states=15</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th></tr>
    <tr><td>mut_0</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span><span class="pp-mut">A</span>CA<span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    <tr><td>mut_1</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span><span class="pp-mut">C</span>CA<span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    <tr><td>mut_2</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span><span class="pp-mut">T</span>CA<span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    <tr><td>mut_3</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span>G<span class="pp-mut">A</span>A<span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    <tr><td>mut_4</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span>G<span class="pp-mut">G</span>A<span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    <tr><td>mut_5</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span>G<span class="pp-mut">T</span>A<span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    <tr><td>mut_6</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span>GC<span class="pp-mut">C</span><span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    <tr><td>mut_7</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span>GC<span class="pp-mut">G</span><span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    <tr><td>mut_8</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span>GC<span class="pp-mut">T</span><span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    <tr><td>del_0.rep_0</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span><span class="pp-style-green">-</span>CA<span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    <tr><td>del_0.rep_1</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span><span class="pp-style-green">-</span>CA<span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    <tr><td>del_1.rep_0</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span>G<span class="pp-style-green">-</span>A<span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    <tr><td>del_1.rep_1</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span>G<span class="pp-style-green">-</span>A<span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    <tr><td>del_2.rep_0</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span>GC<span class="pp-style-green">-</span><span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    <tr><td>del_2.rep_1</td><td>TCCGACT<span class="pp-xtag-light">&lt;tag&gt;</span>GC<span class="pp-style-green">-</span><span class="pp-xtag-light">&lt;/tag&gt;</span>ATTCGGA</td></tr>
    </table>
    </div>

This pipeline combines every concept from the preceding sections: a source
operation creates the template, two transformation operations
(``mutagenize`` and ``deletion_scan``) branch from it targeting the same
``<tag>`` region, ``repeat`` replicates one branch, and ``stack`` merges
them into the final library. The ``prefix`` parameter labels each variant
type so names are self-documenting (``mut_0``, ``del_0.rep_0``, etc.).
The ``style`` parameter applies color annotations visible in the output:
mutations in red, deletions in green.

The DAG view confirms the pipeline structure:

.. code-block:: python

    pool_final.print_dag()

.. code-block:: text

    pool_final (pool, n=15)
    └── op[6]:stack [mode=sequential, n=2]
        ├── mut_pool (pool, n=9)
        │   └── op[1]:mutagenize [mode=sequential, n=9]
        │       └── template (pool, n=1)
        │           └── op[0]:from_seq [mode=fixed, n=1]
        └── del_pool (pool, n=6)
            └── op[5]:repeat [mode=sequential, n=2]
                └── pool[4] (pool, n=3)
                    └── op[4]:deletion_scan(replace_region) [mode=fixed, n=1]
                        ├── pool[2] (pool, n=3)
                        │   └── op[2]:deletion_scan(region_scan) [mode=sequential, n=3]
                        │       └── template (pool, n=1)
                        │           └── op[0]:from_seq [mode=fixed, n=1]
                        └── pool[3] (pool, n=1)
                            └── op[3]:deletion_scan(from_seq) [mode=fixed, n=1]

Each node shows its mode and internal state count. Named pools
(``template``, ``mut_pool``, ``del_pool``, ``pool_final``) appear with
their labels; unnamed intermediate pools use default identifiers.

Because each operation was called with ``cards=``, the exported DataFrame
includes design card columns alongside ``name`` and ``seq``:

.. code-block:: python

    df = pool_final.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 15 rows x 7 columns (seq omitted for clarity)</em>
    <table class="pp-df">
    <tr><th>name</th><th>mut_pos</th><th>wt</th><th>mut</th><th>del_start</th><th>rep_idx</th></tr>
    <tr><td>mut_0</td><td>(0,)</td><td>(G,)</td><td>(A,)</td><td>None</td><td>None</td></tr>
    <tr><td>mut_1</td><td>(0,)</td><td>(G,)</td><td>(C,)</td><td>None</td><td>None</td></tr>
    <tr><td>mut_2</td><td>(0,)</td><td>(G,)</td><td>(T,)</td><td>None</td><td>None</td></tr>
    <tr><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td></tr>
    <tr><td>del_0.rep_0</td><td>None</td><td>None</td><td>None</td><td>0</td><td>0</td></tr>
    <tr><td>del_0.rep_1</td><td>None</td><td>None</td><td>None</td><td>0</td><td>1</td></tr>
    <tr><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td></tr>
    </table>
    </div>

Each operation contributes its own card columns: ``mutagenize`` records
the mutation position, wild-type base, and substituted base;
``deletion_scan`` records the deletion start position; and ``repeat`` records
the copy index. Sequences that did not pass through a given operation
have ``None`` in its columns.
Each operation defines its own set of available card keys.
See :doc:`metadata/design_cards`.

----

Next steps
----------

- Walk through complete real-world library designs in the
  :doc:`tutorials/index` (deep mutational scanning, MPRA)
- Browse the :doc:`operations/index` for the full operation catalog
- See :doc:`pool` for Pool properties, export methods (``to_df``,
  ``to_file``), and context management
