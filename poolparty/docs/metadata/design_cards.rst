Design Cards
============

PoolParty can automatically pair each generated sequence with a **design
card**, a DataFrame row that records how the sequence was constructed.
Columns report the changes applied by each operation: mutation positions,
substituted characters, scores, orientations, and more. Downstream
analysis can filter, group, and model sequences using these columns
directly, without parsing the sequences themselves.

Design cards are opt-in: unless you pass the ``cards`` parameter, the
output contains only ``name`` and ``seq``.

All examples assume:

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Why use design cards?
---------------------

Design cards are especially useful when the parameters that vary across a
library are themselves the object of study. For example:

- In a **deep mutational scanning** library, cards can record which amino
  acid was substituted at which position, enabling direct analysis of
  mutation effects without re-parsing codon sequences.
- In an **MPRA** library, cards can record which binding sites were inserted
  and in what order, supporting grouping and statistical testing by
  design factor.
- In **surrogate modeling** of genomic AI predictions, cards can serve
  directly as covariates in regression models, linking sequence design
  parameters to model outputs without any post-hoc feature extraction.

----

Requesting cards
----------------

The ``cards`` parameter accepts three forms:

``None`` (default)
    No card columns in the output.

``list[str]``
    Request card keys by name. Column names are automatically prefixed
    with the operation's index in the pipeline and its name
    (e.g., ``op[1]:mutagenize.positions``, where ``op[1]`` is the
    second operation).

``dict[str, str]``
    Map card keys to **custom column names**. No prefix is added.

.. code-block:: python

    pool = pp.from_iupac("NNNN", mode="sequential")

    # List-style — column is "op[1]:score.gc"
    scored = pool.score(pp.calc_gc, card_key="gc", cards=["gc"])

    # Dict-style — column is just "gc"
    scored = pool.score(pp.calc_gc, card_key="gc", cards={"gc": "gc"})

Use the **dict form** when you want clean, predictable column names in
your output.

----

Universal card keys
-------------------

Every operation supports two universal keys, regardless of type:

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Key
     - Value
   * - ``"seq"``
     - The output sequence string at this point in the DAG. Useful for
       recording intermediate sequences in a multi-step pipeline.
   * - ``"state"``
     - The numeric state index for this operation (0, 1, 2, ...).

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    muts = wt.mutagenize(num_mutations=1, num_states=5,
                         cards={"state": "mut_state", "seq": "mut_seq"})
    df   = muts.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 5 rows × 4 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th><th>mut_seq</th><th>mut_state</th></tr>
    <tr><td>None</td><td>CTCGATCG</td><td>CTCGATCG</td><td>0</td></tr>
    <tr><td>None</td><td>GTCGATCG</td><td>GTCGATCG</td><td>1</td></tr>
    <tr><td>None</td><td>TTCGATCG</td><td>TTCGATCG</td><td>2</td></tr>
    <tr><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td></tr>
    </table>
    </div>

----

Operation-specific card keys
----------------------------

Each operation defines which additional keys it supports. Requesting an
invalid key raises ``ValueError``.

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Operation
     - Card Keys
     - Description
   * - ``mutagenize``
     - ``positions``, ``wt_chars``, ``mut_chars``
     - Tuple of mutated positions, wild-type characters, and mutant
       characters.
   * - ``mutagenize_orf``
     - ``codon_positions``, ``wt_codons``, ``mut_codons``, ``wt_aas``,
       ``mut_aas``
     - Codon-level mutation details: positions, original/mutant codons,
       and original/mutant amino acids.
   * - ``score``
     - *(the card_key value)*
     - The score computed by the scoring function. Default key is
       ``"score"``; set ``card_key="gc"`` to use ``"gc"`` instead.
   * - ``stack``
     - ``active_parent``
     - Index (0, 1, 2, ...) of which input pool produced this sequence.
   * - ``repeat``
     - ``repeat_index``
     - Which repeat copy this sequence belongs to (0, 1, ..., times-1).
   * - ``flip``
     - ``flip``
     - ``"forward"`` or ``"rc"`` indicating the orientation.
   * - ``recombine``
     - ``breakpoints``, ``pool_assignments``
     - Breakpoint positions and which source pool contributed each
       segment.
   * - ``shuffle_seq``
     - ``permutation``
     - Tuple of the permutation applied to molecular positions.
   * - ``filter``
     - ``passed``
     - ``True`` if the sequence passed the generic predicate or ready-made
       filter check, ``False`` otherwise.
   * - ``from_seqs``
     - ``seq_name``, ``seq_index``
     - Name and index of the selected input sequence.
   * - ``from_vcf``
     - ``chrom``, ``pos``, ``ref``, ``alt``, ``allele``, ``variant_type``,
       ``variant_id``, ``filter``, ``window_start``, ``window_stop``, plus
       ``info_<KEY>`` for each requested INFO field
     - Position, alleles, and annotation of the source variant. The
       alternate-allele keys are ``None`` on a reference row.
   * - ``get_kmers``
     - ``kmer_index``, ``kmer``
     - Index and string of the generated k-mer.
   * - ``get_barcodes``
     - ``barcode_index``, ``barcode``
     - Index and string of the generated barcode.
   * - ``region_scan``
     - ``position_index``, ``start``, ``end``, ``name``, ``region_seq``
     - Scanning position details and the tagged region content.

----

Examples
--------

.. rubric:: Track mutation details

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    muts = wt.mutagenize(num_mutations=2, num_states=5,
                         cards={"positions": "mut_pos",
                                "wt_chars": "wt",
                                "mut_chars": "mut"})
    df = muts.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 5 rows × 5 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th><th>mut_pos</th><th>wt</th><th>mut</th></tr>
    <tr><td>None</td><td>GTCGACCG</td><td>(0, 5)</td><td>('A', 'T')</td><td>('G', 'C')</td></tr>
    <tr><td>None</td><td>ATCAATCG</td><td>(3, 4)</td><td>('G', 'A')</td><td>('A', 'A')</td></tr>
    <tr><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td></tr>
    </table>
    </div>

.. rubric:: Score with a clean column name

.. code-block:: python

    wt     = pp.from_iupac("NNNN", mode="sequential")
    scored = wt.score(pp.calc_gc, card_key="gc", cards={"gc": "gc"})
    df     = scored.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 256 rows × 3 columns (no "op[N]:score." prefix)</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th><th>gc</th></tr>
    <tr><td>None</td><td>AAAA</td><td>0.00</td></tr>
    <tr><td>None</td><td>AAAC</td><td>0.25</td></tr>
    <tr><td>None</td><td>AAAG</td><td>0.25</td></tr>
    <tr><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td></tr>
    </table>
    </div>

.. rubric:: Multiple cards across a pipeline

Each operation in the pipeline can export its own cards independently.

.. code-block:: python

    wt     = pp.from_iupac("NNNNNNNN", mode="sequential", num_states=10)
    scored = (wt
        .score(pp.calc_gc,        card_key="gc",         cards={"gc": "gc"})
        .score(pp.calc_complexity, card_key="complexity", cards={"complexity": "complexity"})
    )
    df = scored.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 10 rows × 4 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th><th>gc</th><th>complexity</th></tr>
    <tr><td>None</td><td>AAAAAAAA</td><td>0.00</td><td>0.19</td></tr>
    <tr><td>None</td><td>AAAAAAAC</td><td>0.12</td><td>0.37</td></tr>
    <tr><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td></tr>
    </table>
    </div>

.. rubric:: Identify which pool produced each sequence

.. code-block:: python

    pool_a = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
    pool_b = pp.from_seqs(["GGGG", "TTTT"], mode="sequential")
    combined = pp.stack([pool_a, pool_b],
                        cards={"active_parent": "source"})
    df = combined.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 4 rows × 3 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th><th>source</th></tr>
    <tr><td>None</td><td>AAAA</td><td>0</td></tr>
    <tr><td>None</td><td>CCCC</td><td>0</td></tr>
    <tr><td>None</td><td>GGGG</td><td>1</td></tr>
    <tr><td>None</td><td>TTTT</td><td>1</td></tr>
    </table>
    </div>

.. rubric:: DMS library with codon-level cards

In a deep mutational scanning library, ``mutagenize_orf`` cards record the
amino-acid-level changes for each variant, so no sequence parsing is needed.

.. code-block:: python

    orf  = pp.from_seq("ATGAAATTTGGGCCCTAA")
    muts = (orf
        .annotate_orf("gene")
        .mutagenize_orf(num_mutations=1, mode="sequential",
                        cards={"codon_positions": "position",
                               "wt_aas": "wt_aa",
                               "mut_aas": "mut_aa"})
    )
    df = muts.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 114 rows × 5 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th><th>position</th><th>wt_aa</th><th>mut_aa</th></tr>
    <tr><td>None</td><td><span class="pp-xtag-light">&lt;gene&gt;</span>TTCAAATTTGGGCCCTAA<span class="pp-xtag-light">&lt;/gene&gt;</span></td><td>(0,)</td><td>(M,)</td><td>(F,)</td></tr>
    <tr><td>None</td><td><span class="pp-xtag-light">&lt;gene&gt;</span>CTGAAATTTGGGCCCTAA<span class="pp-xtag-light">&lt;/gene&gt;</span></td><td>(0,)</td><td>(M,)</td><td>(L,)</td></tr>
    <tr><td>None</td><td><span class="pp-xtag-light">&lt;gene&gt;</span>ATCAAATTTGGGCCCTAA<span class="pp-xtag-light">&lt;/gene&gt;</span></td><td>(0,)</td><td>(M,)</td><td>(I,)</td></tr>
    <tr><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td><td><span class="pp-ellipsis">...</span></td></tr>
    </table>
    </div>

.. rubric:: Cards as covariates for modeling

Card columns are ordinary DataFrame columns, so they can be used directly
as covariates in statistical or machine-learning models. This avoids
post-hoc sequence parsing: the design parameters are already structured
as regression features.

.. code-block:: python

    # Pseudocode: score a library with a model, then regress on card features
    df = library.generate_library()
    df["model_score"] = predict_with_model(df["seq"])

    # Card columns become covariates
    import statsmodels.api as sm
    X = df[["position", "strength"]]  # from design cards
    y = df["model_score"]
    model = sm.OLS(y, sm.add_constant(X)).fit()

----

Disabling cards globally
------------------------

To suppress all card computation for performance:

.. code-block:: python

    pp.toggle_cards(on=False)

This causes every operation to skip card computation regardless of the
``cards`` parameter. Re-enable with ``pp.toggle_cards(on=True)``.
