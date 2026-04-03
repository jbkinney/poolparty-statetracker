Design Cards
============

When sequences are generated using PoolParty, each is automatically paired with a design card. 
Design cards are data frames provide a **structured record of how each sequence was
constructed**. Each row can include columns that
report the specific changes applied by each operation — mutation positions,
substituted characters, scores, orientations, and more. Because this
information is recorded automatically during generation, downstream
analysis can filter, group, and model sequences using these columns
directly, without parsing the sequences themselves.

Design cards are opt-in: unless you pass the ``cards`` parameter, the
output contains only ``name`` and ``seq``.

All examples assume::

    import poolparty as pp
    pp.init()

----

Why Use Design Cards?
---------------------

Design cards are especially useful when the parameters that vary across a
library are themselves the object of study. For example:

- In a **deep mutational scanning** library, cards can record which amino
  acid was substituted at which position — enabling direct analysis of
  mutation effects without re-parsing codon sequences.
- In an **MPRA** library, cards can record which binding sites were inserted
  and in what order — supporting grouping and statistical testing by
  design factor.
- In **surrogate modeling** of genomic AI predictions, cards can serve
  directly as covariates in regression models, linking sequence design
  parameters to model outputs without any post-hoc feature extraction.

----

Requesting Cards
----------------

The ``cards`` parameter accepts three forms:

``None`` (default)
    No card columns in the output.

``list[str]``
    Request card keys by name. Column names are **prefixed** with the
    operation id (e.g. ``op[1]:mutagenize.positions``).

``dict[str, str]``
    Map card keys to **custom column names**. No prefix is added.

.. code-block:: python

    # List-style — column is "op[1]:score.gc"
    scored = pp.score(pool, pp.calc_gc, card_key="gc", cards=["gc"])

    # Dict-style — column is just "gc"
    scored = pp.score(pool, pp.calc_gc, card_key="gc", cards={"gc": "gc"})

Use the **dict form** when you want clean, predictable column names in
your output.

----

Universal Card Keys
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
    muts = pp.mutagenize(wt, num_mutations=1, num_states=5,
                         cards={"state": "mut_state", "seq": "mut_seq"})
    df   = muts.generate_library()
    # df has columns: name, seq, mut_state, mut_seq

----

Operation-Specific Card Keys
-----------------------------

Each operation advertises which additional keys it can produce. Requesting
an invalid key raises ``ValueError``.

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
     - ``True`` if the sequence passed the predicate, ``False`` otherwise.
   * - ``from_seqs``
     - ``seq_name``, ``seq_index``
     - Name and index of the selected input sequence.
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

Track mutation details
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    muts = pp.mutagenize(wt, num_mutations=2, num_states=5,
                         cards={"positions": "mut_pos",
                                "wt_chars": "wt",
                                "mut_chars": "mut"})
    df = muts.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">DataFrame columns: name, seq, mut_pos, wt, mut</em>
    <span class="pp-ellipsis">mut_pos=(0, 5) &nbsp; wt=('A', 'T') &nbsp; mut=('G', 'C')</span><br>
    <span class="pp-ellipsis">mut_pos=(2, 7) &nbsp; wt=('C', 'G') &nbsp; mut=('T', 'A')</span><br>
    <span class="pp-ellipsis">... (5 rows total)</span>
    </div>

Score with a clean column name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    wt     = pp.from_iupac("NNNN", mode="sequential")
    scored = pp.score(wt, pp.calc_gc, card_key="gc", cards={"gc": "gc"})
    df     = scored.generate_library()
    # df["gc"] contains the GC fraction — no "op[N]:score." prefix

Multiple cards across a pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each operation in the pipeline can export its own cards independently.

.. code-block:: python

    wt     = pp.from_iupac("NNNNNNNN", mode="sequential", num_states=10)
    scored = pp.score(wt,     pp.calc_gc,        card_key="gc",
                     cards={"gc": "gc"})
    scored = pp.score(scored, pp.calc_complexity, card_key="complexity",
                     cards={"complexity": "complexity"})
    df = scored.generate_library()
    # df has columns: name, seq, gc, complexity

Identify which pool produced each sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    pool_a = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
    pool_b = pp.from_seqs(["GGGG", "TTTT"], mode="sequential")
    combined = pp.stack([pool_a, pool_b],
                        cards={"active_parent": "source"})
    df = combined.generate_library()
    # df["source"]: 0, 0, 1, 1

DMS library with codon-level cards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a deep mutational scanning library, ``mutagenize_orf`` cards record the
amino-acid-level changes for each variant — no sequence parsing needed.

.. code-block:: python

    orf  = pp.from_seq("ATGAAATTTGGGCCCTAA")
    cds  = pp.annotate_orf(orf)
    muts = pp.mutagenize_orf(cds, num_mutations=1, mode="sequential",
                             cards={"codon_positions": "position",
                                    "wt_aas": "wt_aa",
                                    "mut_aas": "mut_aa"})
    df = muts.generate_library()
    # df has columns: name, seq, position, wt_aa, mut_aa
    # Each row records exactly which amino acid was changed and where

Cards as covariates for modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because card columns are ordinary DataFrame columns, they can be passed
directly to statistical or machine-learning models as covariates. This
pattern — using design parameters as regression features without post-hoc
sequence parsing — is especially powerful for surrogate modeling of
genomic AI predictions.

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

Disabling Cards Globally
-------------------------

To suppress all card computation for performance:

.. code-block:: python

    pp.toggle_cards(on=False)

This causes every operation to skip card computation regardless of the
``cards`` parameter. Re-enable with ``pp.toggle_cards(on=True)``.
