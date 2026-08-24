.. _filter-operation:

filter
======

Keep sequences that satisfy a condition. PoolParty provides one generic
``filter`` operation plus ready-made methods for common length and DNA checks.

All examples assume:

.. code-block:: python

    import poolparty as pp
    pp.init()

.. _filtering-model:

Filtering model
---------------

A filter evaluates the clean, tag-free sequence string. A passing sequence is
left unchanged. A rejected sequence becomes a ``NullSeq`` sentinel that
propagates through downstream operations.

``NullSeq`` preserves the pool's state structure, so filtering does not silently
renumber or remove states. Choose the desired output behavior when generating a
library:

- ``discard_null_seqs=False`` keeps rejected rows with a ``None`` sequence value.
- ``discard_null_seqs=True`` omits rejected rows from the returned DataFrame.

.. note::

   When requesting a fixed number of valid sequences from a pool with a low
   acceptance rate, generation can reach ``max_iterations`` before obtaining the
   requested number. See :doc:`generate_library` for the generation controls.

Generic filtering
-----------------

Use :meth:`Pool.filter <poolparty.Pool.filter>` for a condition that is not
covered by a ready-made method. Its predicate accepts a sequence string and
returns ``True`` to keep it.

This example keeps sequences that contain no ``T``:

.. code-block:: python

    candidates = pp.from_seqs(
        ["ACGT", "TATA", "CGCG"],
        mode="sequential",
    )
    no_t = candidates.filter(lambda seq: "T" not in seq).named("no_t")
    no_t.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">no_t: seq_length=4, num_states=3</em>
    None<br>
    None<br>
    CGCG
    </div>

The functional form, :func:`poolparty.filter`, performs the same operation.

Ready-made checks
-----------------

Use a ready-made method when it expresses the intended check directly:

.. list-table::
   :widths: 28 31 41
   :header-rows: 1

   * - Need
     - Method
     - Sequence passes when
   * - Control sequence length
     - :meth:`~poolparty.Pool.filter_length`
     - Length is within the inclusive bounds.
   * - Control GC content
     - :meth:`~poolparty.DnaPool.filter_gc`
     - GC fraction is within the inclusive range.
   * - Limit homopolymer runs
     - :meth:`~poolparty.DnaPool.filter_homopolymer`
     - No single-character run exceeds ``max_length``.
   * - Require a richer short-k vocabulary
     - :meth:`~poolparty.DnaPool.filter_complexity`
     - The short-k complexity score is at least ``min_complexity``.
   * - Limit repeated triplets
     - :meth:`~poolparty.DnaPool.filter_dust`
     - The whole-sequence DUST-style score is at most ``max_score``.
   * - Exclude restriction sites
     - :meth:`~poolparty.DnaPool.filter_restriction_sites`
     - None of the selected enzyme or explicit sites is present.

``filter_length`` works with any Pool type, including DNA and protein pools.
The other five methods are DNA-specific. All six accept the shared ``name``,
``prefix``, and ``cards`` arguments and construct the same underlying filter
operation.

Length boundaries
~~~~~~~~~~~~~~~~~

At least one length bound is required. Bounds are inclusive, and equal bounds
select an exact length:

.. code-block:: python

    candidates = pp.from_seqs(
        ["ACGT", "ACGTAC", "ACGTACGT"],
        mode="sequential",
    )
    sized = candidates.filter_length(
        min_length=6,
        max_length=8,
    ).named("sized")
    sized.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">sized: seq_length=None, num_states=3</em>
    None<br>
    ACGTAC<br>
    ACGTACGT
    </div>

For exact length 8, use
``filter_length(min_length=8, max_length=8)``.

Combining DNA checks
~~~~~~~~~~~~~~~~~~~~

Ready-made methods can be chained. Here only the first candidate has the
required length and GC range, avoids a long homopolymer, and contains no EcoRI
site:

.. code-block:: python

    candidates = pp.from_seqs(
        [
            "ACGTACGTCAGT",
            "GGGGCGGCGGCG",
            "ACGTAAAAAACG",
            "ACGTGAATTCAC",
        ],
        mode="sequential",
    )

    ready = (candidates
        .filter_length(min_length=12, max_length=12)
        .filter_gc(min_gc=0.35, max_gc=0.65)
        .filter_homopolymer(max_length=4)
        .filter_restriction_sites(enzymes=["EcoRI"])
    )
    df = ready.generate_library(
        num_seqs=4,
        discard_null_seqs=True,
    )

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 1 row × 2 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th></tr>
    <tr><td>None</td><td>ACGTACGTCAGT</td></tr>
    </table>
    </div>

Complexity and DUST-style scores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two low-complexity checks overlap, but they are not interchangeable:

- ``filter_complexity`` measures how much of the possible short-k vocabulary is
  observed for the requested k values, which default to ``(1, 2, 3)``. The
  per-k ratios are averaged. IUPAC codes and gaps are counted as literal symbols.
- ``filter_dust`` measures repeated triplets across the complete sequence. It is
  DUST-style, but it does not perform DustMasker's windowing, interval selection,
  or masking. PoolParty thresholds are therefore not DustMasker ``level`` values.

Different checks can make different decisions. For example,
``ACAACCAAACCA`` has short-k complexity ``0.488`` but a DUST-style score of
``0.4``. It fails ``min_complexity=0.5`` while passing ``max_score=0.5``.

Use :meth:`~poolparty.Pool.score` when measured values are needed instead of
only a pass/fail decision:

.. code-block:: python

    candidates = pp.from_seqs(
        [
            "AAAAAAAAAAAA",
            "ACACACACACAC",
            "ACAACCAAACCA",
            "ACGTCAGTGCAT",
        ],
        mode="sequential",
    )
    measured = (candidates
        .score(pp.calc_complexity, card_key="complexity",
               cards={"complexity": "complexity"})
        .score(pp.calc_dust, card_key="dust",
               cards={"dust": "dust"})
    )
    df = measured.generate_library().round({"complexity": 3, "dust": 3})

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 4 rows × 4 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th><th>complexity</th><th>dust</th></tr>
    <tr><td>None</td><td>AAAAAAAAAAAA</td><td>0.147</td><td>4.5</td></tr>
    <tr><td>None</td><td>ACACACACACAC</td><td>0.294</td><td>2.0</td></tr>
    <tr><td>None</td><td>ACAACCAAACCA</td><td>0.488</td><td>0.4</td></tr>
    <tr><td>None</td><td>ACGTCAGTGCAT</td><td>0.939</td><td>0.0</td></tr>
    </table>
    </div>

Cards and final output
----------------------

Every filter exposes one operation-specific design-card key, ``passed``. A list
request creates a prefixed column; a dictionary can give it a stable name:

.. code-block:: python

    candidates = pp.from_seqs(
        ["AAAA", "ACGT", "GGCC"],
        mode="sequential",
    )
    checked = candidates.filter_gc(
        min_gc=0.4,
        max_gc=0.6,
        cards={"passed": "passed_gc"},
    )
    df = checked.generate_library(
        num_seqs=3,
        discard_null_seqs=False,
    )

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 3 rows × 3 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th><th>passed_gc</th></tr>
    <tr><td>None</td><td>None</td><td>False</td></tr>
    <tr><td>None</td><td>ACGT</td><td>True</td></tr>
    <tr><td>None</td><td>None</td><td>False</td></tr>
    </table>
    </div>

Passing ``discard_null_seqs=True`` would return only the ``ACGT`` row. The
``passed`` card records the decision; it does not record GC, length, complexity,
or DUST-style values. Use :meth:`~poolparty.Pool.score` for those measurements.

See also
--------

- :doc:`design cards </metadata/design_cards>` for card naming and provenance.
- :ref:`Library statistics <pool-stats>` for aggregate DUST-style summaries of
  a generated library; short-k complexity is not among its reported statistics.
- :doc:`score` for recording measured sequence properties.
- :doc:`generate_library` for null-row and iteration controls.
- :doc:`get_barcodes` for constructing barcode sets under generation-time
  constraints.
