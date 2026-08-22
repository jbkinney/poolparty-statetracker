score
=====

Evaluate a user-supplied function on each sequence and record the result
as a design card column. The sequence passes through unchanged — ``score``
is a passthrough operation that adds metadata without altering content.
The function receives the clean (tag-stripped) sequence string, or the
clean content of a named region when ``region`` is specified.

Compatible with built-in utilities such as ``pp.calc_gc``,
``pp.calc_dust``, and ``pp.calc_complexity``.

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
     - The Pool or sequence string to score.
   * - ``fn``
     - ``callable``
     - *(required)*
     - Scoring function ``(str) -> any``. Receives a clean (tag-free)
       sequence string and returns any scalar value to record.
   * - ``card_key``
     - ``str``
     - ``'score'``
     - Design card column name under which the result is stored.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Region to score. A named tag (str), ``[start, stop]`` interval, or
       ``None`` to score the full sequence.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``cards``
     - ``list | dict | None``
     - ``None``
     - Design card keys to include. The available key is the value of
       ``card_key`` (default ``'score'``).

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.score` in the
   :doc:`API Reference </api>`.

Examples
--------

Custom scoring function
~~~~~~~~~~~~~~~~~~~~~~~~

The scoring function takes a sequence string and returns any scalar value.
Define it as a regular function so the pattern is explicit.

.. code-block:: python

    def gc_fraction(seq):
        return (seq.count("G") + seq.count("C")) / len(seq)

    pool   = pp.from_seqs(["AAAA", "ACGT", "GCGC", "CCCC", "ATAT"],
                          mode="sequential")
    scored = pp.score(pool, gc_fraction, card_key="gc",
                      cards={"gc": "gc"})
    df     = scored.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 5 rows × 3 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th><th>gc</th></tr>
    <tr><td>None</td><td>AAAA</td><td>0.0</td></tr>
    <tr><td>None</td><td>ACGT</td><td>0.5</td></tr>
    <tr><td>None</td><td>GCGC</td><td>1.0</td></tr>
    <tr><td>None</td><td>CCCC</td><td>1.0</td></tr>
    <tr><td>None</td><td>ATAT</td><td>0.0</td></tr>
    </table>
    </div>

The ``cards`` parameter controls how the card column is named in the
output. A dict ``{"gc": "gc"}`` maps the card key directly to the column
name. A list ``["gc"]`` also works but prefixes the column with the
operation id (e.g., ``op[1]:score.gc``); use the dict form to keep column
names clean.

Built-in scoring functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

PoolParty includes several scoring functions that match the same
``(str) -> scalar`` pattern:

- ``pp.calc_gc`` — GC fraction
- ``pp.calc_complexity`` — short-k linguistic complexity (0–1)
- ``pp.calc_dust`` — DUST-style whole-sequence triplet repetition
  (lower = less repetitive)

.. code-block:: python

    pool   = pp.from_seqs(["AAAA", "ACGT", "GCGC", "CCCC", "ATAT"],
                          mode="sequential")
    scored = pp.score(pool, pp.calc_gc, card_key="gc",
                      cards={"gc": "gc"})
    df     = scored.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 5 rows × 3 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th><th>gc</th></tr>
    <tr><td>None</td><td>AAAA</td><td>0.0</td></tr>
    <tr><td>None</td><td>ACGT</td><td>0.5</td></tr>
    <tr><td>None</td><td>GCGC</td><td>1.0</td></tr>
    <tr><td>None</td><td>CCCC</td><td>1.0</td></tr>
    <tr><td>None</td><td>ATAT</td><td>0.0</td></tr>
    </table>
    </div>

Score only a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~

``region`` restricts scoring to the tagged segment; the full sequence
passes through unchanged.

.. code-block:: python

    wt     = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    muts   = pp.mutagenize(wt, num_mutations=1, region="cre",
                          mode="random", num_states=5)
    scored = pp.score(muts, pp.calc_gc, region="cre", card_key="cre_gc",
                     cards={"cre_gc": "cre_gc"})
    df     = scored.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 5 rows × 3 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th><th>cre_gc</th></tr>
    <tr><td>None</td><td>AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ATCGGTCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT</td><td>0.625</td></tr>
    <tr><td>None</td><td>AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ATCGAACG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT</td><td>0.500</td></tr>
    <tr><td>None</td><td>AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ATCGCTCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT</td><td>0.625</td></tr>
    <tr><td>None</td><td>AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>GTCGATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT</td><td>0.625</td></tr>
    <tr><td>None</td><td>AAAA<span class="pp-xtag-light">&lt;cre&gt;</span>ACCGATCG<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT</td><td>0.625</td></tr>
    </table>
    </div>

Multiple scores in a pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chain two ``score`` calls to record multiple metrics in one library.

.. code-block:: python

    pool   = pp.from_seqs(["AAAA", "ACGT", "GCGC", "CCCC", "ATAT"],
                          mode="sequential")
    scored = pp.score(pool,   pp.calc_gc,        card_key="gc",
                      cards={"gc": "gc"})
    scored = pp.score(scored, pp.calc_complexity, card_key="complexity",
                      cards={"complexity": "complexity"})
    df     = scored.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 5 rows × 4 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th><th>gc</th><th>complexity</th></tr>
    <tr><td>None</td><td>AAAA</td><td>0.00</td><td>0.36</td></tr>
    <tr><td>None</td><td>ACGT</td><td>0.50</td><td>1.00</td></tr>
    <tr><td>None</td><td>GCGC</td><td>1.00</td><td>0.72</td></tr>
    <tr><td>None</td><td>CCCC</td><td>1.00</td><td>0.36</td></tr>
    <tr><td>None</td><td>ATAT</td><td>0.00</td><td>0.72</td></tr>
    </table>
    </div>

See :func:`~poolparty.score`.
