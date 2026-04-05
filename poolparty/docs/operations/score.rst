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

GC content with ``pp.calc_gc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Record the GC fraction of every sequence using the built-in utility.
Design cards are opt-in — pass ``cards={"gc": "gc"}`` (a dict mapping
the card key to the desired column name) to include the column in the
output. Using a list ``["gc"]`` also works but prefixes the column name
with the operation id (e.g. ``op[12]:score.gc``); the dict form avoids
this.

.. code-block:: python

    wt     = pp.from_iupac("NNNN", mode="sequential")
    scored = pp.score(wt, pp.calc_gc, card_key="gc", cards={"gc": "gc"})
    scored.print_library()
    # scored.generate_library() adds a "gc" column per sequence

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scored: seq_length=4, num_states=256</em>
    AAAA<br>
    AAAC<br>
    AAAG<br>
    AAAT<br>
    AACA<br>
    <span class="pp-ellipsis">... (256 total)</span>
    </div>

Custom scoring function
~~~~~~~~~~~~~~~~~~~~~~~~

Any callable works. Here a lambda counts A/T bases for AT richness.

.. code-block:: python

    wt     = pp.from_seqs(["AAAA", "GCGC", "ATCG"], mode="sequential")
    scored = pp.score(wt, lambda s: s.count("A") + s.count("T"),
                     card_key="at_count", cards=["at_count"])
    scored.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scored: seq_length=4, num_states=3</em>
    AAAA<br>
    GCGC<br>
    ATCG
    </div>

Built-in scoring functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

PoolParty includes several sequence property functions that work directly
with ``score``:

- ``pp.calc_gc`` — GC fraction
- ``pp.calc_complexity`` — linguistic complexity (0–1)
- ``pp.calc_dust`` — DUST low-complexity score (lower = more complex)

.. code-block:: python

    wt     = pp.from_iupac("NNNNNNNN", mode="sequential", num_states=5)
    scored = pp.score(wt, pp.calc_complexity, card_key="complexity",
                     cards={"complexity": "complexity"})
    scored.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scored: seq_length=8, num_states=5</em>
    AAAAAAAA<br>
    AAAAAAAC<br>
    AAAAAAAG<br>
    AAAAAAAT<br>
    AAAAAACA
    </div>

Score only a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~

``region`` restricts scoring to the tagged segment; the full sequence
still passes through unchanged. With ``mutagenize(..., mode="random")``,
set ``num_states`` if you want more than one independent draw (the
default is a single random mutant).

.. code-block:: python

    wt     = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    muts   = pp.mutagenize(wt, num_mutations=1, region="cre",
                          mode="random", num_states=5)
    scored = pp.score(muts, pp.calc_gc, region="cre", card_key="cre_gc",
                     cards=["cre_gc"])
    scored.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scored: seq_length=16, num_states=5</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCGGTCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCGAACG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCGCTCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>GTCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ACCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>

Multiple scores in a pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chain two ``score`` calls to record multiple metrics in one library.

.. code-block:: python

    wt     = pp.from_iupac("NNNNNNNN", mode="sequential", num_states=10)
    scored = pp.score(wt,     pp.calc_gc,        card_key="gc",         cards={"gc": "gc"})
    scored = pp.score(scored, pp.calc_complexity, card_key="complexity", cards=["complexity"])
    df     = scored.generate_library()
    print(df.to_string())
    # df has both "gc" and "op[...]:score.complexity" columns

.. raw:: html

    <table>
    <thead>
    <tr>
      <th>name</th>
      <th>seq</th>
      <th>gc</th>
      <th>op[2]:score.complexity</th>
    </tr>
    </thead>
    <tbody>
    <tr><td>None</td><td>AAAAAAAA</td><td>0.000</td><td>0.186508</td></tr>
    <tr><td>None</td><td>AAAAAAAC</td><td>0.125</td><td>0.373016</td></tr>
    <tr><td>None</td><td>AAAAAAAG</td><td>0.125</td><td>0.373016</td></tr>
    <tr><td>None</td><td>AAAAAAAT</td><td>0.000</td><td>0.373016</td></tr>
    <tr><td>None</td><td>AAAAAACA</td><td>0.125</td><td>0.476190</td></tr>
    <tr><td>None</td><td>AAAAAACC</td><td>0.250</td><td>0.476190</td></tr>
    <tr><td>None</td><td>AAAAAACG</td><td>0.250</td><td>0.559524</td></tr>
    <tr><td>None</td><td>AAAAAACT</td><td>0.125</td><td>0.559524</td></tr>
    <tr><td>None</td><td>AAAAAAGA</td><td>0.125</td><td>0.476190</td></tr>
    <tr><td>None</td><td>AAAAAAGC</td><td>0.250</td><td>0.559524</td></tr>
    </tbody>
    </table>

See :func:`~poolparty.score`.
