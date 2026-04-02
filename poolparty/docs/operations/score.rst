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
   :widths: 20 18 12 50

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
    df     = scored.generate_library()
    # df["gc"] contains the GC fraction for each sequence

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (256 sequences, unchanged &mdash; "gc" column added to library output)</em>
    AAAA &nbsp;<em style="color:#6b7280;">gc=0.00</em><br>
    AAAC &nbsp;<em style="color:#6b7280;">gc=0.25</em><br>
    AAAG &nbsp;<em style="color:#6b7280;">gc=0.25</em><br>
    <span class="pp-ellipsis">... (256 total)</span>
    </div>

Custom scoring function
~~~~~~~~~~~~~~~~~~~~~~~~

Any callable works. Here a lambda counts A/T bases for AT richness.

.. code-block:: python

    wt     = pp.from_seqs(["AAAA", "GCGC", "ATCG"], mode="sequential")
    scored = pp.score(wt, lambda s: s.count("A") + s.count("T"),
                     card_key="at_count", cards=["at_count"])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (3 sequences, unchanged &mdash; "at_count" column added)</em>
    AAAA &nbsp;<em style="color:#6b7280;">at_count=4</em><br>
    GCGC &nbsp;<em style="color:#6b7280;">at_count=0</em><br>
    ATCG &nbsp;<em style="color:#6b7280;">at_count=2</em>
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
    df     = scored.generate_library()
    # df["complexity"] contains the linguistic complexity for each sequence

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (5 sequences, unchanged &mdash; "complexity" column added)</em>
    AAAAAAAA &nbsp;<em style="color:#6b7280;">complexity=0.187</em><br>
    AAAAAAAC &nbsp;<em style="color:#6b7280;">complexity=0.373</em><br>
    AAAAAAAG &nbsp;<em style="color:#6b7280;">complexity=0.373</em><br>
    AAAAAAAT &nbsp;<em style="color:#6b7280;">complexity=0.373</em><br>
    AAAAAACA &nbsp;<em style="color:#6b7280;">complexity=0.476</em>
    </div>

Score only a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~

``region`` restricts scoring to the tagged segment; the full sequence
still passes through unchanged.

.. code-block:: python

    wt     = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    muts   = pp.mutagenize(wt, num_mutations=1, region="cre")
    scored = pp.score(muts, pp.calc_gc, region="cre", card_key="cre_gc",
                     cards=["cre_gc"])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; GC of <em>cre</em> region scored; full sequence unchanged)</em>
    AAAA<span class="pp-region">A<span class="pp-mut">G</span>CGATCG</span>TTTT &nbsp;<em style="color:#6b7280;">cre_gc≈0.63</em><br>
    <span class="pp-ellipsis">... (GC computed from region content only)</span>
    </div>

Multiple scores in a pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chain two ``score`` calls to record multiple metrics in one library.

.. code-block:: python

    wt     = pp.from_iupac("NNNNNNNN", mode="sequential", num_states=10)
    scored = pp.score(wt,     pp.calc_gc,        card_key="gc",         cards={"gc": "gc"})
    scored = pp.score(scored, pp.calc_complexity, card_key="complexity", cards=["complexity"])
    df     = scored.generate_library()
    # df has both "gc" and "complexity" columns

.. raw:: html

    <table style="border-collapse:collapse;font-size:0.85em;font-family:monospace;margin:0.5em 0">
    <thead><tr style="border-bottom:1px solid #d1d5db">
      <th style="padding:4px 12px 4px 4px;text-align:right;color:#6b7280">&nbsp;</th>
      <th style="padding:4px 12px;text-align:left">seq</th>
      <th style="padding:4px 12px;text-align:right">gc</th>
      <th style="padding:4px 12px;text-align:right">op[42]:score.complexity</th>
    </tr></thead>
    <tbody>
    <tr><td style="padding:2px 12px 2px 4px;color:#6b7280;text-align:right">0</td><td style="padding:2px 12px">AAAAAAAA</td><td style="padding:2px 12px;text-align:right">0.000</td><td style="padding:2px 12px;text-align:right">0.186508</td></tr>
    <tr><td style="padding:2px 12px 2px 4px;color:#6b7280;text-align:right">1</td><td style="padding:2px 12px">AAAAAAAC</td><td style="padding:2px 12px;text-align:right">0.125</td><td style="padding:2px 12px;text-align:right">0.373016</td></tr>
    <tr><td style="padding:2px 12px 2px 4px;color:#6b7280;text-align:right">2</td><td style="padding:2px 12px">AAAAAAAG</td><td style="padding:2px 12px;text-align:right">0.125</td><td style="padding:2px 12px;text-align:right">0.373016</td></tr>
    <tr><td style="padding:2px 12px 2px 4px;color:#6b7280;text-align:right">3</td><td style="padding:2px 12px">AAAAAAAT</td><td style="padding:2px 12px;text-align:right">0.000</td><td style="padding:2px 12px;text-align:right">0.373016</td></tr>
    <tr><td style="padding:2px 12px 2px 4px;color:#6b7280;text-align:right">4</td><td style="padding:2px 12px">AAAAAACA</td><td style="padding:2px 12px;text-align:right">0.125</td><td style="padding:2px 12px;text-align:right">0.476190</td></tr>
    <tr><td style="padding:2px 12px 2px 4px;color:#6b7280;text-align:right">5</td><td style="padding:2px 12px">AAAAAACC</td><td style="padding:2px 12px;text-align:right">0.250</td><td style="padding:2px 12px;text-align:right">0.476190</td></tr>
    <tr><td style="padding:2px 12px 2px 4px;color:#6b7280;text-align:right">6</td><td style="padding:2px 12px">AAAAAACG</td><td style="padding:2px 12px;text-align:right">0.250</td><td style="padding:2px 12px;text-align:right">0.559524</td></tr>
    <tr><td style="padding:2px 12px 2px 4px;color:#6b7280;text-align:right">7</td><td style="padding:2px 12px">AAAAAACT</td><td style="padding:2px 12px;text-align:right">0.125</td><td style="padding:2px 12px;text-align:right">0.559524</td></tr>
    <tr><td style="padding:2px 12px 2px 4px;color:#6b7280;text-align:right">8</td><td style="padding:2px 12px">AAAAAAGA</td><td style="padding:2px 12px;text-align:right">0.125</td><td style="padding:2px 12px;text-align:right">0.476190</td></tr>
    <tr><td style="padding:2px 12px 2px 4px;color:#6b7280;text-align:right">9</td><td style="padding:2px 12px">AAAAAAGC</td><td style="padding:2px 12px;text-align:right">0.250</td><td style="padding:2px 12px;text-align:right">0.559524</td></tr>
    </tbody></table>

See :func:`~poolparty.score`.
