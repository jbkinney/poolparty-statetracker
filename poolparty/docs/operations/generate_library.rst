generate_library
================

Evaluate a pool pipeline and return the resulting sequences as a
``pandas.DataFrame`` with ``name`` and ``seq`` columns (or a plain ``list``
when ``seqs_only=True``). This is a *terminal* operation: it triggers all
upstream computation and produces concrete output.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Parameters
----------

.. list-table::
   :widths: 20 18 12 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pool``
     - ``Pool | str``
     - *(required)*
     - Pool to evaluate.
   * - ``num_cycles``
     - ``int``
     - ``1``
     - Number of complete cycles through the state space. Each cycle
       visits every state exactly once.
   * - ``num_seqs``
     - ``int | None``
     - ``None``
     - Exact number of sequences to generate. Overrides ``num_cycles``
       when provided.
   * - ``seed``
     - ``int | None``
     - ``None``
     - Random seed for reproducible output.
   * - ``init_state``
     - ``int | None``
     - ``None``
     - Starting state index. ``None`` begins from state 0.
   * - ``seqs_only``
     - ``bool``
     - ``False``
     - If ``True``, return a plain ``list[str]`` instead of a DataFrame.
   * - ``discard_null_seqs``
     - ``bool``
     - ``False``
     - If ``True``, skip sequences that were filtered out (``NullSeq``).
   * - ``max_iterations``
     - ``int | None``
     - ``None``
     - Maximum iterations before stopping (useful with filters that reject
       most draws).
   * - ``min_acceptance_rate``
     - ``float | None``
     - ``None``
     - If the acceptance rate drops below this threshold, generation stops
       early.
   * - ``attempts_per_rate_assessment``
     - ``int``
     - ``100``
     - Number of draws between acceptance-rate checks.

----

Examples
--------

Basic usage: generate sequences from a scan pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build a scan pool and call ``generate_library`` to collect the output into a
DataFrame.

.. code-block:: python

    wt  = pp.from_seq("ATCGATCG")
    pool = pp.mutagenize(wt, num_mutations=1)
    df  = pp.generate_library(pool, num_seqs=5)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">DataFrame output (stochastic &mdash; representative draws)</em>
    <table style="border-collapse:collapse;font-size:0.9em;width:auto;margin-top:4px;">
    <tr>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">name</th>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">seq</th>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">mutagenize.0001</td>
      <td style="padding:2px 14px 2px 0;">A<span class="pp-mut">G</span>CGATCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">mutagenize.0002</td>
      <td style="padding:2px 14px 2px 0;">ATCG<span class="pp-mut">C</span>TCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">mutagenize.0003</td>
      <td style="padding:2px 14px 2px 0;">ATCGAT<span class="pp-mut">A</span>G</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">mutagenize.0004</td>
      <td style="padding:2px 14px 2px 0;"><span class="pp-mut">G</span>TCGATCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">mutagenize.0005</td>
      <td style="padding:2px 14px 2px 0;">AT<span class="pp-mut">T</span>GATCG</td>
    </tr>
    </table>
    </div>

Controlling output size with ``num_seqs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``num_seqs=`` to generate an exact number of sequences regardless of
the pool's state-space size.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    pool = pp.mutagenize(wt, num_mutations=1)
    df   = pp.generate_library(pool, num_seqs=3)
    print(len(df))  # 3

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">DataFrame output (3 rows &mdash; stochastic)</em>
    <table style="border-collapse:collapse;font-size:0.9em;width:auto;margin-top:4px;">
    <tr>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">name</th>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">seq</th>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">mutagenize.0001</td>
      <td style="padding:2px 14px 2px 0;">A<span class="pp-mut">C</span>CGATCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">mutagenize.0002</td>
      <td style="padding:2px 14px 2px 0;">ATCG<span class="pp-mut">T</span>TCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">mutagenize.0003</td>
      <td style="padding:2px 14px 2px 0;">ATCGAT<span class="pp-mut">C</span>G</td>
    </tr>
    </table>
    </div>

Reproducible output with ``seed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``seed=`` to fix the random number generator; calling with the same seed
always produces identical sequences.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    pool = pp.mutagenize(wt, num_mutations=1)

    df1  = pp.generate_library(pool, num_seqs=3, seed=42)
    df2  = pp.generate_library(pool, num_seqs=3, seed=42)
    assert list(df1["seq"]) == list(df2["seq"])  # always True

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Both calls produce identical rows (seed=42)</em>
    <table style="border-collapse:collapse;font-size:0.9em;width:auto;margin-top:4px;">
    <tr>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">name</th>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">seq</th>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">mutagenize.0001</td>
      <td style="padding:2px 14px 2px 0;">ATCG<span class="pp-mut">G</span>TCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">mutagenize.0002</td>
      <td style="padding:2px 14px 2px 0;"><span class="pp-mut">C</span>TCGATCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">mutagenize.0003</td>
      <td style="padding:2px 14px 2px 0;">ATCGAT<span class="pp-mut">G</span>G</td>
    </tr>
    </table>
    </div>

Get a plain list with ``seqs_only=True``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When only the sequence strings are needed (e.g. to pass directly to another
function), set ``seqs_only=True`` to skip DataFrame construction.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    pool = pp.mutagenize(wt, num_mutations=1)
    seqs = pp.generate_library(pool, num_seqs=4, seed=7, seqs_only=True)
    # seqs is a plain Python list of strings
    print(seqs)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Plain list output (seqs_only=True &mdash; stochastic, seed=7)</em>
    ['A<span class="pp-mut">G</span>CGATCG',
     'ATCG<span class="pp-mut">C</span>TCG',
     'AT<span class="pp-mut">A</span>GATCG',
     '<span class="pp-mut">T</span>TCGATCG']
    </div>

Chain a full pipeline: mutagenize &rarr; filter &rarr; generate_library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compose multiple operations and materialise the result in a single call.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    mutants = pp.mutagenize(wt, num_mutations=1)
    singles = pp.filter(
        mutants,
        lambda s: sum(a != b for a, b in zip(s, "ATCGATCG")) == 1,
    )
    df      = pp.generate_library(singles, num_seqs=5, seed=0, discard_null_seqs=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">DataFrame output (stochastic &mdash; single-mutant sequences only)</em>
    <table style="border-collapse:collapse;font-size:0.9em;width:auto;margin-top:4px;">
    <tr>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">name</th>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">seq</th>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">filter.0001</td>
      <td style="padding:2px 14px 2px 0;"><span class="pp-mut">G</span>TCGATCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">filter.0002</td>
      <td style="padding:2px 14px 2px 0;">A<span class="pp-mut">G</span>CGATCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">filter.0003</td>
      <td style="padding:2px 14px 2px 0;">AT<span class="pp-mut">A</span>GATCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">filter.0004</td>
      <td style="padding:2px 14px 2px 0;">ATCG<span class="pp-mut">C</span>TCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">filter.0005</td>
      <td style="padding:2px 14px 2px 0;">ATCGAT<span class="pp-mut">T</span>G</td>
    </tr>
    </table>
    </div>

See :func:`~poolparty.generate_library`.
