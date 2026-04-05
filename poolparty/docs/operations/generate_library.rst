:orphan:

generate_library
================

Evaluate a pool pipeline and return the resulting sequences as a
``pandas.DataFrame`` with ``name`` and ``seq`` columns (or a plain ``list``
when ``seqs_only=True``). This is a *terminal* operation: it triggers all
upstream computation and produces concrete output.  Randomized upstream
operations (for example ``mutagenize(..., mode="random")``) should set
``mode`` explicitly so draws match the intent of the example.

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
     - ``Pool | DnaPool | ProteinPool``
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
     - Random seed for reproducible output (see examples).
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

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.generate_library` in the
   :doc:`API Reference </api>`.

Examples
--------

Basic usage: generate sequences from a scan pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build a mutagenized pool and call ``generate_library`` to collect the output
into a DataFrame.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    pool = pp.mutagenize(wt, num_mutations=1, mode="random")
    df   = pp.generate_library(pool, num_seqs=5)
    print(df.to_string())

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">print(df.to_string()) &mdash; 5 rows</em>
    <table style="border-collapse:collapse;font-size:0.9em;width:auto;margin-top:4px;">
    <tr>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;"></th>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">name</th>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">seq</th>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">0</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">ATCGGTCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">1</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">ATCGAACG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">2</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">ATCGCTCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">3</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">GTCGATCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">4</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">ACCGATCG</td>
    </tr>
    </table>
    </div>

Controlling output size with ``num_seqs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``num_seqs=`` to generate an exact number of sequences regardless of
the pool's state-space size.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    pool = pp.mutagenize(wt, num_mutations=1, mode="random")
    df   = pp.generate_library(pool, num_seqs=3)
    print(len(df))
    print(df.to_string())

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">len(df) and print(df.to_string())</em>
    <pre style="margin:4px 0 8px 0;font-size:0.9em;">3</pre>
    <table style="border-collapse:collapse;font-size:0.9em;width:auto;margin-top:4px;">
    <tr>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;"></th>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">name</th>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">seq</th>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">0</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">ATCGGTCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">1</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">ATCGAACG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">2</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">ATCGCTCG</td>
    </tr>
    </table>
    </div>

Reproducible output with ``seed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``seed=`` to fix the per-row draw for a given pool.  The same ``seed``
and the same pool object yield the same rows within one session.  After
``pp.init()``, rebuilding the pipeline and calling with the same ``seed``
matches a fresh interpreter run (operation IDs enter the internal seed
sequence).

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    pool = pp.mutagenize(wt, num_mutations=1, mode="random")
    df   = pp.generate_library(pool, num_seqs=3, seed=42)
    print(df.to_string())

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">print(df.to_string()) with seed=42</em>
    <table style="border-collapse:collapse;font-size:0.9em;width:auto;margin-top:4px;">
    <tr>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;"></th>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">name</th>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">seq</th>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">0</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">ATCGAACG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">1</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">ACCGATCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">2</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">ATCGATCT</td>
    </tr>
    </table>
    </div>

Get a plain list with ``seqs_only=True``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When only the sequence strings are needed (e.g. to pass directly to another
function), set ``seqs_only=True`` to skip DataFrame construction.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    pool = pp.mutagenize(wt, num_mutations=1, mode="random")
    seqs = pp.generate_library(pool, num_seqs=4, seed=7, seqs_only=True)
    print(seqs)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">print(seqs) with seed=7, seqs_only=True</em>
    <pre style="margin:4px 0 0 0;font-size:0.9em;">['ATCGATAG', 'GTCGATCG', 'ATCGAGCG', 'ATCGCTCG']</pre>
    </div>

Chain a full pipeline: mutagenize &rarr; filter &rarr; generate_library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compose multiple operations and materialise the result in a single call.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    mutants = pp.mutagenize(wt, num_mutations=1, mode="random")
    singles = pp.filter(
        mutants,
        lambda s: sum(a != b for a, b in zip(s, "ATCGATCG")) == 1,
    )
    df      = pp.generate_library(singles, num_seqs=5, seed=0, discard_null_seqs=True)
    print(df.to_string())

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">print(df.to_string()) with seed=0</em>
    <table style="border-collapse:collapse;font-size:0.9em;width:auto;margin-top:4px;">
    <tr>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;"></th>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">name</th>
      <th style="border-bottom:1px solid #ccc;padding:2px 14px 2px 0;font-weight:600;">seq</th>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">0</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">ATCGGTCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">1</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">ATCGAACG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">2</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">ATCGCTCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">3</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">GTCGATCG</td>
    </tr>
    <tr>
      <td style="padding:2px 14px 2px 0;">4</td>
      <td style="padding:2px 14px 2px 0;">None</td>
      <td style="padding:2px 14px 2px 0;">ACCGATCG</td>
    </tr>
    </table>
    </div>

See :func:`~poolparty.generate_library`.
