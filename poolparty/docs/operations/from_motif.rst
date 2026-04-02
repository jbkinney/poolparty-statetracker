from_motif
==========

Sample sequences from a position-probability matrix (PPM), supplied as a
pandas DataFrame with base columns (A, C, G, T) and one row per position.
Each row is normalised automatically so values need not sum exactly to 1.
Sampling is always stochastic; sequential enumeration is not supported.

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
   * - ``prob_df``
     - ``DataFrame``
     - *(required)*
     - ``pandas.DataFrame`` with columns in ``{'A','C','G','T'}`` and one
       row per sequence position. Missing columns are treated as
       probability 0. Each row is normalised to sum to 1.
   * - ``pool``
     - ``Pool | None``
     - ``None``
     - Background pool. When provided with ``region``, the sampled
       sequence replaces the content of that region.
   * - ``region``
     - ``str | None``
     - ``None``
     - Region to replace in ``pool``. Required when ``pool`` is provided.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``mode``
     - ``str``
     - ``'random'``
     - Must be ``'random'``; sequential enumeration is not supported.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Number of sequences to sample. ``None`` means a single independent
       draw each call.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to every generated sequence.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

----

Examples
--------

Uniform motif
~~~~~~~~~~~~~

All positions equal probability — equivalent to sampling random sequences.

.. code-block:: python

    import pandas as pd

    pfm = pd.DataFrame(
        {"A": [0.25, 0.25], "C": [0.25, 0.25],
         "G": [0.25, 0.25], "T": [0.25, 0.25]}
    )
    pool = pp.from_motif(pfm, num_states=4)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; uniform over all dinucleotides)</em>
    AG<br>CT<br>GA<br>TT<br>
    <span class="pp-ellipsis">... (each draw is independent)</span>
    </div>

Biased motif
~~~~~~~~~~~~~

Position 0 strongly prefers A (80%), position 1 prefers C (80%).
Draws cluster near the consensus ``AC`` but vary stochastically.

.. code-block:: python

    import pandas as pd

    pfm = pd.DataFrame(
        {"A": [0.80, 0.05], "C": [0.05, 0.80],
         "G": [0.10, 0.10], "T": [0.05, 0.05]}
    )
    pool = pp.from_motif(pfm, num_states=5)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; biased toward AC; consensus = AC)</em>
    AC<br>AC<br>GC<br>AC<br>AA<br>
    <span class="pp-ellipsis">... (draws cluster near consensus)</span>
    </div>

Fixing ``num_states`` for a reproducible library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``num_states`` to pre-commit the number of draws, then fix the seed in
:func:`~poolparty.generate_library` for fully reproducible output.

.. code-block:: python

    import pandas as pd

    pfm4 = pd.DataFrame({
        "A": [0.8, 0.05, 0.05, 0.1],
        "C": [0.05, 0.8, 0.05, 0.1],
        "G": [0.1,  0.1, 0.8,  0.1],
        "T": [0.05, 0.05, 0.1, 0.7],
    })
    pool = pp.from_motif(pfm4, num_states=5)
    df   = pool.generate_library(seed=42)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (5 sequences, seed=42 &mdash; consensus ACGT)</em>
    ACGT<br>ACGT<br>ACAT<br>GCGT<br>ACGG
    </div>

Sampling into a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide ``pool`` and ``region`` to draw motif sequences into a fixed context.

.. code-block:: python

    import pandas as pd

    pfm = pd.DataFrame(
        {"A": [0.7, 0.1], "C": [0.1, 0.7],
         "G": [0.1, 0.1], "T": [0.1, 0.1]}
    )
    bg   = pp.from_seq("GCGC<insert>XX</insert>GCGC")
    pool = pp.from_motif(pfm, pool=bg, region="insert", num_states=4)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (4 stochastic draws &mdash; motif sampled into <em>insert</em> region)</em>
    GCGC<span class="pp-region">AC</span>GCGC<br>
    GCGC<span class="pp-region">AC</span>GCGC<br>
    GCGC<span class="pp-region">GC</span>GCGC<br>
    GCGC<span class="pp-region">AC</span>GCGC
    </div>

See :func:`~poolparty.from_motif`.
