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
   :widths: auto

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
     - ``Pool | str | None``
     - ``None``
     - Background pool or sequence string. When provided with ``region``,
       the sampled sequence replaces the content of that region.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Region to replace in ``pool``: a marker name or ``[start, stop]``
       interval. Required when ``pool`` is provided.
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
     - Enumeration order when combined with other pools.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to every generated sequence.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.from_motif` in the
   :doc:`API Reference </api>`.

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
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=2, num_states=4</em>
    GC<br>CC<br>GC<br>TC
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
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=2, num_states=5</em>
    AC<br>AC<br>AC<br>AC<br>TC
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
    pool.print_library()
    df   = pool.generate_library(seed=42)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=4, num_states=5</em>
    ACAA<br>ACGG<br>ACGT<br>ACGT<br>TCGT
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
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=10, num_states=4</em>
    GCGC<span class="pp-xtag-light">&lt;insert&gt;</span>AC<span class="pp-xtag-light">&lt;/insert&gt;</span>GCGC<br>
    GCGC<span class="pp-xtag-light">&lt;insert&gt;</span>AC<span class="pp-xtag-light">&lt;/insert&gt;</span>GCGC<br>
    GCGC<span class="pp-xtag-light">&lt;insert&gt;</span>CA<span class="pp-xtag-light">&lt;/insert&gt;</span>GCGC<br>
    GCGC<span class="pp-xtag-light">&lt;insert&gt;</span>AC<span class="pp-xtag-light">&lt;/insert&gt;</span>GCGC
    </div>

Pool method shorthand
~~~~~~~~~~~~~~~~~~~~~

When inserting into a region, the same operation is available as a method
on any ``DnaPool``.  The call ``bg.insert_from_motif(...)`` is equivalent
to ``pp.from_motif(..., pool=bg)`` — it simply passes ``self`` as the
background pool.

.. code-block:: python

    import pandas as pd

    pfm = pd.DataFrame(
        {"A": [0.7, 0.1], "C": [0.1, 0.7],
         "G": [0.1, 0.1], "T": [0.1, 0.1]}
    )
    bg   = pp.from_seq("GCGC<insert>XX</insert>GCGC")
    pool = bg.insert_from_motif(pfm, region="insert", num_states=4)
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=10, num_states=4</em>
    GCGC<span class="pp-xtag-light">&lt;insert&gt;</span>AC<span class="pp-xtag-light">&lt;/insert&gt;</span>GCGC<br>
    GCGC<span class="pp-xtag-light">&lt;insert&gt;</span>AC<span class="pp-xtag-light">&lt;/insert&gt;</span>GCGC<br>
    GCGC<span class="pp-xtag-light">&lt;insert&gt;</span>CA<span class="pp-xtag-light">&lt;/insert&gt;</span>GCGC<br>
    GCGC<span class="pp-xtag-light">&lt;insert&gt;</span>AC<span class="pp-xtag-light">&lt;/insert&gt;</span>GCGC
    </div>

See :func:`~poolparty.from_motif` and :meth:`~poolparty.DnaPool.insert_from_motif`.
