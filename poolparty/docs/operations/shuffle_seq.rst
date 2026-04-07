shuffle_seq
===========

Randomly permute the bases of a sequence, optionally restricting the shuffle
to a named region while leaving flanking sequence untouched. Only
``mode="random"`` is supported; each draw is an independent permutation.

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
     - The Pool to shuffle. Can also be a plain sequence string.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Restrict the shuffle to a named region or ``[start, stop]`` interval.
       Flanking sequences are returned unchanged.
   * - ``shuffle_type``
     - ``str``
     - ``"mono"``
     - ``"mono"`` randomly permutes individual bases (preserves
       mononucleotide composition). ``"dinuc"`` uses an Euler-path shuffle
       that preserves dinucleotide frequencies.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``mode``
     - ``str``
     - ``'random'``
     - Only ``'random'`` is meaningful; each draw produces an independent
       random permutation.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Number of shuffled variants to draw. ``None`` defaults to 1.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to the shuffled result.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

.. note::

   With ``shuffle_type="dinuc"``, the **first and last bases are always
   fixed** — this is a mathematical constraint of the Euler-path algorithm
   used to preserve dinucleotide frequencies.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.shuffle_seq` in the
   :doc:`API Reference </api>`.

Examples
--------

Shuffle the full sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~

With default ``num_states`` (``None`` → ``1``), each library generation
returns one random permutation of all bases. Use a larger ``num_states`` to
enumerate several shuffles in one call (see below).

.. code-block:: python

    import poolparty as pp
    pp.init()

    wt       = pp.from_seq("ATCGATCG")
    shuffled = pp.shuffle_seq(wt, mode="random")
    shuffled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">shuffled: seq_length=8, num_states=1</em>
    TGGCAACT
    <span class="pp-ellipsis">... (stochastic; sequences change each run)</span>
    </div>

Shuffle only a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only bases inside the tagged region are permuted; flanks are unchanged.

.. code-block:: python

    import poolparty as pp
    pp.init()

    wt       = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    shuffled = pp.shuffle_seq(wt, region="cre", mode="random")
    shuffled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">shuffled: seq_length=16, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>TGGCAACT<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    <span class="pp-ellipsis">... (stochastic; flanks fixed)</span>
    </div>

Dinucleotide-preserving shuffle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``shuffle_type="dinuc"`` preserves dinucleotide frequencies using an
Euler-path algorithm. The first and last bases are always fixed.

The input sequence must have **repeated dinucleotides** for the shuffle to
produce diverse outputs — highly regular sequences like ``ACGTACGT`` have
only one valid Euler path and will always return the same result.

.. code-block:: python

    import poolparty as pp
    pp.init()

    wt       = pp.from_seq("AACGAACGTTGC")
    shuffled = pp.shuffle_seq(wt, shuffle_type="dinuc", num_states=3, mode="random")
    shuffled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">shuffled: seq_length=12, num_states=3</em>
    ACGCGTTGAAAC<br>
    AAACGTTGCGAC<br>
    AACGCGTTGAAC
    </div>

Fixing the number of variants with ``num_states``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``num_states`` fixes how many shuffled variants are drawn from the pool in
one library generation call.

.. code-block:: python

    import poolparty as pp
    pp.init()

    wt       = pp.from_seq("ATCGATCG")
    shuffled = pp.shuffle_seq(wt, num_states=5, mode="random")
    shuffled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">shuffled: seq_length=8, num_states=5</em>
    TGGCAACT<br>
    AGTACGTC<br>
    GTTAAGCC<br>
    CACTTGGA<br>
    TCATGACG
    </div>

See :func:`~poolparty.shuffle_seq`.
