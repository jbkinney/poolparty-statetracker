shuffle_seq
===========

Randomly permute the bases of a sequence, optionally restricting the shuffle
to a named region while leaving flanking sequence untouched.

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
     - The Pool to shuffle. Can also be a plain sequence string.
   * - ``region``
     - ``str | None``
     - ``None``
     - Name of a tagged region to restrict the shuffle to. Flanking sequences
       are returned unchanged.
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
     - Fix how many shuffled variants are drawn per :func:`~poolparty.generate_library`
       call.
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

Examples
--------

Shuffle the full sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each draw is an independent random permutation of all bases.

.. code-block:: python

    wt       = pp.from_seq("ATCGATCG")
    shuffled = pp.shuffle_seq(wt)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; random permutation of all 8 bases each draw)</em>
    CTAGATCG<br>
    GATCATCG<br>
    TCGAATCG<br>
    <span class="pp-ellipsis">... (stochastic; each draw is a unique permutation)</span>
    </div>

Shuffle only a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only bases inside the tagged region are permuted; flanks are unchanged.

.. code-block:: python

    wt       = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    shuffled = pp.shuffle_seq(wt, region="cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; only <em>cre</em> bases shuffled; flanks unchanged)</em>
    AAAA<span class="pp-region">CGATAATC</span>TTTT<br>
    AAAA<span class="pp-region">TCAGCGAT</span>TTTT<br>
    AAAA<span class="pp-region">GATCATCG</span>TTTT<br>
    <span class="pp-ellipsis">... (AAAA and TTTT always unchanged)</span>
    </div>

Dinucleotide-preserving shuffle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``shuffle_type="dinuc"`` preserves dinucleotide frequencies using an
Euler-path algorithm. Note that the first and last bases are always fixed.

.. code-block:: python

    wt       = pp.from_seq("ACGTACGTAC")
    shuffled = pp.shuffle_seq(wt, shuffle_type="dinuc", num_states=3)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (3 stochastic draws &mdash; dinucleotide frequencies preserved)</em>
    ACGTACGTAC<br>
    ACGACGTAC<br>
    ACTACGGTAC<br>
    <span class="pp-ellipsis">... (first A and last C always fixed)</span>
    </div>

Fixing the number of variants with ``num_states``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``num_states`` fixes how many shuffled variants are drawn from the pool in
one library generation call.

.. code-block:: python

    wt       = pp.from_seq("ATCGATCG")
    shuffled = pp.shuffle_seq(wt, num_states=5)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (5 stochastic draws)</em>
    CTAGATCG<br>
    GATCTCGA<br>
    TCGATCAG<br>
    AGCTCGAT<br>
    CGATTAGC
    </div>

See :func:`~poolparty.shuffle_seq`.
