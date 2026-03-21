mutagenize_scan
===============

Slide a mutagenesis window of fixed length across the sequence (or a named
region) and, at each position, apply random point mutations within that window.
Bases outside the window are returned unchanged, enabling systematic scanning
for position-specific mutational sensitivity.

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
     - The Pool to scan. Can also be a plain sequence string.
   * - ``mutagenize_length``
     - ``int``
     - *(required)*
     - Width of the mutagenesis window in bases. A sequence of length *L*
       produces *L* - ``mutagenize_length`` + 1 window positions.
   * - ``num_mutations``
     - ``int | None``
     - ``None``
     - Fixed number of point mutations introduced per window draw. Mutually
       exclusive with ``mutation_rate``.
   * - ``mutation_rate``
     - ``float | None``
     - ``None``
     - Per-base probability of mutation within the window. Each base is
       independently mutated with this probability. Mutually exclusive with
       ``num_mutations``.
   * - ``positions``
     - ``list[int] | None``
     - ``None``
     - Explicit list of window start positions. ``None`` = all valid positions.
   * - ``region``
     - ``str | None``
     - ``None``
     - Name of a tagged region to restrict the scan to. Flanks are never
       modified.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` iterates positions left-to-right; ``'random'``
       shuffles.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Fix the total number of output states.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.

----

Examples
--------

1 mutation per 3-base window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A length-3 window with one fixed mutation slides across 6 positions.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    scan = wt.mutagenize_scan(mutagenize_length=3, num_mutations=1)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 6 window positions, 1 mutation per 3-base window)</em>
    <span class="pp-mut">T</span>CGTACGT<br>
    A<span class="pp-mut">T</span>GTACGT<br>
    AC<span class="pp-mut">A</span>TACGT<br>
    ACG<span class="pp-mut">G</span>ACGT<br>
    <span class="pp-ellipsis">... (6 positions, each stochastic)</span>
    </div>

2 mutations per 4-base window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``num_mutations=2`` introduces two substitutions per window. The 5-window
scan (8 - 4 + 1 = 5) produces 5 stochastic variants.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    scan = wt.mutagenize_scan(mutagenize_length=4, num_mutations=2)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 5 window positions, 2 mutations per 4-base window)</em>
    <span class="pp-mut">T</span>C<span class="pp-mut">A</span>TACGT<br>
    A<span class="pp-mut">A</span>G<span class="pp-mut">G</span>ACGT<br>
    AC<span class="pp-mut">C</span>T<span class="pp-mut">C</span>CGT<br>
    <span class="pp-ellipsis">... (5 positions, each stochastic)</span>
    </div>

Per-base mutation rate (mutation_rate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mutation_rate=0.5`` mutates each base in the window independently with
50% probability; the number of mutations per draw varies.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    scan = wt.mutagenize_scan(mutagenize_length=4, mutation_rate=0.5)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 5 window positions; each base independently mutated at 50% rate)</em>
    <span class="pp-mut">T</span>CGT<span class="pp-ellipsis" style="font-style:normal;color:inherit;">A</span>CGT<br>
    A<span class="pp-mut">T</span>G<span class="pp-mut">C</span>ACGT<br>
    AC<span class="pp-mut">A</span>TACGT<br>
    ACGT<span class="pp-mut">T</span>CGT<br>
    <span class="pp-ellipsis">... (5 positions; mutation count varies per draw)</span>
    </div>

Mutagenize scan within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict the scan to the ``cre`` region. Flanking sequences are always
unchanged.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    scan = wt.mutagenize_scan(mutagenize_length=3, num_mutations=1,
                              region="cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 6 window positions within <em>cre</em>; flanks unchanged)</em>
    AAAA<span class="pp-region"><span class="pp-mut">G</span>TCGATCG</span>TTTT<br>
    AAAA<span class="pp-region">A<span class="pp-mut">A</span>CGATCG</span>TTTT<br>
    AAAA<span class="pp-region">AT<span class="pp-mut">A</span>GATCG</span>TTTT<br>
    <span class="pp-ellipsis">... (6 positions; AAAA and TTTT always unchanged)</span>
    </div>

Non-overlapping tiles (explicit positions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``positions=[0, 3, 6]`` to mutagenize only non-overlapping 3-base tiles.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    scan = wt.mutagenize_scan(mutagenize_length=3, num_mutations=1,
                              positions=[0, 3, 6])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 3 non-overlapping tiles, 1 mutation per window)</em>
    <span class="pp-mut">T</span>CGTACGT<br>
    ACG<span class="pp-mut">G</span>ACGT<br>
    ACGTAC<span class="pp-mut">A</span>T
    </div>

See :func:`~poolparty.mutagenize_scan`.
