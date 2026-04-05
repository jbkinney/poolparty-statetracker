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
   :widths: auto

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
   * - ``style``
     - ``str | None``
     - ``None``
     - Named display style applied to mutated bases (e.g., ``'red'``,
       ``'blue bold'``).
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``mode``
     - ``str | tuple``
     - ``'random'``
     - ``'sequential'`` or ``'random'``. A scalar is broadcast to both
       the scan (position) and mutagenize (mutation) sub-operations.
       A 2-tuple ``(scan_mode, mut_mode)`` controls each independently.
   * - ``num_states``
     - ``int | tuple | None``
     - ``None``
     - Number of output states. A scalar is broadcast to both
       sub-operations. A 2-tuple ``(scan_states, mut_states)`` sets
       each independently; total states = product of the two.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.mutagenize_scan` in the
   :doc:`API Reference </api>`.

Examples
--------

Unlike :func:`~poolparty.mutagenize`, which mutates across the *entire*
sequence, ``mutagenize_scan`` confines mutations to a fixed-width **window**
that slides along the sequence. This makes it a two-dimensional operation:
one dimension controls *where* the window sits (position), and the other
controls *what* mutations occur inside it. Both ``mode`` and ``num_states``
accept 2-tuples ``(position_value, mutation_value)`` for independent control
of each dimension.

Full sequential enumeration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode=("sequential", "sequential")`` systematically enumerates every
window position and every single-base variant within each window. A 3-base
window over an 8-mer with ``num_mutations=1`` yields 6 positions ×
(3 bases × 3 non-wild-type substitutions) = 54 total states.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    scan = wt.mutagenize_scan(mutagenize_length=3, num_mutations=1,
                              mode=("sequential", "sequential"),
                              style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=8, num_states=54</em>
    <span class="pp-mut">C</span>CGTACGT<br>
    <span class="pp-mut">G</span>CGTACGT<br>
    <span class="pp-mut">T</span>CGTACGT<br>
    A<span class="pp-mut">A</span>GTACGT<br>
    A<span class="pp-mut">G</span>GTACGT
    <span class="pp-ellipsis">... (54 total)</span>
    </div>

Random scan with multiple draws
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode=("random", "random")`` samples both window position and mutation
randomly. ``num_states=(5, 1)`` draws 5 random positions with one random
mutation at each, for 5 total states.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGTACGT")
    scan = wt.mutagenize_scan(mutagenize_length=4, num_mutations=1,
                              mode=("random", "random"),
                              num_states=(5, 1), style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=12, num_states=5</em>
    ACGTACG<span class="pp-mut">C</span>ACGT<br>
    ACGTACGTA<span class="pp-mut">T</span>GT<br>
    ACGT<span class="pp-mut">C</span>CGTACGT<br>
    A<span class="pp-mut">G</span>GTACGTACGT<br>
    AC<span class="pp-mut">C</span>TACGTACGT
    </div>

Multiple mutations per window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``num_mutations=2`` introduces two substitutions, both confined to the
4-base window. Bases outside the window are always unchanged — this is the
key difference from :func:`~poolparty.mutagenize`, which can place
mutations anywhere in the sequence.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGTACGT")
    scan = wt.mutagenize_scan(mutagenize_length=4, num_mutations=2,
                              mode=("random", "random"),
                              num_states=(5, 1), style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=12, num_states=5</em>
    ACGT<span class="pp-mut">G</span>C<span class="pp-mut">A</span>TACGT<br>
    ACGTACGT<span class="pp-mut">CT</span>GT<br>
    ACGT<span class="pp-mut">G</span>CG<span class="pp-mut">A</span>ACGT<br>
    A<span class="pp-mut">A</span>G<span class="pp-mut">C</span>ACGTACGT<br>
    AC<span class="pp-mut">A</span>T<span class="pp-mut">T</span>CGTACGT
    </div>

Deep scan: random positions, all mutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode=("random", "sequential")`` picks random window positions but
enumerates every single-base variant at each — useful for deeply profiling
a few candidate regions. ``num_states=(3, None)`` selects 3 random
positions; ``None`` lets the mutation dimension auto-enumerate all variants
(12 per position → 36 total).

.. code-block:: python

    wt   = pp.from_seq("ACGTACGTACGT")
    scan = wt.mutagenize_scan(mutagenize_length=4, num_mutations=1,
                              mode=("random", "sequential"),
                              num_states=(3, None), style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=12, num_states=36</em>
    ACGT<span class="pp-mut">C</span>CGTACGT<br>
    ACGT<span class="pp-mut">G</span>CGTACGT<br>
    ACGT<span class="pp-mut">T</span>CGTACGT<br>
    ACGTA<span class="pp-mut">A</span>GTACGT<br>
    ACGTA<span class="pp-mut">G</span>GTACGT
    <span class="pp-ellipsis">... (36 total)</span>
    </div>

Broad scan: all positions, sampled mutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode=("sequential", "random")`` visits every window position
systematically but draws only a few random mutations at each — a
cost-effective way to scan the entire sequence. ``num_states=(None, 3)``
auto-enumerates all 9 positions and samples 3 mutations per window
(27 total).

.. code-block:: python

    wt   = pp.from_seq("ACGTACGTACGT")
    scan = wt.mutagenize_scan(mutagenize_length=4, num_mutations=1,
                              mode=("sequential", "random"),
                              num_states=(None, 3), style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=12, num_states=27</em>
    ACG<span class="pp-mut">C</span>ACGTACGT<br>
    ACG<span class="pp-mut">G</span>ACGTACGT<br>
    <span class="pp-mut">C</span>CGTACGTACGT<br>
    ACGT<span class="pp-mut">G</span>CGTACGT<br>
    ACGT<span class="pp-mut">T</span>CGTACGT
    <span class="pp-ellipsis">... (27 total)</span>
    </div>

Per-base mutation rate
~~~~~~~~~~~~~~~~~~~~~~~~

``mutation_rate=0.5`` mutates each base in the window independently with
50 % probability; the number of mutations per draw varies (including zero).
``num_states=(5, 1)`` samples 5 random window positions with one stochastic
mutation draw per position.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGTACGT")
    scan = wt.mutagenize_scan(mutagenize_length=4, mutation_rate=0.5,
                              mode=("random", "random"),
                              num_states=(5, 1), style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=12, num_states=5</em>
    ACGTA<span class="pp-mut">G</span>GTACGT<br>
    ACGTAC<span class="pp-mut">A</span>TA<span class="pp-mut">A</span>GT<br>
    ACGTACGTACGT<br>
    AC<span class="pp-mut">C</span>T<span class="pp-mut">C</span>CGTACGT<br>
    AC<span class="pp-mut">AAC</span>CGTACGT
    </div>

Scan within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict the scan to the ``cre`` region; flanking sequences are never
modified. Here ``mode=("sequential", "random")`` visits every position
within the region and draws 3 random mutations at each.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCGATCG</cre>TTTT")
    scan = wt.mutagenize_scan(mutagenize_length=4, num_mutations=1,
                              region="cre",
                              mode=("sequential", "random"),
                              num_states=(None, 3), style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=20, num_states=27</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATC<span class="pp-mut">C</span>ATCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATC<span class="pp-mut">T</span>ATCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><span class="pp-mut">C</span>TCGATCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCG<span class="pp-mut">G</span>TCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCG<span class="pp-mut">T</span>TCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    <span class="pp-ellipsis">... (27 total)</span>
    </div>

See :func:`~poolparty.mutagenize_scan`.
