replacement_multiscan
=====================

Place multiple non-overlapping replacement windows simultaneously at
randomly chosen positions, producing combinatorial libraries with paired
substitutions.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Parameters
----------

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pool``
     - ``Pool | str``
     - *(required)*
     - Input pool or sequence string.
   * - ``num_replacements``
     - ``int``
     - *(required)*
     - Number of simultaneous non-overlapping replacement windows per draw.
   * - ``replacement_pools``
     - ``Pool | list[Pool]``
     - *(required)*
     - Pool(s) supplying replacement content. A single pool is reused at
       every site; a list assigns one pool per site.
   * - ``positions``
     - ``list | None``
     - ``None``
     - Allowed position sets for each replacement window. ``None`` allows
       any valid non-overlapping arrangement.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Named region or interval to restrict replacements to.
   * - ``names``
     - ``list[str] | None``
     - ``None``
     - Names for each replacement window.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style for replaced content.
   * - ``insertion_mode``
     - ``str``
     - ``"ordered"``
     - ``"ordered"`` preserves left-to-right order of positions;
       ``"unordered"`` allows any permutation.
   * - ``min_spacing``
     - ``int | None``
     - ``None``
     - Minimum gap (in bases) between replacement windows.
   * - ``max_spacing``
     - ``int | None``
     - ``None``
     - Maximum gap (in bases) between replacement windows.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.
   * - ``mode``
     - ``str``
     - ``"random"``
     - ``"random"`` or ``"sequential"``.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Number of states. ``None`` lets PoolParty choose automatically.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Iteration priority for downstream multi-pool iteration.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.replacement_multiscan` in the
   :doc:`API Reference </api>`.

Examples
--------

Two simultaneous replacements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Draw two non-overlapping single-base substitution positions at random per
sequence.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCGATCG")
    alt  = pp.from_seqs(["A", "C", "G", "T"], mode="sequential")
    scan = wt.replacement_multiscan(num_replacements=2,
                                    replacement_pools=alt, mode="random",
                                    style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=12, num_states=16</em>
    ATCGAT<span class="pp-mut">A</span>GAT<span class="pp-mut">A</span>G<br>
    <span class="pp-mut">A</span>TCGATCGATC<span class="pp-mut">C</span><br>
    ATCGAT<span class="pp-mut">A</span>GATC<span class="pp-mut">G</span><br>
    <span class="pp-mut">A</span>TCGATCG<span class="pp-mut">T</span>TCG<br>
    ATCGA<span class="pp-mut">C</span>CGA<span class="pp-mut">A</span>CG
    <span class="pp-ellipsis">... (16 total)</span>
    </div>

Two simultaneous degenerate motif replacements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace with a degenerate 6-base IUPAC motif (``R`` = A|G, ``Y`` = C|T) at
two random positions on a 24-mer. Each draw samples independently from the
64 possible ``RRYYYY`` sequences.

.. code-block:: python

    wt    = pp.from_seq("ATCGATCGATCGATCGATCGATCG")
    motif = pp.from_iupac("RRYYYY")
    scan  = wt.replacement_multiscan(num_replacements=2,
                                     replacement_pools=motif, mode="random",
                                     style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=24, num_states=1</em>
    ATCGATCGA<span class="pp-mut">GACCCT</span>GA<span class="pp-mut">GGTTTC</span>G
    </div>

Multiscan within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict both replacement windows to the ``cre`` region; flanking bases are
never touched.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCGATCG</cre>TTTT")
    alt  = pp.from_seqs(["A", "C", "G", "T"], mode="sequential")
    scan = wt.replacement_multiscan(num_replacements=2,
                                    replacement_pools=alt,
                                    region="cre", mode="random",
                                    style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=20, num_states=16</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCGAT<span class="pp-mut">A</span>GAT<span class="pp-mut">A</span>G<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><span class="pp-mut">A</span>TCGATCGATC<span class="pp-mut">C</span><span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCGAT<span class="pp-mut">A</span>GATC<span class="pp-mut">G</span><span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><span class="pp-mut">A</span>TCGATCG<span class="pp-mut">T</span>TCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCGA<span class="pp-mut">C</span>CGA<span class="pp-mut">A</span>CG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    <span class="pp-ellipsis">... (16 total)</span>
    </div>

Spacing constraints (min_spacing, max_spacing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``min_spacing`` and ``max_spacing`` control the gap between replacement
windows. Here two 6-base motif replacements must be 3–6 bases apart on a
24-mer.

.. code-block:: python

    wt    = pp.from_seq("ATCGATCGATCGATCGATCGATCG")
    motif = pp.from_seq("GATTAC")
    scan  = wt.replacement_multiscan(num_replacements=2,
                                     replacement_pools=motif,
                                     min_spacing=3, max_spacing=6,
                                     mode="sequential", style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=24, num_states=34</em>
    <span class="pp-mut">GATTAC</span>CGA<span class="pp-mut">GATTAC</span>GATCGATCG<br>
    <span class="pp-mut">GATTAC</span>CGAT<span class="pp-mut">GATTAC</span>ATCGATCG<br>
    <span class="pp-mut">GATTAC</span>CGATC<span class="pp-mut">GATTAC</span>TCGATCG<br>
    <span class="pp-mut">GATTAC</span>CGATCG<span class="pp-mut">GATTAC</span>CGATCG<br>
    A<span class="pp-mut">GATTAC</span>GAT<span class="pp-mut">GATTAC</span>ATCGATCG
    <span class="pp-ellipsis">... (34 total)</span>
    </div>

PPM-based replacement pool (from_motif)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`~poolparty.from_motif` to supply a position-probability matrix
as the replacement content. Each draw samples a different 6-mer from the
PPM, producing biologically realistic variation at each window.

.. code-block:: python

    import pandas as pd

    pfm = pd.DataFrame({
        "A": [0.8, 0.1, 0.5, 0.1, 0.7, 0.1],
        "C": [0.1, 0.7, 0.2, 0.1, 0.1, 0.1],
        "G": [0.05, 0.1, 0.2, 0.1, 0.1, 0.7],
        "T": [0.05, 0.1, 0.1, 0.7, 0.1, 0.1],
    })
    wt    = pp.from_seq("ATCGATCGATCGATCGATCGATCG")
    motif = pp.from_motif(pfm)
    scan  = wt.replacement_multiscan(num_replacements=2,
                                     replacement_pools=motif, mode="random",
                                     num_states=5, style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=24, num_states=5</em>
    ATCGATCGA<span class="pp-mut">CCGTAG</span>GA<span class="pp-mut">ACCCAG</span>G<br>
    <span class="pp-mut">CCATAT</span>CGATCGATCGA<span class="pp-mut">AGGCAT</span>G<br>
    ATCGATCGAT<span class="pp-mut">CCGCAG</span>A<span class="pp-mut">AAATAG</span>G<br>
    <span class="pp-mut">CCATAT</span>CGATCGA<span class="pp-mut">ACATAG</span>GATCG<br>
    ATCGATCG<span class="pp-mut">ACATAG</span>C<span class="pp-mut">ACATAC</span>TCG
    </div>

Explicit position sets (positions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify allowed starts for each window independently, using a distinct pool
for each site. Below, the first replacement (``GGG``) can start at position
0, 3, or 6, while the second (``AAA``) can start at position 9 or 13.

.. code-block:: python

    wt    = pp.from_seq("ATCGATCGATCGATCG")
    pools = [pp.from_seq("GGG"), pp.from_seq("AAA")]
    scan  = wt.replacement_multiscan(num_replacements=2,
                                     replacement_pools=pools,
                                     positions=[[0, 3, 6], [9, 13]],
                                     mode="sequential", style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=16, num_states=6</em>
    <span class="pp-mut">GGG</span>GATCGA<span class="pp-mut">AAA</span>ATCG<br>
    <span class="pp-mut">GGG</span>GATCGATCGA<span class="pp-mut">AAA</span><br>
    ATC<span class="pp-mut">GGG</span>CGA<span class="pp-mut">AAA</span>ATCG<br>
    ATC<span class="pp-mut">GGG</span>CGATCGA<span class="pp-mut">AAA</span><br>
    ATCGAT<span class="pp-mut">GGGAAA</span>ATCG<br>
    ATCGAT<span class="pp-mut">GGG</span>TCGA<span class="pp-mut">AAA</span>
    </div>

See :func:`~poolparty.replacement_multiscan`.
