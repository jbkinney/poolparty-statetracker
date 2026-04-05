insertion_multiscan
===================

Insert sequences at multiple positions simultaneously, lengthening the
output sequence by the total inserted content. Insertion sites are chosen
randomly and are guaranteed to be non-overlapping.

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
   * - ``num_insertions``
     - ``int``
     - *(required)*
     - Number of simultaneous non-overlapping insertion sites per draw.
   * - ``insertion_pools``
     - ``Pool | list[Pool]``
     - *(required)*
     - Pool(s) supplying inserted content. A single pool is reused at every
       site; a list assigns one pool per site.
   * - ``positions``
     - ``list | None``
     - ``None``
     - Allowed position sets for each insertion site. ``None`` allows any
       valid non-overlapping arrangement.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Named region or interval to restrict insertions to.
   * - ``names``
     - ``list[str] | None``
     - ``None``
     - Names for each insertion window.
   * - ``replace``
     - ``bool``
     - ``False``
     - If ``True``, replace the bases at each site instead of inserting
       between them (same behaviour as ``replacement_multiscan``).
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style for inserted content.
   * - ``insertion_mode``
     - ``str``
     - ``"ordered"``
     - ``"ordered"`` preserves the left-to-right order of positions;
       ``"unordered"`` allows any permutation.
   * - ``min_spacing``
     - ``int | None``
     - ``None``
     - Minimum gap (in bases) between insertion sites.
   * - ``max_spacing``
     - ``int | None``
     - ``None``
     - Maximum gap (in bases) between insertion sites.
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
   parameter list, see :func:`~poolparty.insertion_multiscan` in the
   :doc:`API Reference </api>`.

Examples
--------

Two simultaneous single-base insertions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert a single random base at each of two independently chosen positions.
``mode="random"`` makes each ``print_library()`` draw one stochastic
outcome (``num_states=1`` per preview).

.. code-block:: python

    wt     = pp.from_seq("ATCGATCGATCG")
    insert = pp.from_iupac("N")          # any single base
    scan   = wt.insertion_multiscan(num_insertions=2,
                                    insertion_pools=insert, mode="random",
                                    style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=14, num_states=1</em>
    ATCGAT<span class="pp-mut">G</span>CGATC<span class="pp-mut">T</span>G
    </div>

Two simultaneous 2-mer insertions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``from_iupac("NN")`` to enumerate all 16 dinucleotide insertions at
each of the two chosen positions.

.. code-block:: python

    wt     = pp.from_seq("ATCGATCGATCG")
    insert = pp.from_iupac("NN")         # all 16 dinucleotides
    scan   = wt.insertion_multiscan(num_insertions=2,
                                    insertion_pools=insert, mode="random",
                                    style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=16, num_states=1</em>
    ATCGAT<span class="pp-mut">GA</span>CGATC<span class="pp-mut">TT</span>G
    </div>

Multiscan insertion within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict both insertion sites to within the ``cre`` region; flanking bases
are never modified.

.. code-block:: python

    wt     = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    insert = pp.from_iupac("N")
    scan   = wt.insertion_multiscan(num_insertions=2,
                                    insertion_pools=insert,
                                    region="cre", mode="random",
                                    style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=18, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCG<span class="pp-mut">G</span>ATCG<span class="pp-mut">T</span><span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>

Spacing constraints (min_spacing, max_spacing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``min_spacing`` and ``max_spacing`` control the gap between insertion sites.
Here two 6-base motif insertions must be 4–8 bases apart on a 24-mer.

.. code-block:: python

    wt     = pp.from_seq("ATCGATCGATCGATCGATCGATCG")
    motif  = pp.from_seq("GATTAC")
    scan   = wt.insertion_multiscan(num_insertions=2,
                                    insertion_pools=motif,
                                    min_spacing=4, max_spacing=8,
                                    mode="sequential", style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=36, num_states=95</em>
    <span class="pp-mut">GATTAC</span>ATCG<span class="pp-mut">GATTAC</span>ATCGATCGATCGATCGATCG<br>
    <span class="pp-mut">GATTAC</span>ATCGA<span class="pp-mut">GATTAC</span>TCGATCGATCGATCGATCG<br>
    <span class="pp-mut">GATTAC</span>ATCGAT<span class="pp-mut">GATTAC</span>CGATCGATCGATCGATCG<br>
    <span class="pp-mut">GATTAC</span>ATCGATC<span class="pp-mut">GATTAC</span>GATCGATCGATCGATCG<br>
    <span class="pp-mut">GATTAC</span>ATCGATCG<span class="pp-mut">GATTAC</span>ATCGATCGATCGATCG
    <span class="pp-ellipsis">... (95 total)</span>
    </div>

PPM-based insertion pool (from_motif)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`~poolparty.from_motif` to supply a position-probability matrix
as the inserted content. Each draw samples a different 6-mer from the PPM,
producing biologically realistic variation at each insertion site.

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
    scan  = wt.insertion_multiscan(num_insertions=2,
                                   insertion_pools=motif, mode="random",
                                   num_states=5, style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=36, num_states=5</em>
    ATCGATCGATCG<span class="pp-mut">CCGTAG</span>ATCGATCGAT<span class="pp-mut">ACCCAG</span>CG<br>
    <span class="pp-mut">CCATAT</span>ATCGATCGATCGATCGATCGAT<span class="pp-mut">AGGCAT</span>CG<br>
    ATCGATCGATCGA<span class="pp-mut">CCGCAG</span>TCGATCGATC<span class="pp-mut">AAATAG</span>G<br>
    <span class="pp-mut">CCATAT</span>ATCGATCGATCGATCGA<span class="pp-mut">ACATAG</span>TCGATCG<br>
    ATCGATCGAT<span class="pp-mut">ACATAG</span>CGATCGATCG<span class="pp-mut">ACATAC</span>ATCG
    </div>

Explicit position sets (positions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify allowed insertion sites for each window, using a distinct pool for
each site. Below, the first insertion (``GGG``) can occur at position 0, 4,
or 8 and the second (``AAA``) at position 10 or 14.

.. code-block:: python

    wt    = pp.from_seq("ATCGATCGATCGATCG")
    pools = [pp.from_seq("GGG"), pp.from_seq("AAA")]
    scan  = wt.insertion_multiscan(num_insertions=2,
                                   insertion_pools=pools,
                                   positions=[[0, 4, 8], [10, 14]],
                                   mode="sequential", style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=22, num_states=6</em>
    <span class="pp-mut">GGG</span>ATCGATCGAT<span class="pp-mut">AAA</span>CGATCG<br>
    <span class="pp-mut">GGG</span>ATCGATCGATCGAT<span class="pp-mut">AAA</span>CG<br>
    ATCG<span class="pp-mut">GGG</span>ATCGAT<span class="pp-mut">AAA</span>CGATCG<br>
    ATCG<span class="pp-mut">GGG</span>ATCGATCGAT<span class="pp-mut">AAA</span>CG<br>
    ATCGATCG<span class="pp-mut">GGG</span>AT<span class="pp-mut">AAA</span>CGATCG<br>
    ATCGATCG<span class="pp-mut">GGG</span>ATCGAT<span class="pp-mut">AAA</span>CG
    </div>

See :func:`~poolparty.insertion_multiscan`.
