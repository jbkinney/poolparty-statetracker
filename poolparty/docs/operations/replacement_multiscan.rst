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
   :widths: 20 18 12 50
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

Examples
--------

Two simultaneous replacements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Draw two non-overlapping single-base substitution positions at random per
sequence.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCGATCG")
    alt  = pp.from_seqs(["A", "C", "G", "T"])
    scan = pp.replacement_multiscan(wt, num_replacements=2,
                                    replacement_pools=alt)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 2 simultaneous non-overlapping substitutions per draw)</em>
    <span class="pp-mut">C</span>TCGA<span class="pp-mut">G</span>CATCG<br>
    A<span class="pp-mut">A</span>CGAT<span class="pp-mut">T</span>GATCG<br>
    ATCG<span class="pp-mut">C</span>TCG<span class="pp-mut">A</span>TCG<br>
    <span class="pp-ellipsis">... each draw places 2 substitutions at distinct positions</span>
    </div>

Three simultaneous replacements on a longer sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scale to three concurrent substitutions across a 16-mer background.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCGATCGATCG")
    alt  = pp.from_seqs(["A", "C", "G", "T"])
    scan = pp.replacement_multiscan(wt, num_replacements=3,
                                    replacement_pools=alt)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 3 simultaneous substitutions per draw on 16-mer)</em>
    <span class="pp-mut">G</span>TCG<span class="pp-mut">C</span>TCGAT<span class="pp-mut">T</span>GCG<br>
    ATCG<span class="pp-mut">A</span>TCG<span class="pp-mut">G</span>TCG<span class="pp-mut">C</span>TCG<br>
    A<span class="pp-mut">T</span>CGAT<span class="pp-mut">A</span>GATCG<span class="pp-mut">G</span>CG<br>
    <span class="pp-ellipsis">... each draw has 3 substitutions at non-overlapping positions</span>
    </div>

Multiscan within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict both replacement windows to the ``cre`` region; flanking bases are
never touched.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    alt  = pp.from_seqs(["A", "C", "G", "T"])
    scan = pp.replacement_multiscan(wt, num_replacements=2,
                                    replacement_pools=alt,
                                    positions=range(4, 12))

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 2 simultaneous substitutions confined to <em>cre</em>)</em>
    AAAA<span class="pp-region"><span class="pp-mut">G</span>TCGAT<span class="pp-mut">A</span>G</span>TTTT<br>
    AAAA<span class="pp-region">A<span class="pp-mut">A</span>CG<span class="pp-mut">C</span>TCG</span>TTTT<br>
    AAAA<span class="pp-region">ATCG<span class="pp-mut">C</span>T<span class="pp-mut">T</span>G</span>TTTT<br>
    <span class="pp-ellipsis">... flanks always AAAA...TTTT; 2 mutations inside cre</span>
    </div>

See :func:`~poolparty.replacement_multiscan`.
