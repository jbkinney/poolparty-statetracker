replacement_multiscan
=====================

Place multiple non-overlapping replacement windows simultaneously at
randomly chosen positions, producing combinatorial libraries with paired
substitutions.

.. code-block:: python

    import poolparty as pp
    pp.init()

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

See :func:`~poolparty.replacement_multiscan`.

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

See :func:`~poolparty.replacement_multiscan`.

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
