insertion_multiscan
===================

Insert sequences at multiple positions simultaneously, lengthening the
output sequence by the total inserted content. Insertion sites are chosen
randomly and are guaranteed to be non-overlapping.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Examples
--------

Two simultaneous single-base insertions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert a single random base at each of two independently chosen positions.

.. code-block:: python

    wt     = pp.from_seq("ATCGATCGATCG")
    insert = pp.from_iupac("N")          # any single base
    scan   = pp.insertion_multiscan(wt, num_insertions=2,
                                    insertion_pools=insert)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 2 simultaneous single-base insertions per draw)</em>
    <span class="pp-ins">A</span>ATCG<span class="pp-ins">T</span>ATCGATCG<br>
    AT<span class="pp-ins">C</span>CGATCG<span class="pp-ins">G</span>GATCG<br>
    ATCGA<span class="pp-ins">T</span>TCGAT<span class="pp-ins">A</span>CG<br>
    <span class="pp-ellipsis">... each draw inserts 1 base at each of 2 distinct positions</span>
    </div>

See :func:`~poolparty.insertion_multiscan`.

Two simultaneous 2-mer insertions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``from_iupac("NN")`` to enumerate all 16 dinucleotide insertions at
each of the two chosen positions.

.. code-block:: python

    wt     = pp.from_seq("ATCGATCGATCG")
    insert = pp.from_iupac("NN")         # all 16 dinucleotides
    scan   = pp.insertion_multiscan(wt, num_insertions=2,
                                    insertion_pools=insert)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 2 simultaneous 2-mer insertions per draw)</em>
    <span class="pp-ins">AT</span>ATCG<span class="pp-ins">GC</span>ATCGATCG<br>
    A<span class="pp-ins">CC</span>TCGATCG<span class="pp-ins">TT</span>ATCG<br>
    ATCG<span class="pp-ins">AA</span>ATCG<span class="pp-ins">GT</span>TCG<br>
    <span class="pp-ellipsis">... each draw inserts a random 2-mer at each of 2 positions</span>
    </div>

See :func:`~poolparty.insertion_multiscan`.

Multiscan insertion within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict both insertion sites to within the ``cre`` region; flanking bases
are never modified.

.. code-block:: python

    wt     = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    insert = pp.from_iupac("N")
    scan   = pp.insertion_multiscan(wt, num_insertions=2,
                                    insertion_pools=insert,
                                    positions=range(4, 13))

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 2 single-base insertions confined to <em>cre</em>)</em>
    AAAA<span class="pp-region"><span class="pp-ins">G</span>ATCG<span class="pp-ins">T</span>ATCG</span>TTTT<br>
    AAAA<span class="pp-region">A<span class="pp-ins">C</span>TCGAT<span class="pp-ins">A</span>CG</span>TTTT<br>
    AAAA<span class="pp-region">ATCG<span class="pp-ins">T</span>ATCG<span class="pp-ins">G</span></span>TTTT<br>
    <span class="pp-ellipsis">... flanks always AAAA...TTTT; insertions inside cre only</span>
    </div>

See :func:`~poolparty.insertion_multiscan`.
