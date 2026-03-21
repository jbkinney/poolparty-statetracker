mutagenize
==========

Introduce point mutations into every sequence in a pool. Exactly one of
``num_mutations`` or ``mutation_rate`` must be supplied. Pass ``region`` to
restrict mutagenesis to a named tagged segment; use ``allowed_chars`` to limit
which substitutions are permitted.

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
     - The Pool to mutagenize. Can also be a plain sequence string.
   * - ``num_mutations``
     - ``int | None``
     - ``None``
     - Fixed number of point mutations per draw. Mutually exclusive with
       ``mutation_rate``.
   * - ``mutation_rate``
     - ``float | None``
     - ``None``
     - Per-base probability of mutation. Each base is mutated independently
       with this probability. Mutually exclusive with ``num_mutations``.
   * - ``region``
     - ``str | None``
     - ``None``
     - Name of a tagged region to restrict mutations to. Flanks are unchanged.
   * - ``allowed_chars``
     - ``str | None``
     - ``None``
     - IUPAC string of the same length as the sequence specifying the allowed
       bases at each position. Only positions with more than one allowed base
       are mutable.
   * - ``style``
     - ``str | None``
     - ``None``
     - Named display style applied to mutated bases.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` enumerates all single-substitution variants in order;
       ``'random'`` samples each draw independently.
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

Single random mutation (num_mutations=1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each draw returns one sequence with a single substitution at a randomly chosen
position.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    mutants = pp.mutagenize(wt, num_mutations=1)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 1 random point mutation per draw)</em>
    A<span class="pp-mut">G</span>CGATCG<br>
    ATCG<span class="pp-mut">C</span>TCG<br>
    ATCGAT<span class="pp-mut">A</span>G<br>
    <span class="pp-ellipsis">... (stochastic; each draw carries one substitution)</span>
    </div>

Per-base mutation rate (mutation_rate=0.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mutation_rate`` applies an independent per-position probability; the number
of substitutions per draw follows a Binomial distribution and may be zero.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    mutants = pp.mutagenize(wt, mutation_rate=0.1)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; each base mutated independently with p=0.1)</em>
    A<span class="pp-mut">G</span>CGATCG<br>
    ATCGATCG<br>
    ATCGAT<span class="pp-mut">T</span>G<br>
    <span class="pp-ellipsis">... (number of mutations per draw follows Binomial(8, 0.1))</span>
    </div>

Mutate only within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``region`` confines all mutations to the tagged segment; flanks are returned
unchanged.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    mutants = pp.mutagenize(wt, num_mutations=1, region="cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 1 mutation per draw, restricted to <em>cre</em>)</em>
    AAAA<span class="pp-region">A<span class="pp-mut">G</span>CGATCG</span>TTTT<br>
    AAAA<span class="pp-region">ATCG<span class="pp-mut">C</span>TCG</span>TTTT<br>
    AAAA<span class="pp-region">ATCGAT<span class="pp-mut">A</span>G</span>TTTT<br>
    <span class="pp-ellipsis">... flanks AAAA and TTTT are always unchanged</span>
    </div>

Restrict substitutions with ``allowed_chars``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``allowed_chars="SSSSSSSS"`` (S = {G,C}) restricts mutations to G&harr;C
swaps at every position; no A or T substitutions are made.

.. code-block:: python

    wt      = pp.from_seq("GCGCGCGC")
    mutants = pp.mutagenize(wt, num_mutations=1, allowed_chars="SSSSSSSS")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; only G&harr;C swaps; no A or T mutations)</em>
    <span class="pp-mut">C</span>CGCGCGC<br>
    GC<span class="pp-mut">C</span>CGCGC<br>
    GCGCGC<span class="pp-mut">C</span>C<br>
    <span class="pp-ellipsis">... (stochastic; each draw swaps exactly one G&harr;C)</span>
    </div>

Sequential enumeration (mode="sequential")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode='sequential'`` with ``num_mutations=1`` enumerates every single-point
variant in deterministic order, covering all positions and non-wild-type bases.

.. code-block:: python

    wt      = pp.from_seq("ACGT")
    mutants = pp.mutagenize(wt, num_mutations=1, mode="sequential")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (9 sequences &mdash; all single-point variants in order)</em>
    <span class="pp-mut">C</span>CGT<br>
    <span class="pp-mut">G</span>CGT<br>
    <span class="pp-mut">T</span>CGT<br>
    A<span class="pp-mut">A</span>GT<br>
    A<span class="pp-mut">G</span>GT<br>
    A<span class="pp-mut">T</span>GT<br>
    AC<span class="pp-mut">A</span>T<br>
    AC<span class="pp-mut">C</span>T<br>
    <span class="pp-ellipsis">... (9 total &mdash; 4 positions &times; 3 non-wt bases each)</span>
    </div>

See :func:`~poolparty.mutagenize`.
