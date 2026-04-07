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
   :widths: auto

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
     - Region to restrict mutations to: a tag name (``str``), an explicit
       ``[start, stop]`` interval, or ``None`` for the full sequence.
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
     - ``'sequential'`` enumerates mutation variants in order (requires
       ``num_mutations``); ``'random'`` samples each draw independently.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Number of output states. ``None`` auto-computes in sequential mode
       or defaults to 1 in random mode.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.mutagenize` in the
   :doc:`API Reference </api>`.

Examples
--------

Single random mutation (num_mutations=1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each draw returns one sequence with a single substitution at a randomly chosen
position.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    mutants = wt.mutagenize(num_mutations=1, mode="random", style="red")
    mutants.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mutants: seq_length=8, num_states=1</em>
    ATCG<span class="pp-mut">G</span>TCG
    </div>

Multiple independent mutants with ``num_states``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``num_states`` to draw multiple independent single-mutant sequences in
one ``generate_library`` call.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    mutants = wt.mutagenize(num_mutations=1, num_states=5, mode="random", style="red")
    mutants.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mutants: seq_length=8, num_states=5</em>
    ATCG<span class="pp-mut">T</span>TCG<br>
    ATCGATC<span class="pp-mut">T</span><br>
    ATCG<span class="pp-mut">C</span>TCG<br>
    ATCGA<span class="pp-mut">A</span>CG<br>
    ATCGAT<span class="pp-mut">G</span>G
    </div>

Per-base mutation rate (mutation_rate=0.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mutation_rate`` applies an independent per-position probability; the number
of substitutions per draw follows a Binomial distribution and may be zero.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    mutants = wt.mutagenize(mutation_rate=0.1, mode="random", style="red")
    mutants.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mutants: seq_length=8, num_states=1</em>
    A<span class="pp-mut">C</span>CGATCG
    </div>

Mutate only within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``region`` confines all mutations to the tagged segment; flanks are returned
unchanged. With ``mode='sequential'``, every single-base variant within the
region is enumerated.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    mutants = wt.mutagenize(num_mutations=1, region="cre", mode="sequential",
                            style="red")
    mutants.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mutants: seq_length=16, num_states=24</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><span class="pp-mut">C</span>TCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><span class="pp-mut">G</span>TCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><span class="pp-mut">T</span>TCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>A<span class="pp-mut">A</span>CGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>A<span class="pp-mut">C</span>CGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    <span class="pp-ellipsis">... (24 total)</span>
    </div>

Restrict substitutions with ``allowed_chars``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``allowed_chars="SSSSSSSS"`` (S = {G,C}) restricts mutations to G&harr;C
swaps at every position; no A or T substitutions are made.
``mode='sequential'`` enumerates every allowed swap.

.. code-block:: python

    wt      = pp.from_seq("GCGCGCGC")
    mutants = wt.mutagenize(num_mutations=1, allowed_chars="SSSSSSSS",
                            mode="sequential", style="red")
    mutants.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mutants: seq_length=8, num_states=8</em>
    <span class="pp-mut">C</span>CGCGCGC<br>
    G<span class="pp-mut">G</span>GCGCGC<br>
    GC<span class="pp-mut">C</span>CGCGC<br>
    GCG<span class="pp-mut">G</span>GCGC<br>
    GCGC<span class="pp-mut">C</span>CGC<br>
    GCGCG<span class="pp-mut">G</span>GC<br>
    GCGCGC<span class="pp-mut">C</span>C<br>
    GCGCGCG<span class="pp-mut">G</span>
    </div>

Sequential enumeration (mode="sequential")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode='sequential'`` with ``num_mutations=1`` enumerates every single-point
variant in deterministic order, covering all positions and non-wild-type bases.

.. code-block:: python

    wt      = pp.from_seq("ACGT")
    mutants = wt.mutagenize(num_mutations=1, mode="sequential", style="red")
    mutants.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mutants: seq_length=4, num_states=12</em>
    <span class="pp-mut">C</span>CGT<br>
    <span class="pp-mut">G</span>CGT<br>
    <span class="pp-mut">T</span>CGT<br>
    A<span class="pp-mut">A</span>GT<br>
    A<span class="pp-mut">G</span>GT
    <span class="pp-ellipsis">... (12 total)</span>
    </div>

See :func:`~poolparty.mutagenize`.
