Library Size
============

Every pool has a ``num_states`` property — the number of internal states in its
design. That is not the same as the number of distinct sequences the pool
produces, and is neither an upper nor a lower bound on it (see
:ref:`states-vs-sequences` below). Each operation has an **internal state**
(see :doc:`modes`) whose count is determined by its mode and parameters. How
``num_states`` composes when operations are combined depends on the operation
type. Three rules cover all cases: multiplication, addition, and no change.
These are described below.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Composition rules
-----------------

Multiply (Cartesian product)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chaining an operation pairs every input sequence with every possibility of
the operation, producing all combinations. The resulting ``num_states`` is
the input pool's ``num_states`` multiplied by ``operation.num_states``.

.. code-block:: python

    seqs    = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")  # seqs (pool): 3 states
    mutants = pp.mutagenize(seqs, num_mutations=1, mode="sequential")  # mutagenize (op): 9 internal states
    print(mutants.num_states)   # 27  (3 × 9)

Add (disjoint union)
~~~~~~~~~~~~~~~~~~~~

``stack`` (or the ``+`` operator) places its input pools side by side. Sequences
from each branch appear in the output but are not combined with each other,
so the resulting ``num_states`` is the sum of the inputs' ``num_states``.

.. code-block:: python

    a = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")          # a (pool): 3 states
    b = pp.from_seqs(["TTT", "ATA", "GAG", "CTC"], mode="sequential")  # b (pool): 4 states
    combined = pp.stack([a, b])
    print(combined.num_states)  # 7  (3 + 4)

Unchanged (×1)
~~~~~~~~~~~~~~

Fixed-mode operations transform each input sequence in exactly one
deterministic way, so the number of sequences stays the same.
``operation.num_states`` is 1.

.. code-block:: python

    seqs    = pp.from_seqs(["ACG", "TGA", "CCC"], mode="sequential")  # seqs (pool): 3 states
    flipped = pp.rc(seqs)                                              # rc (op): 1 internal state
    print(flipped.num_states)   # 3  (3 × 1)

----

Worked example
--------------

A realistic pipeline that uses chaining (multiply), stack (add), and a final
chained step (multiply again):

.. code-block:: python

    wt = pp.from_seq("ACGT<cre>ATCG</cre>TTTT<bc/>GGGG")        # wt.num_states: 1

    # Branch 1: sequential mutagenesis
    mutants = wt.mutagenize(region="cre", num_mutations=1,
                            mode="sequential")
    # mutagenize (op): 4 positions × 3 alt bases = 12 internal states
    # mutants.num_states: 1 × 12 = 12

    # Branch 2: deletion scan
    dels = wt.deletion_scan(region="cre", deletion_length=2,
                            mode="sequential")
    # deletion_scan (op): 3 window positions = 3 internal states
    # dels.num_states: 1 × 3 = 3

    # Stack branches (addition)
    combined = pp.stack([mutants, dels])
    # combined.num_states: 12 + 3 = 15

    # Add barcodes (Cartesian product)
    barcoded = combined.insert_kmers(region="bc", length=2,
                                     mode="sequential")
    # insert_kmers (op): 4² = 16 internal states
    # barcoded.num_states: 15 × 16 = 240

    print(barcoded.num_states)  # 240

----

Synchronisation
---------------

Without ``sync``, joining two independent 3-state pools produces
3 × 3 = 9 states (Cartesian product). After syncing, the pools iterate in
lockstep, producing only 3 paired states. See :doc:`sync` for details.

.. code-block:: python

    left  = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")
    right = pp.from_seqs(["TTT", "AAA", "CCC"], mode="sequential")

    # Without sync: 3 × 3 = 9
    print(pp.join([left, right]).num_states)   # 9

.. code-block:: python

    # Start a fresh context to demonstrate the synced case
    pp.init()

    left  = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")
    right = pp.from_seqs(["TTT", "AAA", "CCC"], mode="sequential")

    pp.sync([left, right])
    paired = pp.join([left, right])
    print(paired.num_states)                   # 3

----

Per-category behaviour
----------------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Category
     - Effect
     - Operation(s)
   * - Source
     - sets initial size
     - ``from_seq``, ``from_seqs``, ``from_fasta``, ``from_iupac``, ``from_motif``, ``get_kmers``, ``get_barcodes``
   * - Mutagenesis
     - multiplies
     - ``mutagenize``, ``shuffle_seq``, ``recombine``, ``flip``
   * - Scanning
     - multiplies
     - ``deletion_scan``, ``insertion_scan``, ``replacement_scan``, ``shuffle_scan``, ``mutagenize_scan``, ``subseq_scan``, and multi-window variants
   * - Regions
     - multiplies
     - ``region_scan``, ``region_multiscan``
   * - Regions
     - multiplies or unchanged
     - ``replace_region`` — multiplies with ``sync=False`` (Cartesian product);
       unchanged with ``sync=True`` (default, 1:1 pairing)
   * - Regions
     - unchanged
     - ``insert_tags``, ``remove_tags``, ``annotate_region``, ``apply_at_region``, ``extract_region``
   * - ORF
     - multiplies
     - ``mutagenize_orf``, ``reverse_translate``
   * - ORF
     - unchanged
     - ``translate``, ``annotate_orf``, ``stylize_orf``
   * - Composition
     - multiplies
     - ``join``
   * - Composition
     - adds
     - ``stack``
   * - State
     - multiplies
     - ``repeat``
   * - State
     - sets a new size
     - ``sample``
   * - State
     - reduces
     - ``slice_states``
   * - State
     - unchanged (rejected sequences become ``NullSeq``)
     - ``filter``
   * - State
     - unchanged
     - ``shuffle_states``, ``sync``, ``score``, ``materialize``
   * - Utilities
     - unchanged
     - ``rc``, ``upper``, ``lower``, ``swapcase``, ``stylize``, ``clear_gaps``, ``clear_annotation``, ``slice_seq``, ``add_prefix``

----

.. _states-vs-sequences:

Distinct states are not distinct sequences
------------------------------------------

``num_states`` counts states, and several ordinary designs give a state count
that differs from the number of distinct sequences produced:

- ``repeat`` asks for copies, so its extra states are duplicates by design.
- ``filter`` replaces a rejected sequence with a ``NullSeq`` rather than
  removing its state, so ``num_states`` is unchanged and some states yield no
  sequence at all.
- ``sample`` draws with replacement by default, so it can select one state
  more than once.
- Two different states can happen to produce the same sequence.
- A random operation built **without** ``num_states`` draws afresh for every
  sequence. It contributes one state, so ``num_states`` stops counting
  sequences at all and the design has no fixed size.

.. code-block:: python

    # The same design, written two ways.
    seq = "ACGTACGTAC"

    print(pp.from_seq(seq).mutagenize(num_mutations=2, mode="sequential").num_states)
    # 405  (45 position pairs × 9 base pairs)

    print(pp.from_seq(seq).mutagenize(num_mutations=2, mode="random").num_states)
    # 1    (a fresh draw per sequence, so there is no total)

Use :ref:`pool.stats() <pool-stats>` to count what a design actually produces.

----

Practical tips
--------------

- Use the ``num_states`` parameter to cap large sequential enumerations before
  they multiply downstream.
- Use :doc:`sync` to pair pools that should iterate together instead of
  forming a Cartesian product.
- Use ``sample`` or ``slice_states`` to reduce an oversized library after
  construction.
- In random mode, passing ``num_states=N`` draws *N* fixed random designs that
  multiply with the input pool. Without ``num_states``, each sequence gets a
  fresh draw and ``pool.num_states`` is unchanged.
