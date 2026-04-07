Library Size
============

Every pool has a ``num_states`` property — the number of distinct sequences it
can produce. Each operation has an **internal state** (see :doc:`modes`) whose
count is determined by its mode and parameters. When operations are chained, the output pool's ``num_states`` is the product
of all internal state counts along the chain. Other composition patterns
(stacking, synchronisation) follow different rules described below.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Composition rules
-----------------

Three rules determine how ``pool.num_states`` changes as operations are
applied.

.. rubric:: Multiply (Cartesian product)

Chaining an operation pairs every input sequence with every possibility of
the operation, producing all combinations. The resulting ``num_states`` is
the input pool's ``num_states`` multiplied by ``operation.num_states``.

.. code-block:: python

    seqs    = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")  # seqs (pool): 3 states
    mutants = pp.mutagenize(seqs, num_mutations=1, mode="sequential")  # mutagenize (op): 9 internal states
    print(mutants.num_states)   # 27  (3 × 9)

.. rubric:: Add (disjoint union)

``stack`` (or the ``+`` operator) places its input pools side by side. Sequences
from each branch appear in the output but are not combined with each other,
so the resulting ``num_states`` is the sum of the inputs' ``num_states``.

.. code-block:: python

    a = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")          # a (pool): 3 states
    b = pp.from_seqs(["TTT", "ATA", "GAG", "CTC"], mode="sequential")  # b (pool): 4 states
    combined = pp.stack([a, b])
    print(combined.num_states)  # 7  (3 + 4)

.. rubric:: No change (×1)

Fixed-mode operations transform each input sequence in exactly one
deterministic way, so the number of sequences stays the same.
``operation.num_states`` is 1.

.. code-block:: python

    seqs    = pp.from_seqs(["ACG", "TGA", "CCC"], mode="sequential")  # seqs (pool): 3 states
    flipped = pp.rc(seqs)                                              # rc (op): 1 internal state
    print(flipped.num_states)   # 3  (3 × 1)

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
     - reduces
     - ``sample``, ``filter``, ``slice_states``
   * - State
     - unchanged
     - ``shuffle_states``, ``sync``, ``score``, ``materialize``
   * - Utilities
     - unchanged
     - ``rc``, ``upper``, ``lower``, ``swapcase``, ``stylize``, ``clear_gaps``, ``clear_annotation``, ``slice_seq``, ``add_prefix``

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

    left  = pp.from_seqs(["AAA", "CCC", "GGG"], mode="sequential")
    right = pp.from_seqs(["TTT", "AAA", "CCC"], mode="sequential")

    pp.sync([left, right])
    paired = pp.join([left, right])
    print(paired.num_states)                   # 3

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

Practical tips
--------------

- Use the ``num_states`` parameter to cap large sequential enumerations before
  they multiply downstream.
- Use :doc:`sync` to pair pools that should iterate together instead of
  forming a Cartesian product.
- Use ``sample`` or ``slice_states`` to reduce an oversized library after
  construction.
- In random mode without the ``num_states`` parameter, each sequence gets a
  fresh random draw (×1, no multiplication). With ``num_states=N``, the
  operation contributes *N* randomly chosen designs that multiply with the
  input pool (×N).
