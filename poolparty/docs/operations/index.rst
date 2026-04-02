Operations
==========

PoolParty provides composable operations for designing DNA sequence libraries.
Each operation returns a new :class:`~poolparty.Pool`, so calls can be chained
into a declarative pipeline. All examples assume:

.. code-block:: python

    import poolparty as pp
    pp.init()

Output blocks show representative pool contents. Colour indicates
**regions** (blue), **mutations** (red), **deletions** (grey), and
**insertions** (green). Stochastic operations show example draws.

----

Creating Pools
--------------
Operations that create pools from various input formats or specifications. They do not require an existing pool as input.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`from_seq`
     - Create a pool from a single sequence string, optionally with region tags.
   * - :doc:`from_seqs`
     - Create a pool that draws uniformly from a list of sequences.
   * - :doc:`from_fasta`
     - Load sequences from a FASTA file.
   * - :doc:`from_iupac`
     - Enumerate all sequences consistent with an IUPAC ambiguity string.
   * - :doc:`from_motif`
     - Sample sequences from a position-probability matrix.
   * - :doc:`get_kmers`
     - Enumerate every k-mer of a given length over a specified alphabet.
   * - :doc:`get_barcodes`
     - Generate DNA barcodes satisfying distance, GC, and homopolymer constraints.

----

Combining Pools
---------------
Operations that combine multiple pools into a single pool, either by concatenating sequences end-to-end or by stacking their state spaces.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`join`
     - Concatenate a list of pools end-to-end into a single composite pool.
   * - :doc:`concatenate`
     - Shorthand operators ``+`` (two pools) and ``*`` (repeat N times) for joining pools.

----

Sequence Mutagenesis
--------------------
Operations that introduce random mutations, insertions, deletions, or recombinations into sequences. These operations are stochastic and produce a different draw each time.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`mutagenize`
     - Introduce random point mutations at a specified rate or count.
   * - :doc:`shuffle_seq`
     - Randomly permute the bases of a sequence or a tagged region.
   * - :doc:`recombine`
     - Produce chimeric sequences by recombining with source sequences at random breakpoints.
   * - :doc:`flip`
     - Produce forward and reverse-complement variants as distinct pool states.

----

Scan Operations
---------------
Operations that systematically tile across a sequence with a sliding window, applying a transformation at each position.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`replacement_scan`
     - Replace a sliding window at each position with sequences drawn from an insertion pool.
   * - :doc:`deletion_scan`
     - Systematically delete a fixed-length window at each position.
   * - :doc:`insertion_scan`
     - Insert sequences from a pool at each position along the sequence.
   * - :doc:`shuffle_scan`
     - Shuffle bases within a sliding window at each position.
   * - :doc:`mutagenize_scan`
     - Apply random point mutations within a sliding window tiling across the sequence.
   * - :doc:`subseq_scan`
     - Extract the subsequence at each sliding-window position.

----

Multiscan Operations
--------------------
Operations that apply multiple simultaneous scans across a sequence, allowing for non-overlapping windows to be modified in parallel.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`replacement_multiscan`
     - Place multiple non-overlapping replacement windows simultaneously.
   * - :doc:`deletion_multiscan`
     - Apply multiple simultaneous deletion windows.
   * - :doc:`insertion_multiscan`
     - Insert sequences at multiple positions simultaneously.

----

Region Operations
-----------------
Operations that modify sequences by tagging, replacing, or applying transformations to specific regions defined by tags. Regions can be nested and overlapping, and are identified by their tag names.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`insert_tags`
     - Add region tags to a sequence by specifying start/stop indices.
   * - :doc:`remove_tags`
     - Remove region tags, optionally keeping or discarding their contents.
   * - :doc:`annotate_region`
     - Tag a region by position range with an optional display style.
   * - :doc:`replace_region`
     - Replace the content of a named region with sequences from a pool.
   * - :doc:`apply_at_region`
     - Apply a transformation function to the content of a named region.
   * - :doc:`extract_region`
     - Extract the content of a named region as a new pool.
   * - :doc:`region_scan`
     - Run a replacement scan confined to a named region.

----


State Operations
----------------
Operations that manipulate the state space of a pool without directly modifying sequences. These operations can be used to combine, sample, or reorder the underlying sequences drawn from a pool.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`stack`
     - Combine multiple pools by stacking their state spaces.
   * - :doc:`repeat`
     - Repeat the state space of a pool N times, producing duplicate draws.
   * - :doc:`sample`
     - Draw a fixed number of sequences, optionally with a reproducibility seed.
   * - :doc:`slice_states`
     - Retain a contiguous slice of the state space.
   * - :doc:`shuffle_states`
     - Randomly reorder the state space.
   * - :doc:`sync`
     - Synchronize multiple pools to iterate in lockstep (in-place).

----

ORF Operations
--------------
Operations that specifically target open reading frames (ORFs) within sequences, allowing for codon-level manipulations and translations while respecting reading frame boundaries.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`mutagenize_orf`
     - Introduce codon-level missense mutations within a coding sequence.
   * - :doc:`translate`
     - Translate a DNA pool to a protein pool.
   * - :doc:`annotate_orf`
     - Tag an ORF region within a longer sequence.
   * - :doc:`stylize_orf`
     - Apply alternating codon colours to visualize reading frames.
   * - :doc:`reverse_translate`
     - Back-translate a protein pool to DNA using a codon table.

----

Fixed Operations
----------------
Operations that apply deterministic transformations to sequences, such as taking the reverse complement, changing case, or applying display styles.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`rc`
     - Take the reverse complement of a sequence or named region.
   * - :doc:`case_ops`
     - Convert sequence case: ``upper``, ``lower``, ``swapcase``.
   * - :doc:`stylize`
     - Apply a named display style to a sequence or region.
   * - :doc:`clear_gaps`
     - Remove gap characters (``-``) from sequences.
   * - :doc:`clear_annotation`
     - Strip region tags while keeping the underlying sequence.
   * - :doc:`slice_seq`
     - Extract a subsequence by index range, named region, or both.
   * - :doc:`add_prefix`
     - Add a label prefix to sequence names without modifying sequences.

----

Filtering and Scoring
---------------------
Operations that evaluate sequence content: ``filter`` removes sequences that fail a predicate; ``score`` records a metric as a design card column without changing the sequence.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`filter`
     - Retain only sequences satisfying a predicate function.
   * - :doc:`score`
     - Evaluate a function on each sequence and record the result as a design card column.

----

Library Generation
------------------
Operations typically used at the end of a design pipeline to generate a final library of sequences for synthesis or analysis.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`generate_library`
     - Evaluate the pool DAG and return a DataFrame of sequences.
   * - :doc:`materialize`
     - Eagerly generate sequences and cache them in a new standalone pool.

----

.. toctree::
   :hidden:

   from_seq
   from_seqs
   from_fasta
   from_iupac
   from_motif
   get_kmers
   join
   concatenate
   mutagenize
   shuffle_seq
   recombine
   replacement_scan
   deletion_scan
   insertion_scan
   shuffle_scan
   mutagenize_scan
   insert_tags
   remove_tags
   annotate_region
   replace_region
   apply_at_region
   region_scan
   replacement_multiscan
   deletion_multiscan
   insertion_multiscan
   stack
   repeat
   sample
   slice_states
   shuffle_states
   mutagenize_orf
   translate
   annotate_orf
   stylize_orf
   rc
   case_ops
   stylize
   clear_gaps
   clear_annotation
   filter
   score
   flip
   get_barcodes
   subseq_scan
   extract_region
   sync
   reverse_translate
   slice_seq
   add_prefix
   generate_library
   materialize