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

----

Combining Pools
---------------

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

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`mutagenize`
     - Introduce random point mutations at a specified rate or count.
   * - :doc:`shuffle_seq`
     - Randomly permute the bases of a sequence or a tagged region.
   * - :doc:`recombine`
     - Produce chimeric sequences by recombining with source sequences at random breakpoints.

----

Scan Operations
---------------

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

----

Region Operations
-----------------

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
   * - :doc:`region_scan`
     - Run a replacement scan confined to a named region.

----

Multiscan Operations
--------------------

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

State Operations
----------------

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

----

ORF Operations
--------------

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

----

Fixed Operations
----------------

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

----

Filtering
---------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`filter`
     - Retain only sequences satisfying a predicate function.

----

Library Generation
------------------

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
   generate_library
   materialize