get_barcodes
============

Generate a pool of DNA barcodes satisfying distance and quality
constraints. All barcodes are pre-generated at construction time using a
greedy random algorithm, so the resulting pool is a sequential leaf with
``num_states == num_barcodes``.

Constraints available: minimum edit (Levenshtein) distance, minimum
Hamming distance (fixed-length only), GC content range, maximum
homopolymer run length, and minimum edit distance from a set of
user-supplied sequences to avoid (e.g. adapters).

.. note::

   ``get_barcodes`` must be called inside an active
   :class:`~poolparty.Party` context (i.e. within a ``with pp.Party()``
   block).

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
   * - ``num_barcodes``
     - ``int``
     - *(required)*
     - Number of barcodes to generate.
   * - ``length``
     - ``int | list[int]``
     - *(required)*
     - Barcode length. A single ``int`` gives fixed-length barcodes; a
       list of ints generates variable-length barcodes padded to the
       maximum length.
   * - ``length_proportions``
     - ``list[float] | None``
     - ``None``
     - Target fraction of each length in ``length`` list. Values are
       normalised to sum to 1. ``None`` distributes evenly. Ignored
       when ``length`` is a single int.
   * - ``min_edit_distance``
     - ``int | None``
     - ``None``
     - Minimum Levenshtein distance between any two barcodes. Works for
       both fixed- and variable-length sets.
   * - ``min_hamming_distance``
     - ``int | None``
     - ``None``
     - Minimum Hamming distance between same-length barcodes. Cannot be
       combined with variable-length ``length`` lists; use
       ``min_edit_distance`` instead.
   * - ``gc_range``
     - ``tuple[float, float] | None``
     - ``None``
     - ``(min_gc, max_gc)`` as fractions in [0, 1]. Barcodes outside
       this range are rejected.
   * - ``max_homopolymer``
     - ``int | None``
     - ``None``
     - Maximum consecutive identical bases allowed. Barcodes with longer
       runs are rejected.
   * - ``avoid_sequences``
     - ``list[str] | None``
     - ``None``
     - External sequences (e.g. adapters) that barcodes must stay away
       from. Requires ``avoid_min_distance``.
   * - ``avoid_min_distance``
     - ``int | None``
     - ``None``
     - Minimum edit distance from every sequence in ``avoid_sequences``.
       Required when ``avoid_sequences`` is provided.
   * - ``padding_char``
     - ``str``
     - ``'-'``
     - Character used to pad shorter variable-length barcodes to the
       maximum length.
   * - ``padding_side``
     - ``str``
     - ``'right'``
     - ``'right'`` appends padding; ``'left'`` prepends it.
   * - ``seed``
     - ``int | None``
     - ``None``
     - Random seed for reproducible barcode generation.
   * - ``max_attempts``
     - ``int``
     - ``100000``
     - Maximum candidate attempts before raising a ``ValueError``. Raise
       this or relax constraints if generation fails.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to barcode sequences.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``cards``
     - ``list | dict | None``
     - ``None``
     - Design card keys to include. Available keys: ``'barcode_index'``,
       ``'barcode'``.

----

Examples
--------

Basic fixed-length barcodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate 10 length-8 barcodes with minimum edit distance 3.

.. code-block:: python

    with pp.Party() as party:
        barcodes = pp.get_barcodes(
            num_barcodes=10,
            length=8,
            min_edit_distance=3,
            seed=42,
        )

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (10 barcodes, sequential &mdash; min edit distance 3)</em>
    ACGTACGT<br>
    TTGACCAG<br>
    GGCATCGA<br>
    <span class="pp-ellipsis">... (10 total; pairwise edit distance &ge; 3)</span>
    </div>

GC content and homopolymer constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict GC content to 40–60 % and disallow runs of 3 or more identical
bases.

.. code-block:: python

    with pp.Party() as party:
        barcodes = pp.get_barcodes(
            num_barcodes=20,
            length=10,
            min_edit_distance=3,
            gc_range=(0.4, 0.6),
            max_homopolymer=2,
            seed=0,
        )

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (20 barcodes &mdash; 40&ndash;60% GC, no homopolymer run &gt; 2)</em>
    ACGATCGATC<br>
    GCTAGCTAGC<br>
    <span class="pp-ellipsis">... (20 total)</span>
    </div>

Avoiding adapter sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep all barcodes at least edit distance 4 from a set of adapter
sequences to prevent ligation artefacts.

.. code-block:: python

    adapters = ["AGATCGGAAG", "CTGTCTCTTA"]
    with pp.Party() as party:
        barcodes = pp.get_barcodes(
            num_barcodes=50,
            length=8,
            min_edit_distance=3,
            avoid_sequences=adapters,
            avoid_min_distance=4,
            seed=1,
        )

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (50 barcodes &mdash; edit distance &ge; 4 from adapter sequences)</em>
    <span class="pp-ellipsis">... (50 total; no barcode within edit distance 4 of either adapter)</span>
    </div>

Variable-length barcodes
~~~~~~~~~~~~~~~~~~~~~~~~~

Mix 6-mer and 8-mer barcodes in a 1:1 ratio; shorter barcodes are
right-padded with ``-``.

.. code-block:: python

    with pp.Party() as party:
        barcodes = pp.get_barcodes(
            num_barcodes=10,
            length=[6, 8],
            length_proportions=[0.5, 0.5],
            min_edit_distance=3,
            seed=7,
        )

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (10 barcodes &mdash; 5 &times; 6-mer padded to 8, 5 &times; 8-mer)</em>
    ACGTAC--<br>
    TTGACCAG<br>
    GCTAGC--<br>
    <span class="pp-ellipsis">... (10 total; 6-mers right-padded with &ldquo;--&rdquo;)</span>
    </div>

See :func:`~poolparty.get_barcodes`.
