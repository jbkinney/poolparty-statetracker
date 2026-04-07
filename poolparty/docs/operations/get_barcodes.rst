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
   * - ``name``
     - ``str | None``
     - ``None``
     - Operation name.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to barcode sequences.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Enumeration order when combined with other pools.
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

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.get_barcodes` in the
   :doc:`API Reference </api>`.

Examples
--------

Basic fixed-length barcodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate 10 length-8 barcodes with minimum edit distance 3.

.. code-block:: python

    barcodes = pp.get_barcodes(
        num_barcodes=10,
        length=8,
        min_edit_distance=3,
        seed=42,
    )
    barcodes.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">barcodes: seq_length=8, num_states=10</em>
    AAGCCCAA<br>
    TAAACCAC<br>
    TCTGACTG<br>
    GCCGAATA<br>
    GGGATATA<br>
    GGCAACGA<br>
    CATGTGCG<br>
    GCGACCCT<br>
    TGCGACAG<br>
    TGACGCTT
    </div>

GC content and homopolymer constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict GC content to 40–60 % and disallow runs of 3 or more identical
bases.

.. code-block:: python

    barcodes = pp.get_barcodes(
        num_barcodes=20,
        length=10,
        min_edit_distance=3,
        gc_range=(0.4, 0.6),
        max_homopolymer=2,
        seed=0,
    )
    barcodes.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">barcodes: seq_length=10, num_states=20</em>
    TTAGTTGTGC<br>
    TAGTGCTTGA<br>
    AATATGCGAC<br>
    GAGCGTATGC<br>
    CAATGCCTGT<br>
    <span class="pp-ellipsis">... (20 total)</span>
    </div>

Avoiding adapter sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep all barcodes at least edit distance 4 from a set of adapter
sequences to prevent ligation artefacts.

.. code-block:: python

    adapters = ["AGATCGGAAG", "CTGTCTCTTA"]
    barcodes = pp.get_barcodes(
        num_barcodes=50,
        length=8,
        min_edit_distance=3,
        avoid_sequences=adapters,
        avoid_min_distance=4,
        seed=1,
    )
    barcodes.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">barcodes: seq_length=8, num_states=50</em>
    CAGATTTT<br>
    GCAGAAAA<br>
    TCTACTTC<br>
    GCCTGATA<br>
    CGAGTCGG<br>
    <span class="pp-ellipsis">... (50 total)</span>
    </div>

Variable-length barcodes
~~~~~~~~~~~~~~~~~~~~~~~~~

Mix 6-mer and 8-mer barcodes in a 1:1 ratio; shorter barcodes are
right-padded with ``-``. To obtain the unpadded sequences, apply
``clear_gaps()`` before calling ``generate_library()``.

.. code-block:: python

    barcodes = pp.get_barcodes(
        num_barcodes=10,
        length=[6, 8],
        length_proportions=[0.5, 0.5],
        min_edit_distance=3,
        seed=7,
        prefix="bc",
    )
    barcodes.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">barcodes: seq_length=8, num_states=10</em>
    CTAAAGAC<br>
    ATTACA--<br>
    AACATACA<br>
    GTCAGC--<br>
    CGAAAC--<br>
    TGTTGGCC<br>
    AGTGTG--<br>
    ATCGCT--<br>
    AAGGGTTA<br>
    GTAAGTGT
    </div>

.. code-block:: python

    cleaned = barcodes.clear_gaps()
    df = cleaned.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 10 rows × 2 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th></tr>
    <tr><td>bc_0</td><td>CTAAAGAC</td></tr>
    <tr><td>bc_1</td><td>ATTACA</td></tr>
    <tr><td>bc_2</td><td>AACATACA</td></tr>
    <tr><td>bc_3</td><td>GTCAGC</td></tr>
    <tr><td>bc_4</td><td>CGAAAC</td></tr>
    <tr><td>bc_5</td><td>TGTTGGCC</td></tr>
    <tr><td>bc_6</td><td>AGTGTG</td></tr>
    <tr><td>bc_7</td><td>ATCGCT</td></tr>
    <tr><td>bc_8</td><td>AAGGGTTA</td></tr>
    <tr><td>bc_9</td><td>GTAAGTGT</td></tr>
    </table>
    </div>

See :func:`~poolparty.get_barcodes`.
