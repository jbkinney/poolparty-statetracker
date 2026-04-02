from_fasta
==========

Extract one or more genomic regions from a FASTA file and create a pool.
Coordinates are 0-based half-open intervals ``[start, stop)`` following the
convention ``(chrom, start, stop, strand)``. A single tuple gives a fixed
pool; a list of tuples gives a sequential pool that iterates through all
extracted sequences.

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
   * - ``fasta_path``
     - ``str | Path``
     - *(required)*
     - Path to the FASTA file. Indexed via ``pyfaidx`` on first use; a
       ``.fai`` index file is created automatically if absent.
   * - ``coordinates``
     - ``tuple | list[tuple]``
     - *(required)*
     - A single tuple ``(chrom, start, stop, strand)`` or a list of such
       tuples. ``start``/``stop`` are 0-based integers. ``strand`` is
       ``'+'`` or ``'-'``; ``'-'`` triggers automatic reverse
       complementation.
   * - ``pool``
     - ``Pool | None``
     - ``None``
     - Background pool. When provided with ``region``, the extracted
       sequence replaces the content of that region.
   * - ``region``
     - ``str | None``
     - ``None``
     - Region to replace in ``pool``. Required when ``pool`` is provided.
   * - ``remove_tags``
     - ``bool | None``
     - ``None``
     - If ``True``, strip region tags from the output (single-coordinate
       mode only).
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names. Names follow
       ``{prefix}_{chrom}:{start}-{stop}({strand})``.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to every extracted sequence.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

.. note::

   For **circular genomes**, ``start > stop`` indicates wrap-around across
   the origin — the extracted sequence runs from ``start`` to the end of
   the chromosome and continues from the beginning to ``stop``.

----

Examples
--------

Single region, forward strand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract 20 bases from chromosome 1, forward strand.

.. code-block:: python

    pool = pp.from_fasta(
        "genome.fa",
        coordinates=("chr1", 1000, 1020, "+"),
    )

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; chr1:1000&ndash;1020(+))</em>
    ATCGATCGATCGATCGATCG
    </div>

Reverse-strand extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~

``strand='-'`` automatically returns the reverse complement of the interval
— use this for genes encoded on the minus strand.

.. code-block:: python

    pool = pp.from_fasta(
        "genome.fa",
        coordinates=("chr2", 5000, 5010, "-"),
    )

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; chr2:5000&ndash;5010(-), reverse complemented)</em>
    CGTAGCTAGC
    </div>

Multiple coordinates — sequential pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A list of tuples creates a sequential pool. Each sequence is named
automatically; ``prefix`` prepends a custom label.

.. code-block:: python

    coords = [
        ("chr1", 1000, 1010, "+"),
        ("chr2", 5000, 5010, "-"),
        ("chr3", 200,  210,  "+"),
    ]
    pool = pp.from_fasta("genome.fa", coordinates=coords, prefix="enh")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (3 sequences, sequential)</em>
    ATCGATCGAT &nbsp;<em style="color:#6b7280;">enh_chr1:1000-1010(+)</em><br>
    CGTAGCTAGC &nbsp;<em style="color:#6b7280;">enh_chr2:5000-5010(-)</em><br>
    GCATGCATGC &nbsp;<em style="color:#6b7280;">enh_chr3:200-210(+)</em>
    </div>

Inserting into a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide ``pool`` and ``region`` to place extracted sequences inside a fixed
flanking context — useful for tiling genomic sequences into a library vector.

.. code-block:: python

    vector = pp.from_seq("GCGCGC<insert>XXXXXXXXXX</insert>GCGCGC")
    coords = [("chr1", 1000, 1010, "+"), ("chr1", 2000, 2010, "+")]
    pool   = pp.from_fasta("genome.fa", coordinates=coords,
                           pool=vector, region="insert")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (2 sequences &mdash; genomic tiles in <em>insert</em> region)</em>
    GCGCGC<span class="pp-region">ATCGATCGAT</span>GCGCGC<br>
    GCGCGC<span class="pp-region">TTGGAACCTA</span>GCGCGC
    </div>

See :func:`~poolparty.from_fasta`.
