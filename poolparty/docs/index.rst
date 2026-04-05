PoolParty Documentation
=======================

**PoolParty** is a Python package for designing complex oligonucleotide
sequence libraries. It provides a declarative, composable interface for
generating DNA libraries used in MPRA (massively parallel reporter assays),
deep mutational scanning, in silico analysis of genomic DNNs, and other
high-throughput experiments.

Why PoolParty?
--------------

Designing DNA libraries often involves combining multiple types of sequence
modifications — mutations, insertions, deletions, shuffles — across multiple
regions with mixed coverage requirements. PoolParty lets you:

- **Compose operations**: Chain operations like ``.mutagenize()``,
  ``.deletion_scan()``, and ``.insertion_scan()`` to build complex libraries
- **Tag regions**: Use XML-like syntax to mark and manipulate specific regions
  of sequences
- **Use lazy evaluation**: Sequences are generated on-demand, enabling libraries
  with billions of potential variants
- **Track provenance**: Each sequence comes with a structured record of how it
  was built — ready for filtering, grouping, and modeling
- **Style output**: Visual annotations highlight sequence modifications and
  regions for quick auditing

.. code-block:: python

    import poolparty as pp

    # Initialize PoolParty
    pp.init()

    # Create a template with tagged regions
    template = pp.from_seq("ACGT<cre>GGAAAGCGGGCAGTGAGC</cre>TTTT<bc/>GGGG")

    # Generate single-nucleotide mutations in the CRE region
    mutant_library = template.mutagenize(
        region="cre",
        num_mutations=1,
        mode="sequential"
    )

    # Generate the library as a DataFrame
    df = mutant_library.generate_library()
    print(f"Generated {len(df)} sequences")

Operations
----------

**Source**
    Create sequence pools from sequences, FASTA files, IUPAC codes, motifs,
    k-mer enumeration, and constrained barcodes.

**Transformation**
    Apply nucleotide and codon-level mutagenesis, shuffling, and recombination.
    Codon-aware operations preserve reading frames for protein-coding sequences.

**Scanning**
    Perform positional scanning with insertion, deletion, replacement, and
    mutagenesis scans across sequence regions.

**Region**
    Tag regions with XML-like syntax, extract or replace tagged regions,
    and target operations to specific sequence regions.

**Composition & Control**
    Combine pools with stack and join. Slice, shuffle, sample, repeat, filter,
    and synchronize library states.

**Export**
    Generate libraries as DataFrames, CSV, or FASTA files.

Installation
------------

Install from PyPI:

.. code-block:: bash

    pip install poolparty

Or install from source:

.. code-block:: bash

    git clone https://github.com/jbkinney/poolparty-statetracker.git
    cd poolparty-statetracker/poolparty
    pip install -e .

Quick Example
-------------

Stack different variant types into a single barcoded library:

.. code-block:: python

    import poolparty as pp

    pp.init()

    # Create a template with tagged regions
    template = pp.from_seq("ACGT<cre>GGAAAGCGGGCAGTGAGC</cre>TTTT<bc/>GGGG")

    # Create different variant pools
    mutations = template.mutagenize(region="cre", num_mutations=1)
    deletions = template.deletion_scan(region="cre", deletion_length=5)

    # Combine into one library
    combined = pp.stack([mutations, deletions])

    # Add barcodes to all variants
    barcoded = combined.insert_kmers(region="bc", length=10)

    # Generate final library
    df = barcoded.generate_library()
    print(f"Generated {len(df)} sequences")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   tutorials/index
   pool
   Sequence Regions <regions>
   Sequence Metadata <metadata/index>
   operations/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

See Also
--------

- `StateTracker <https://statetracker.readthedocs.io>`_: Composable states for
  combinatorial enumeration (used internally by PoolParty)
