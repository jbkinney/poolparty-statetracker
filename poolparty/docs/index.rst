PoolParty Documentation
=======================

**PoolParty** is a Python package for declarative design of complex
oligonucleotide sequence pools. It provides a powerful, composable API
for generating DNA libraries used in multiplexed assays such as MPRAs,
deep mutational scanning, and MAVEs.

Why PoolParty?
--------------

Designing complex DNA libraries typically requires ad-hoc scripting that is
difficult to reproduce, validate, and share. PoolParty shifts library design
from imperative code to a declarative specification using a directed acyclic
graph (DAG) of composable operations.

**Key benefits:**

- **Declarative design**: Specify *what* you want, not *how* to compute it
- **Composable operations**: Build complex libraries from simple, reusable parts
- **Automatic provenance**: Track the origin of every sequence in your library
- **State algebra**: Leverage `StateTracker <https://statetracker.readthedocs.io>`_
  for combinatorial enumeration with random access
- **Region-aware**: Tag and manipulate sequence regions with XML-like syntax

.. code-block:: python

    import poolparty as pp

    # Initialize PoolParty
    pp.init()

    # Create a promoter variant pool with single mutations
    promoter = pp.from_seq("ATCGATCGATCGATCGATCG", region="promoter")
    variants = promoter.mutagenize(alphabet="ACGT", k=1)

    # Add random barcodes
    barcodes = pp.get_kmers(length=10, alphabet="ACGT")

    # Combine promoter variants with barcodes
    library = pp.join([variants, barcodes])

    # Generate the complete library
    df = library.generate_library(num_seqs=100)
    print(f"Generated {len(df)} sequences")

Features
--------

**Base Operations**
    Create sequence pools from sequences, FASTA files, IUPAC codes, motifs,
    or k-mer enumeration. Apply mutations, shuffling, and recombination.

**Scan Operations**
    Perform tiled mutagenesis with insertion, deletion, replacement, and
    shuffle scans across sequence positions.

**Region Operations**
    Tag regions with XML-like syntax, extract or replace tagged regions,
    and perform region-aware scans.

**State Operations**
    Slice, shuffle, sample, and repeat library states using StateTracker's
    composable state algebra.

**ORF Operations**
    Codon-aware mutagenesis that preserves reading frames for protein-coding
    sequences.

Installation
------------

Install from PyPI:

.. code-block:: bash

    pip install poolparty

Or install from source:

.. code-block:: bash

    git clone https://github.com/jbkinney/poolparty-statecounter.git
    cd poolparty-statecounter/poolparty
    pip install -e .

Quick Example
-------------

Design a library with promoter variants and barcodes:

.. code-block:: python

    import poolparty as pp

    # Initialize the Party context
    pp.init()

    # Define the wild-type sequence with a tagged region
    wt_seq = "AAAA<cre>ATCGATCGATCG</cre>TTTT"
    wt = pp.from_seq(wt_seq)

    # Create single-nucleotide variants in the CRE region
    variants = wt.replacement_scan(
        region="cre",
        replacement_seqs=["A", "C", "G", "T"],
        length=1
    )

    # Generate the library
    df = variants.generate_library()
    print(df[["seq", "seq_name"]].head(10))

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   operations/index
   pool

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Related Projects
----------------

- `StateTracker <https://statetracker.readthedocs.io>`_: Composable states for
  combinatorial enumeration (used internally by PoolParty)
