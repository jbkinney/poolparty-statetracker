PoolParty Documentation
=======================

**PoolParty** is a Python package that streamlines the design of complex
DNA sequence libraries. Each library is specified as a computational graph
in a few lines of code, with sequences generated on demand. Over 50
built-in operations cover nucleotide- and codon-level mutagenesis,
scanning, barcode generation, and more. Applications include massively
parallel reporter assays, deep mutational scanning, and in silico
analysis of genomic models.

.. image:: /_static/images/figure1a.drawio.svg
   :width: 90%
   :align: center
   :alt: PoolParty overview: Pools represent sequence collections, Operations transform them, and together they form a directed acyclic graph (DAG) specifying the library design.

.. raw:: html

   <div style="margin-bottom: 1.5em;"></div>

Why PoolParty?
--------------

Designing DNA libraries often involves combining multiple types of
sequence modifications (mutations, insertions, deletions, replacements)
across multiple regions. PoolParty lets you:

- **Chain operations**: Build pipelines from operations like ``mutagenize``,
  ``deletion_scan``, and ``insertion_scan`` to produce complex variant
  libraries
- **Tag regions**: Mark segments of a sequence with XML-style tags so
  operations can target them by name
- **Track construction history**: Each sequence carries a design card
  recording how it was built, ready for filtering and analysis
- **Style output**: Visual annotations highlight mutations, deletions,
  and regions for quick auditing

PoolParty provides over 50 built-in operations across six categories.
See :doc:`operations/index` for the full catalog.

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

Example
-------

Stack different variant types into a single barcoded library:

.. code-block:: python

    import poolparty as pp

    pp.init()

    template = pp.from_seq("ACGT<cre>GGAAAGCGGGCAGTGAGC</cre>TTTT<bc/>GGGG")

    mutations = template.mutagenize(region="cre", num_mutations=1)
    deletions = template.deletion_scan(region="cre", deletion_length=5)

    combined = pp.stack([mutations, deletions])
    library = combined.insert_kmers(region="bc", length=10).named("library")

    library.print_library(num_seqs=6, seed=0)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">library: seq_length=40, num_states=2</em>
    ACGT<span class="pp-xtag-light">&lt;cre&gt;</span>GGAAAGCGG<span class="pp-mut">C</span>CAGTGAGC<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<span class="pp-xtag-light">&lt;bc&gt;</span>CCGATGGGGG<span class="pp-xtag-light">&lt;/bc&gt;</span>GGGG<br>
    ACGT<span class="pp-xtag-light">&lt;cre&gt;</span>GGAAAGCGGGCAG<span class="pp-style-green">-----</span><span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<span class="pp-xtag-light">&lt;bc&gt;</span>TTCGGGATAG<span class="pp-xtag-light">&lt;/bc&gt;</span>GGGG<br>
    ACGT<span class="pp-xtag-light">&lt;cre&gt;</span>GGAAAGCGG<span class="pp-mut">A</span>CAGTGAGC<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<span class="pp-xtag-light">&lt;bc&gt;</span>TACCTATTTT<span class="pp-xtag-light">&lt;/bc&gt;</span>GGGG<br>
    ACGT<span class="pp-xtag-light">&lt;cre&gt;</span>GGA<span class="pp-style-green">-----</span>GGCAGTGAGC<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<span class="pp-xtag-light">&lt;bc&gt;</span>CTTGAATGGC<span class="pp-xtag-light">&lt;/bc&gt;</span>GGGG<br>
    ACGT<span class="pp-xtag-light">&lt;cre&gt;</span>GGA<span class="pp-mut">G</span>AGCGGGCAGTGAGC<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<span class="pp-xtag-light">&lt;bc&gt;</span>CGCAGCCTGG<span class="pp-xtag-light">&lt;/bc&gt;</span>GGGG<br>
    ACGT<span class="pp-xtag-light">&lt;cre&gt;</span>GGAAAG<span class="pp-style-green">-----</span>AGTGAGC<span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT<span class="pp-xtag-light">&lt;bc&gt;</span>AACTGGTCGG<span class="pp-xtag-light">&lt;/bc&gt;</span>GGGG
    </div>

The :doc:`quickstart` walks through each concept step by step. For
complete real-world examples, see the :doc:`tutorials/index`.

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
