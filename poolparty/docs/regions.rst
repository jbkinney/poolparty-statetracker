Sequence Regions
================

You often want to perform different operations on different parts of a
sequence. Regions let you mark specific segments with XML-style tags so that
operations can target them by name.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Tag syntax
----------

PoolParty supports two forms of region tag:

**Opening/closing pairs** enclose a segment of the sequence:

.. code-block:: python

    wt = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    wt.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool[0]: seq_length=16, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>

**Self-closing tags** mark a zero-length insertion point:

.. code-block:: python

    wt = pp.from_seq("ACGT<ins/>ACGT")
    wt.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool[0]: seq_length=8, num_states=1</em>
    ACGT<span class="pp-xtag-cre">&lt;ins/&gt;</span>ACGT
    </div>

Tags can be written inline when creating a pool with ``from_seq`` or
``from_seqs``, or added programmatically with :doc:`operations/insert_tags`
or :doc:`operations/annotate_region`.

----

Targeting operations with ``region=``
-------------------------------------

Many operations accept a ``region`` parameter that restricts the operation to
the tagged region. Flanking sequences are left unchanged:

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    mutants = wt.mutagenize(num_mutations=1, region="cre", mode="sequential")
    mutants.print_library(num_seqs=4)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool[1]: seq_length=16, num_states=24</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>CTCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>GTCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>TTCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>AACGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    <span class="pp-ellipsis">... (24 total)</span>
    </div>

Only the 8 bases inside ``<cre>`` are mutated; the flanking ``AAAA`` and
``TTTT`` remain intact. See :doc:`operations/region_operations` for the full
list of region-aware operations.

----

Persistence through the DAG
----------------------------

Region tags persist through the DAG and remain valid even when upstream
operations change the content within a region. This means multiple operations
can target the same region in series:

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    mutants = wt.mutagenize(num_mutations=1, region="cre", mode="sequential")
    dels    = mutants.deletion_scan(deletion_length=3, region="cre", mode="sequential")
    dels.print_library(num_seqs=4)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool[4]: seq_length=16, num_states=144</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>---GATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>C---ATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>CT---TCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>CTC---CG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    <span class="pp-ellipsis">... (144 total)</span>
    </div>

Here ``mutagenize`` produces 24 single-point mutants of the ``cre`` region,
and ``deletion_scan`` then slides a 3-bp deletion across the same region (6
positions per mutant), giving 24 × 6 = 144 total sequences. The ``cre`` tag
is valid at both steps.

----

Inspecting regions
------------------

Every pool tracks which regions are present in its sequences via the
``pool.regions`` property:

.. code-block:: python

    wt = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT<ins/>GGGG")
    wt.regions

.. code-block:: text

    {Region(name='cre', seq_length=8), Region(name='ins', seq_length=0)}

Each :class:`~poolparty.Region` object records the region's name and the
length of its content (``0`` for self-closing tags). See
:class:`~poolparty.Region` in the :doc:`api` for full details.
