Operations
==========

PoolParty provides composable operations for designing DNA sequence libraries.
Each operation returns a new :class:`~poolparty.Pool`, so calls can be chained
into a declarative pipeline. All examples assume:

.. code-block:: python

    import poolparty as pp
    pp.init()

.. toctree::
   :hidden:

   Modes <modes>
   Library Size <library_size>
   Sources <source_operations>
   Mutagenesis <mutagenesis>
   Scanning <scanning>
   Regions <region_operations>
   ORF <orf_operations>
   Utilities <sequence_utilities>
   Composition <composition_operations>
   State <state_operations>

----

.. rubric:: Key concepts

Most operations accept a **mode** parameter that controls whether outputs are
enumerated exhaustively (``sequential``), sampled randomly (``random``), or
uniquely determined by the input (``fixed``). Each operation has **internal
states** determined by its mode and parameters — see :doc:`modes` for details.

When operations are combined, their internal states compose to determine the
output pool's ``num_states`` — through multiplication, addition, or other rules
depending on the operation. See :doc:`library_size` for details and a worked
example.

Every operation that accepts a ``pool`` argument is also available as a method
on any Pool. ``pp.mutagenize(wt, ...)`` and ``wt.mutagenize(...)`` are
equivalent, so operations can be chained left-to-right into a pipeline.

----

.. rubric:: Source Operations

Create the initial Pools from which all downstream Pools are constructed.
See :doc:`source_operations`.

----

.. rubric:: Transformation Operations

Modify sequence content through mutations, positional scanning, region
manipulation, codon-level operations, and deterministic transforms.

- :doc:`mutagenesis` — mutations, shuffling, recombination, and orientation flipping
- :doc:`scanning` — systematic positional tiling (single and multi-window)
- :doc:`region_operations` — tag and manipulate named sequence regions
- :doc:`orf_operations` — codon-aware operations for protein-coding sequences
- :doc:`sequence_utilities` — deterministic per-sequence transforms

----

.. rubric:: Composition Operations

Combine sequences from multiple Pools, either by concatenating sequences
end-to-end or by merging state spaces.
See :doc:`composition_operations`.

----

.. rubric:: State Operations

Control which sequences are in a Pool and how they are ordered, including
methods to replicate, sample, filter, score, and synchronize.
See :doc:`state_operations`.
