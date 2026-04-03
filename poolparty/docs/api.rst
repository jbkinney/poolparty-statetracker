API Reference
=============

This page provides complete API documentation for all public classes and functions
in PoolParty, automatically generated from source code docstrings.

.. contents:: On this page
   :local:
   :depth: 2

Core Classes
------------

Pool
~~~~

The main class for building and manipulating sequence pools.

.. autoclass:: poolparty.Pool
   :members:
   :special-members: __init__, __add__, __iter__, __getitem__, __len__

Party
~~~~~

Context manager for PoolParty sessions.

.. autoclass:: poolparty.Party
   :members:
   :special-members: __init__, __enter__, __exit__

Operation
~~~~~~~~~

Abstract base class for all pool operations.

.. autoclass:: poolparty.Operation
   :members:

Region
~~~~~~

Represents a tagged region within a sequence.

.. autoclass:: poolparty.Region
   :members:

Initialization Functions
------------------------

.. autofunction:: poolparty.init

.. autofunction:: poolparty.get_active_party

.. autofunction:: poolparty.clear_pools

.. autofunction:: poolparty.configure_logging

.. autofunction:: poolparty.set_default

.. autofunction:: poolparty.toggle_styles

.. autofunction:: poolparty.toggle_cards

Base Operations
---------------

Functions for creating and transforming sequence pools.

Sequence Creation
~~~~~~~~~~~~~~~~~

.. autofunction:: poolparty.from_seq

.. autofunction:: poolparty.from_seqs

.. autofunction:: poolparty.from_fasta

.. autofunction:: poolparty.from_iupac

.. autofunction:: poolparty.from_motif

.. autofunction:: poolparty.get_kmers

Sequence Transformation
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: poolparty.mutagenize

.. autofunction:: poolparty.shuffle_seq

.. autofunction:: poolparty.recombine

.. autofunction:: poolparty.join

Fixed Operations
----------------

Operations that transform sequences without changing pool size.

.. autofunction:: poolparty.rc

.. autofunction:: poolparty.upper

.. autofunction:: poolparty.lower

.. autofunction:: poolparty.swapcase

.. autofunction:: poolparty.slice_seq

.. autofunction:: poolparty.clear_gaps

.. autofunction:: poolparty.clear_annotation

.. autofunction:: poolparty.stylize

Scan Operations
---------------

Tiled mutagenesis operations that scan across sequence positions.

.. autofunction:: poolparty.insertion_scan

.. autofunction:: poolparty.deletion_scan

.. autofunction:: poolparty.replacement_scan

.. autofunction:: poolparty.shuffle_scan

.. autofunction:: poolparty.mutagenize_scan

.. autofunction:: poolparty.subseq_scan

Region Operations
-----------------

Operations for working with tagged sequence regions.

.. autofunction:: poolparty.insert_tags

.. autofunction:: poolparty.remove_tags

.. autofunction:: poolparty.extract_region

.. autofunction:: poolparty.replace_region

.. autofunction:: poolparty.apply_at_region

.. autofunction:: poolparty.region_scan

.. autofunction:: poolparty.region_multiscan

Multiscan Operations
--------------------

Multi-region scanning operations.

.. autofunction:: poolparty.deletion_multiscan

.. autofunction:: poolparty.insertion_multiscan

.. autofunction:: poolparty.replacement_multiscan

State Operations
----------------

Operations that manipulate the state space of pools.

.. autofunction:: poolparty.stack

.. autofunction:: poolparty.repeat

.. autofunction:: poolparty.state_slice

.. autofunction:: poolparty.state_shuffle

.. autofunction:: poolparty.sample

.. autofunction:: poolparty.sync

ORF Operations
--------------

Codon-aware operations for protein-coding sequences.

.. autofunction:: poolparty.mutagenize_orf

Library Generation
------------------

.. autofunction:: poolparty.generate_library

Utility Functions
-----------------

.. autofunction:: poolparty.print_named_colors

Constants
---------

DNA Constants
~~~~~~~~~~~~~

.. py:data:: poolparty.BASES

   Standard DNA bases: ``['A', 'C', 'G', 'T']``

.. py:data:: poolparty.COMPLEMENT

   Complement mapping for DNA bases.

.. py:data:: poolparty.IUPAC_TO_DNA

   Mapping from IUPAC ambiguity codes to DNA bases.

.. py:data:: poolparty.VALID_CHARS

   Set of valid characters in DNA sequences.

.. py:data:: poolparty.IGNORE_CHARS

   Characters to ignore in DNA sequences (gaps, annotations).

Operation Classes
-----------------

These are the underlying operation classes. Most users will use the convenience
functions above instead of instantiating these directly.

Base Operation Classes
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: poolparty.FromSeqsOp
   :show-inheritance:

.. autoclass:: poolparty.FromIupacOp
   :show-inheritance:

.. autoclass:: poolparty.FromMotifOp
   :show-inheritance:

.. autoclass:: poolparty.GetKmersOp
   :show-inheritance:

.. autoclass:: poolparty.MutagenizeOp
   :show-inheritance:

.. autoclass:: poolparty.SeqShuffleOp
   :show-inheritance:

.. autoclass:: poolparty.RecombineOp
   :show-inheritance:

Fixed Operation Classes
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: poolparty.FixedOp
   :show-inheritance:

.. autoclass:: poolparty.StylizeOp
   :show-inheritance:

State Operation Classes
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: poolparty.StackOp
   :show-inheritance:

.. autoclass:: poolparty.RepeatOp
   :show-inheritance:

.. autoclass:: poolparty.StateSliceOp
   :show-inheritance:

.. autoclass:: poolparty.StateShuffleOp
   :show-inheritance:

.. autoclass:: poolparty.SampleOp
   :show-inheritance:

ORF Operation Classes
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: poolparty.MutagenizeOrfOp
   :show-inheritance:
