State Operations
================

State operations affect which sequences are in a Pool and how they are
ordered, without directly modifying sequence content. They include methods
to replicate, sample, slice, reorder, synchronize, filter, score, and
eagerly evaluate Pool contents.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`repeat`
     - Repeat the state space of a pool N times (also available as ``*``).
   * - :doc:`sample`
     - Draw a fixed number of sequences, optionally with a reproducibility seed.
   * - :doc:`slice_states`
     - Retain a contiguous slice of the state space.
   * - :doc:`shuffle_states`
     - Randomly reorder the state space.
   * - :doc:`sync`
     - Synchronize multiple pools to iterate in lockstep (in-place).
   * - :doc:`filter`
     - Retain only sequences satisfying a predicate function.
   * - :doc:`score`
     - Evaluate a function on each sequence and record the result as a design card column.
   * - :doc:`materialize`
     - Eagerly generate sequences and cache them in a new standalone pool.

.. toctree::
   :hidden:

   repeat (*) <repeat>
   sample
   slice_states
   shuffle_states
   sync
   filter
   score
   materialize
