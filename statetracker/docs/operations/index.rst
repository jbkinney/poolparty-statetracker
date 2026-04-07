Operations
==========

StateTracker provides composable operations for building complex state
structures from simple leaf states. Each operation returns a new
:class:`~statetracker.State`, so calls can be chained into declarative
pipelines. All examples assume:

.. code-block:: python

    from statetracker import Manager, State

.. toctree::
   :hidden:

   Product <product>
   Stack <stack>
   Sync <sync>
   Slice <slice>
   Repeat <repeat>
   Shuffle <shuffle>
   Sample <sample>
   Split <split>
   Interleave <interleave>

----

.. rubric:: Composition Operations

Combine multiple states into a single derived state.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operation
     - Description
   * - :doc:`product`
     - Cartesian product of states (values multiply)
   * - :doc:`stack`
     - Disjoint union of states (values add)
   * - :doc:`interleave`
     - Round-robin alternation between states

----

.. rubric:: Transformation Operations

Create a new state by transforming an existing state's values.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operation
     - Description
   * - :doc:`slice`
     - Select a subset of values using Python slice syntax
   * - :doc:`repeat`
     - Cycle through a state's values multiple times
   * - :doc:`shuffle`
     - Random permutation of values
   * - :doc:`sample`
     - Random subset of values (with or without replacement)
   * - :doc:`split`
     - Divide a state into multiple sub-states

----

.. rubric:: Synchronization Operations

Link states so they track each other's values.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operation
     - Description
   * - :doc:`sync`
     - Synchronize two states and create mirrored aliases
