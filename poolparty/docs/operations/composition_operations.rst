Composition Operations
======================

Composition operations combine sequences from multiple Pools into a single
Pool. ``join`` concatenates parent sequences end-to-end into a composite
sequence, while ``stack`` merges multiple Pools into one Pool whose state
space is the disjoint union of all inputs.

.. code-block:: python

    import poolparty as pp
    pp.init()

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`join`
     - Concatenate a list of pools end-to-end into a single composite pool.
   * - :doc:`stack`
     - Combine multiple pools by stacking their state spaces (also available as ``+``).

.. toctree::
   :hidden:

   join
   stack (+) <stack>
