StateCounter Documentation
==========================

**StateCounter** is a Python library for creating composable counters with
unidirectional state propagation. It provides a powerful way to enumerate
combinatorial spaces through counter algebra operations like products, sums,
slices, and more.

Features
--------

- **Composable Counters**: Build complex iteration patterns from simple counters
- **Unidirectional State Propagation**: Set child state and parent states update automatically
- **Counter Algebra**: Product (×), sum (+), slice, repeat, shuffle, and synchronize operations
- **Conflict Detection**: Automatic detection of conflicting state assignments
- **Tree Visualization**: Built-in ASCII tree visualization for debugging

Quick Start
-----------

.. code-block:: python

    from statecounter import Counter, Manager

    with Manager():
        # Create leaf counters
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        
        # Combine with product (Cartesian product)
        C = A * B  # 6 states total
        
        # Iterate and see parent states update
        for state in C:
            print(f"C={state}, A={A.state}, B={B.state}")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
