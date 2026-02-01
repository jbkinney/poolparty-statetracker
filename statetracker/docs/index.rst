StateTracker Documentation
==========================

**StateTracker** is a Python library for creating composable states with
unidirectional value propagation. It provides a powerful way to enumerate
combinatorial spaces through state algebra operations.

Why StateTracker?
-----------------

StateTracker was developed to support the design of complex DNA sequence
libraries (see `PoolParty <https://github.com/jkinney/poolparty>`_), but it
solves a general problem: **random access to combinatorial spaces**.

If you've ever written nested loops to enumerate a Cartesian product and then
wished you could shuffle the order, sample a subset, or split into train/test
sets—all while tracking which component indices correspond to each item—
StateTracker is for you. Build your combinatorial structure once using state
algebra, and StateTracker handles the index math automatically.

See the :doc:`motivation` page for a detailed explanation of the problem
StateTracker solves.

.. code-block:: python

    from statetracker import State, Manager, product

    with Manager():
        A = State(num_values=2, name='A')
        B = State(num_values=3, name='B')
        C = product([A, B])  # Cartesian product: 6 values

        for value in C:
            print(f"C={value}, A={A.value}, B={B.value}")

Output:

.. code-block:: text

    C=0, A=0, B=0
    C=1, A=1, B=0
    C=2, A=0, B=1
    C=3, A=1, B=1
    C=4, A=0, B=2
    C=5, A=1, B=2

Features
--------

**Composable States**
    Build complex iteration patterns from simple states using algebraic operations.

**Unidirectional Value Propagation**
    Set a derived state's value and all parent states update automatically.

**Rich Operation Set**
    Product, stack, slice, repeat, shuffle, sample, split, interleave, and more.

**Conflict Detection**
    Automatic detection of conflicting value assignments in the state DAG.

**Visualization**
    Built-in ASCII tree visualization for debugging state relationships.

Installation
------------

Install from PyPI:

.. code-block:: bash

    pip install statetracker

Or install from source:

.. code-block:: bash

    git clone https://github.com/jkinney/statetracker.git
    cd statetracker
    pip install -e .

Quick Example
-------------

Create states and combine them to enumerate a combinatorial space:

.. code-block:: python

    from statetracker import State, Manager, stack, shuffle

    with Manager():
        # Create leaf states
        control = State(num_values=5, name='control')
        treatment = State(num_values=5, name='treatment')

        # Combine with stack (disjoint union)
        samples = stack([control, treatment])  # 10 values

        # Shuffle for randomization
        randomized = shuffle(samples, seed=42)

        # Iterate through all samples
        for value in randomized:
            if control.is_active:
                print(f"Control sample {control.value}")
            else:
                print(f"Treatment sample {treatment.value}")

Output:

.. code-block:: text

    Treatment sample 2
    Control sample 3
    Control sample 2
    Treatment sample 3
    Treatment sample 0
    Treatment sample 1
    Treatment sample 4
    Control sample 4
    Control sample 0
    Control sample 1

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   motivation
   quickstart
   concepts
   operations
   alternatives

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
