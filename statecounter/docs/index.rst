StateCounter Documentation
==========================

**StateCounter** is a Python library for creating composable counters with
unidirectional state propagation. It provides a powerful way to enumerate
combinatorial spaces through counter algebra operations.

.. code-block:: python

    from statecounter import Counter, Manager, product

    with Manager():
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        C = product([A, B])  # Cartesian product: 6 states
        
        for state in C:
            print(f"C={state}, A={A.state}, B={B.state}")

Features
--------

**Composable Counters**
    Build complex iteration patterns from simple counters using algebraic operations.

**Unidirectional State Propagation**
    Set a derived counter's state and all parent counters update automatically.

**Rich Operation Set**
    Product, stack, slice, repeat, shuffle, sample, split, interleave, and more.

**Conflict Detection**
    Automatic detection of conflicting state assignments in the counter DAG.

**Visualization**
    Built-in ASCII tree visualization for debugging counter relationships.

Installation
------------

Install from PyPI:

.. code-block:: bash

    pip install statecounter

Or install from source:

.. code-block:: bash

    git clone https://github.com/jkinney/statecounter.git
    cd statecounter
    pip install -e .

Quick Example
-------------

Create counters and combine them to enumerate a combinatorial space:

.. code-block:: python

    from statecounter import Counter, Manager, stack, shuffle

    with Manager():
        # Create leaf counters
        control = Counter(num_states=5, name='control')
        treatment = Counter(num_states=5, name='treatment')
        
        # Combine with stack (disjoint union)
        samples = stack([control, treatment])  # 10 states
        
        # Shuffle for randomization
        randomized = shuffle(samples, seed=42)
        
        # Iterate through all samples
        for state in randomized:
            if control.is_active():
                print(f"Control sample {control.state}")
            else:
                print(f"Treatment sample {treatment.state}")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   concepts
   operations

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
