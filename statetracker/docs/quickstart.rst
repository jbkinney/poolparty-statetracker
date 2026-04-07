Quick Start Guide
=================

This guide provides a quick introduction to StateTracker, a Python library for
creating composable states with unidirectional value propagation.


Installation
------------

Install from PyPI:

.. code-block:: bash

    pip install statetracker


Basic Usage
-----------

All states must be created within a ``Manager`` context:

.. code-block:: python

    from statetracker import Manager, State

    with Manager():
        A = State(num_values=3, name="A")
        print(list(A))

.. code-block:: text

    [0, 1, 2]


State Operations
----------------

Product (Cartesian Product)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``product`` to create Cartesian products:

.. code-block:: python

    from statetracker import Manager, State, product

    with Manager():
        A = State(num_values=2, name="A")
        B = State(num_values=3, name="B")
        C = product([A, B], name="C")  # 6 values

        for _ in C:
            C.print_states()

.. code-block:: text

    C=0, A=0, B=0
    C=1, A=1, B=0
    C=2, A=0, B=1
    C=3, A=1, B=1
    C=4, A=0, B=2
    C=5, A=1, B=2


Stack (Disjoint Union)
~~~~~~~~~~~~~~~~~~~~~~

Use ``stack`` to create disjoint unions where only one state is active at a
time:

.. code-block:: python

    from statetracker import Manager, State, stack

    with Manager():
        A = State(num_values=2, name="A")
        B = State(num_values=3, name="B")
        C = stack([A, B], name="C")  # 5 values

        for _ in C:
            # Only one of A or B is active at a time (the other is None)
            C.print_states()

.. code-block:: text

    C=0, A=0, B=None
    C=1, A=1, B=None
    C=2, A=None, B=0
    C=3, A=None, B=1
    C=4, A=None, B=2


Slicing
~~~~~~~

Use Python slice syntax to select subsets of values:

.. code-block:: python

    from statetracker import Manager, State

    with Manager():
        A = State(num_values=10, name="A")
        B = A[2:5].named("B")   # Values 2, 3, 4
        C = A[::-1].named("C")  # Reversed: 9, 8, 7, ..., 0

        print(f"A has {A.num_values} values")
        print(f"\nB (slice [2:5]) has {B.num_values} values: {list(B)}")
        for _ in B:
            B.print_states()

        print(f"\nC (reversed) has {C.num_values} values: {list(C)}")
        for _ in C:
            C.print_states()

.. code-block:: text

    A has 10 values

    B (slice [2:5]) has 3 values: [0, 1, 2]
    B=0, A=2
    B=1, A=3
    B=2, A=4

    C (reversed) has 10 values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    C=0, A=9
    C=1, A=8
    C=2, A=7
    C=3, A=6
    C=4, A=5
    C=5, A=4
    C=6, A=3
    C=7, A=2
    C=8, A=1
    C=9, A=0


Value Propagation
-----------------

StateTracker uses unidirectional value propagation. Setting a child state's
value automatically updates all parent states:

.. code-block:: python

    from statetracker import Manager, State, product

    with Manager():
        A = State(num_values=2, name="A")
        B = State(num_values=3, name="B")
        C = product([A, B])

        C.value = 5  # Set child value
        print("After setting C.value = 5:")
        print(f"A.value = {A.value}")  # 1 (automatically updated)
        print(f"B.value = {B.value}")  # 2 (automatically updated)

.. code-block:: text

    After setting C.value = 5:
    A.value = 1
    B.value = 2


Inspecting States
-----------------

Use ``get_states()`` and ``print_states()`` to conveniently view all state
values at once:

.. code-block:: python

    from statetracker import Manager, State, product

    with Manager():
        A = State(num_values=2, name="A")
        B = State(num_values=3, name="B")
        C = product([A, B], name="C")

        C.value = 5
        C.print_states()  # Print all values in one line

        values = C.get_states()  # Get values as a dictionary
        print(values)

.. code-block:: text

    C=5, A=1, B=2
    {'C': 5, 'A': 1, 'B': 2}


Visualization
-------------

Use ``print_dag()`` to visualize state dependencies:

.. code-block:: python

    from statetracker import Manager, State, product

    with Manager():
        A = State(num_values=2, name="A")
        B = State(num_values=3, name="B")
        C = product([A, B], name="C")

        C.print_dag()

.. code-block:: text

    C (counter, io=0, n=6)
    +-- [op=Product]
        +-- A (counter, io=0, n=2)
        +-- B (counter, io=0, n=3)


Next Steps
----------

- See :doc:`concepts` for a deeper understanding of state algebra
- Explore the :doc:`operations/index` reference for all available operations
- Check the :doc:`api` for detailed function documentation
