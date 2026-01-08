Quick Start Guide
=================

Installation
------------

Install from PyPI:

.. code-block:: bash

    pip install statecounter

Basic Usage
-----------

All counters must be created within a ``Manager`` context:

.. code-block:: python

    from statecounter import Counter, Manager

    with Manager():
        A = Counter(num_states=3, name='A')
        print(list(A))  # [0, 1, 2]

Counter Operations
------------------

Product (Cartesian Product)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``*`` or ``product_counters()`` to create Cartesian products:

.. code-block:: python

    from statecounter import Counter, Manager

    with Manager():
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        C = A * B  # 6 states
        
        for _ in C:
            print(f"A={A.state}, B={B.state}")

Stack
~~~~~~~~~~~~~~~~~~~~

Use ``+`` or ``sum_counters()`` to create disjoint unions:

.. code-block:: python

    with Manager():
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        C = stack([A,B])  # 5 states
        
        for _ in C:
            # Only one of A or B is active at a time
            print(f"A={A.state}, B={B.state}")

Slicing
~~~~~~~

Use Python slice syntax to select subsets of states:

.. code-block:: python

    with Manager():
        A = Counter(num_states=10, name='A')
        B = A[2:5]  # States 2, 3, 4
        C = A[::-1]  # Reversed: 9, 8, 7, ..., 0

State Propagation
-----------------

StateCounter uses unidirectional state propagation. Setting a child counter's
state automatically updates all parent counters:

.. code-block:: python

    with Manager():
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        C = A * B
        
        C.state = 5  # Set child state
        print(A.state)  # 1 (automatically updated)
        print(B.state)  # 2 (automatically updated)

Visualization
-------------

Use ``print_dag()`` to visualize counter dependencies:

.. code-block:: python

    with Manager():
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        C = A * B
        C.name = 'C'
        
        C.print_dag()
        # Output:
        # C [Multiply, n=6]
        # ├── A [Leaf, n=2]
        # └── B [Leaf, n=3]
