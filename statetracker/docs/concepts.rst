Core Concepts
=============

This guide explains the fundamental concepts behind StateTracker's design.


.. _concepts-states:

States
------

A **State** is an object that can take on a finite number of discrete values,
numbered from 0 to ``num_values - 1``. The simplest state is a "leaf" state
created directly with a specified number of values:

.. code-block:: python

    from statetracker import Manager, State

    with Manager():
        A = State(num_values=5, name="A")
        print(list(A))

.. code-block:: text

    [0, 1, 2, 3, 4]

States can be **iterated** to cycle through all their values, and their current
value can be read or set via the ``value`` property.


The Manager Context
-------------------

All states must be created within a ``Manager`` context. The Manager tracks all
states and their relationships:

.. code-block:: python

    with Manager() as mgr:
        A = State(num_values=3, name="A")
        B = State(num_values=2, name="B")

        # Manager tracks all states
        print(mgr.get_all_names())

.. code-block:: text

    ['A', 'B']

Creating states outside a Manager context raises a ``RuntimeError``.


State Composition
-----------------

The power of StateTracker comes from **composing** states using operations.
When you combine states, you create a new "derived" state that depends on its
"parent" states:

.. code-block:: python

    from statetracker import product

    with Manager():
        A = State(num_values=2, name="A")
        B = State(num_values=3, name="B")

        # C is derived from A and B via product
        C = product([A, B], name="C")  # 6 values (2 x 3)

        print(f"C has {C.num_values} values")

.. code-block:: text

    C has 6 values

This creates a **directed acyclic graph (DAG)** of state dependencies. You can
visualize this structure using ``print_dag()``:

.. code-block:: python

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


Unidirectional Value Propagation
--------------------------------

StateTracker uses **unidirectional value propagation**: when you set the value
of a derived state, the values of all its parent states are automatically
computed and updated.

.. code-block:: python

    with Manager():
        A = State(num_values=2, name="A")
        B = State(num_values=3, name="B")
        C = product([A, B])

        C.value = 5  # Set the derived state's value

        # Parent values are automatically computed
        print(f"A.value = {A.value}")  # 1
        print(f"B.value = {B.value}")  # 2

.. code-block:: text

    A.value = 1
    B.value = 2

This is the key insight: you iterate over the **derived** state, and the
**parent** states automatically track along.


Active vs Inactive States
--------------------------

A state's value can be either:

- **Active**: An integer from 0 to ``num_values - 1``
- **Inactive**: ``None``

Inactive states arise with operations like ``stack`` where only one parent is
"active" at a time:

.. code-block:: python

    from statetracker import stack

    with Manager():
        A = State(num_values=2, name="A")
        B = State(num_values=3, name="B")
        C = stack([A, B])  # 5 values total

        for value in C:
            print(f"C={value}, A={A.value}, B={B.value}")

.. code-block:: text

    C=0, A=0, B=None
    C=1, A=1, B=None
    C=2, A=None, B=0
    C=3, A=None, B=1
    C=4, A=None, B=2

Use ``is_active`` to check if a state is active. You can also use
``print_states()`` to conveniently display all values at once, or
``print_states(include_inactive=False)`` to show only active states.


Conflict Detection
------------------

When a state appears in multiple branches of the DAG, StateTracker detects
**conflicting value assignments**. This happens when two different paths would
assign different values to the same parent state.

.. code-block:: python

    with Manager():
        A = State(num_values=3, name="A")
        B = A[0:2]  # B=0 -> A=0, B=1 -> A=1
        C = A[1:3]  # C=0 -> A=1, C=1 -> A=2

        # Using product tries to activate BOTH B and C simultaneously
        D = product([B, C])

        # D value 0 means B=0 (A=0) and C=0 (A=1)
        # But A can't be both 0 and 1 at the same time!
        try:
            for value in D:
                pass
        except Exception as e:
            print(f"Error: {type(e).__name__}")

.. code-block:: text

    Error: ConflictingValueAssignmentError

Note that ``stack`` does NOT cause conflicts because it only activates one
parent at a time. The conflict arises when an operation like ``product`` tries
to activate multiple states that share a common ancestor with incompatible
value requirements.


Iteration Order and ``ordered_product()``
-----------------------------------------

Sometimes you want certain states to have priority over others when they appear
together in Cartesian products. The ``ordered_product()`` function reads an
``iter_order`` property from each state to determine ordering. States with
lower ``iter_order`` values iterate "faster" (change more frequently in the
inner loop).

.. code-block:: python

    from statetracker import ordered_product

    with Manager():
        A = State(num_values=2, name="A", iter_order=0)  # Fast
        B = State(num_values=2, name="B", iter_order=1)  # Slow

        C = ordered_product([A, B])

        for _ in C:
            print(f"A={A.value}, B={B.value}")

.. code-block:: text

    A=0, B=0
    A=1, B=0
    A=0, B=1
    A=1, B=1

The default ``iter_order`` is 0 for leaf states. Derived states inherit the
minimum ``iter_order`` of their parents.

Note that the regular ``product()`` function does not use ``iter_order`` --- it
preserves the exact order of states you pass to it.


State Identity
--------------

Each state has a unique ``id`` assigned by the Manager. This is used for:

- Deduplication in ``ordered_product()``
- Tie-breaking when states have the same ``iter_order``

States are compared by identity (``is``), not value, so two states with the
same ``num_values`` are still distinct objects.


Copying States
--------------

States support two types of copying:

- ``copy()``: Creates a new state with the same parents (shallow copy)
- ``deepcopy()``: Recursively creates a new state with copied ancestors

.. code-block:: python

    with Manager():
        A = State(num_values=3, name="A")
        B = A[1:3]

        # Shallow copy: C shares parent A with B
        C = B.copy(name="C")

        # Deep copy: D has its own copy of the parent
        D = B.deepcopy(name="D")

        print("B's DAG:")
        B.print_dag()
        print("\nC's DAG (shallow copy - shares A):")
        C.print_dag()
        print("\nD's DAG (deep copy - has own parent):")
        D.print_dag()

.. code-block:: text

    B's DAG:
    Counter[1] (counter, io=0, n=2)
    +-- [op=Slice]
        +-- A (counter, io=0, n=3)

    C's DAG (shallow copy - shares A):
    C (counter, io=0, n=2)
    +-- [op=Slice]
        +-- A (counter, io=0, n=3)

    D's DAG (deep copy - has own parent):
    D (counter, io=0, n=2)
    +-- [op=Slice]
        +-- Counter[3] (counter, io=0, n=3)
