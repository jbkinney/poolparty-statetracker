Slice
=====

The **slice** operation selects a subset of values from a parent state, similar
to Python list slicing.


Using Index Notation
--------------------

The most convenient way is Python's slice syntax:

.. code-block:: python

    from statetracker import Manager, State

    with Manager():
        A = State(num_values=10, name="A")

        B = A[2:5]   # Values 2, 3, 4
        C = A[::2]   # Even values: 0, 2, 4, 6, 8
        D = A[::-1]  # Reversed: 9, 8, 7, ..., 0
        E = A[3]     # Single value: 3

        print(f"B (A[2:5]) has {B.num_values} values")
        print(f"C (A[::2]) has {C.num_values} values")
        print(f"D (A[::-1]) has {D.num_values} values")
        print(f"E (A[3]) has {E.num_values} values")

.. code-block:: text

    B (A[2:5]) has 3 values
    C (A[::2]) has 5 values
    D (A[::-1]) has 10 values
    E (A[3]) has 1 values


Using ``slice()``
-----------------

The ``slice()`` function provides the same functionality:

.. code-block:: python

    from statetracker import slice

    with Manager():
        A = State(num_values=10, name="A")

        B = slice(A, start=2, stop=5)  # Values 2, 3, 4
        C = slice(A, step=2)           # Even values

        print(f"B has {B.num_values} values")
        print(f"C has {C.num_values} values")

.. code-block:: text

    B has 3 values
    C has 5 values
