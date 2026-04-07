Stack (Disjoint Union)
======================

The **stack** operation creates a state that iterates through each parent state
sequentially --- a disjoint union. Only one parent is active at a time.

.. code-block:: python

    from statetracker import Manager, State, stack

    with Manager():
        A = State(num_values=2, name="A")
        B = State(num_values=3, name="B")

        C = stack([A, B])  # 5 values (2 + 3)

        for value in C:
            print(f"C={value}, A={A.value}, B={B.value}")

.. code-block:: text

    C=0, A=0, B=None
    C=1, A=1, B=None
    C=2, A=None, B=0
    C=3, A=None, B=1
    C=4, A=None, B=2

Notice that when A is active, B is ``None`` (inactive), and vice versa.

Because ``stack`` only activates one parent at a time, it does **not** cause
:class:`~statetracker.ConflictingValueAssignmentError` even when the stacked
states share a common ancestor.
