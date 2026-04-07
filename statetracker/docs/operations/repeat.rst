Repeat
======

The **repeat** operation creates a state that cycles through the parent's
values multiple times.

.. code-block:: python

    from statetracker import Manager, State, repeat

    with Manager():
        A = State(num_values=3, name="A")

        B = repeat(A, times=2)  # 6 values

        for value in B:
            print(f"B={value}, A={A.value}")

.. code-block:: text

    B=0, A=0
    B=1, A=1
    B=2, A=2
    B=3, A=0
    B=4, A=1
    B=5, A=2
