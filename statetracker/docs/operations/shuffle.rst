Shuffle
=======

The **shuffle** operation creates a state with randomly permuted values. The
shuffle is deterministic when a seed is provided.

.. code-block:: python

    from statetracker import Manager, State, shuffle

    with Manager():
        A = State(num_values=5, name="A")

        # Random permutation with seed for reproducibility
        B = shuffle(A, seed=42)

        for value in B:
            print(f"B={value}, A={A.value}")

.. code-block:: text

    B=0, A=3
    B=1, A=1
    B=2, A=2
    B=3, A=4
    B=4, A=0

You can also provide an explicit permutation:

.. code-block:: python

    with Manager():
        A = State(num_values=4, name="A")

        # Explicit permutation: reverse order
        B = shuffle(A, permutation=[3, 2, 1, 0])

        for value in B:
            print(f"B={value}, A={A.value}")

.. code-block:: text

    B=0, A=3
    B=1, A=2
    B=2, A=1
    B=3, A=0
