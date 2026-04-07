Sample
======

The **sample** operation creates a state with sampled values from the parent.
This is useful for creating random subsets.

.. code-block:: python

    from statetracker import Manager, State, sample

    with Manager():
        A = State(num_values=100, name="A")

        # Sample 10 values with replacement
        B = sample(A, num_values=10, seed=42)

        # Sample 10 values without replacement
        C = sample(A, num_values=10, seed=42, with_replacement=False)

        print(f"B (with replacement) has {B.num_values} values")
        print(f"C (without replacement) has {C.num_values} values")

.. code-block:: text

    B (with replacement) has 10 values
    C (without replacement) has 10 values

You can also provide explicit values to sample:

.. code-block:: python

    with Manager():
        A = State(num_values=10, name="A")

        # Explicit values to sample
        B = sample(A, sampled_states=[0, 2, 4, 6, 8])

        for value in B:
            print(f"B={value}, A={A.value}")

.. code-block:: text

    B=0, A=0
    B=1, A=2
    B=2, A=4
    B=3, A=6
    B=4, A=8
