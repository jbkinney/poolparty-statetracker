Interleave
==========

The **interleave** operation creates a state that alternates between parent
states' values in a round-robin fashion.

.. code-block:: python

    from statetracker import Manager, State, interleave

    with Manager():
        A = State(num_values=3, name="A")
        B = State(num_values=3, name="B")

        C = interleave([A, B])  # 6 values

        for value in C:
            print(f"C={value}, A={A.value}, B={B.value}")

.. code-block:: text

    C=0, A=0, B=None
    C=1, A=None, B=0
    C=2, A=1, B=None
    C=3, A=None, B=1
    C=4, A=2, B=None
    C=5, A=None, B=2

The values alternate: A's value 0, B's value 0, A's value 1, B's value 1, etc.

.. warning::

    All states passed to ``interleave()`` must have the same ``num_values``.


Combining Operations
--------------------

Operations can be freely combined to create complex iteration patterns:

.. code-block:: python

    from statetracker import Manager, State, stack, shuffle, split

    with Manager():
        # Create base states for different conditions
        control = State(num_values=10, name="control")
        treatment = State(num_values=10, name="treatment")

        # Stack them (20 values total)
        all_samples = stack([control, treatment])

        # Shuffle for randomization
        randomized = shuffle(all_samples, seed=42)

        # Split into train/test
        train, test = split(randomized, [0.8, 0.2])

        print(f"Train set: {train.num_values} samples")
        print(f"Test set: {test.num_values} samples")

.. code-block:: text

    Train set: 16 samples
    Test set: 4 samples
