Split
=====

The **split** operation divides a state into multiple sub-states.


Equal Split
-----------

Split into N roughly equal parts:

.. code-block:: python

    from statetracker import Manager, State, split

    with Manager():
        A = State(num_values=10, name="A")

        # Split into 3 parts: sizes 4, 3, 3
        B, C, D = split(A, 3)

        print(f"B: {B.num_values} values")
        print(f"C: {C.num_values} values")
        print(f"D: {D.num_values} values")

.. code-block:: text

    B: 4 values
    C: 3 values
    D: 3 values


Proportional Split
------------------

Split according to proportions:

.. code-block:: python

    with Manager():
        A = State(num_values=100, name="A")

        # Split 80/20
        train, test = split(A, [0.8, 0.2], names=["train", "test"])

        print(f"train: {train.num_values} values")
        print(f"test: {test.num_values} values")

.. code-block:: text

    train: 80 values
    test: 20 values
