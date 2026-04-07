Synchronize
===========

StateTracker provides two synchronization mechanisms: ``sync()`` for linking
two existing states, and ``synced_to()`` for creating a mirrored alias.


``sync()``
----------

The ``sync()`` function synchronizes two existing states so they keep the same
value. Both states must have the same ``num_values``.

.. code-block:: python

    from statetracker import Manager, State, sync

    with Manager():
        A = State(num_values=4, name="A")
        B = State(num_values=4, name="B")

        sync(A, B)  # A and B are now linked

        for _ in A:
            print(f"A={A.value}, B={B.value}")

.. code-block:: text

    A=0, B=0
    A=1, B=1
    A=2, B=2
    A=3, B=3

.. warning::

    Both states passed to ``sync()`` must have the same ``num_values``.


``synced_to()``
---------------

The ``synced_to()`` function creates a new state that mirrors its parent
exactly. This is useful for creating an alias or checkpoint in the state DAG.

.. code-block:: python

    from statetracker import Manager, State, synced_to

    with Manager():
        A = State(num_values=5, name="A")

        B = synced_to(A, name="B")  # B mirrors A exactly

        for value in B:
            print(f"B={value}, A={A.value}")

.. code-block:: text

    B=0, A=0
    B=1, A=1
    B=2, A=2
    B=3, A=3
    B=4, A=4


``State.synced_parent()``
-------------------------

The ``synced_parent()`` method on a State provides the same functionality as
``synced_to()`` as a convenience method:

.. code-block:: python

    from statetracker import Manager, State, product

    with Manager():
        a = State(num_values=4, name="a")
        b = State(num_values=3, name="b")
        c = product([a, b], name="c")

        # Create synced parents using either approach
        d = synced_to(c).named("d")
        e = c.synced_parent("e")

        c.print_dag()

.. code-block:: text

    c (counter, io=0, n=12)
    +-- [op=Product]
        +-- a (counter, io=0, n=4)
        +-- b (counter, io=0, n=3)
