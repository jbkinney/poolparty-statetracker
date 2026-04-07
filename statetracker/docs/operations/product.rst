Product (Cartesian Product)
===========================

The **product** operation creates a state whose values enumerate all
combinations of parent state values --- the Cartesian product.


``product()``
-------------

Use the ``product()`` function to create a Cartesian product:

.. code-block:: python

    from statetracker import Manager, State, product

    with Manager():
        A = State(num_values=2, name="A")
        B = State(num_values=3, name="B")

        C = product([A, B])  # 6 values (2 x 3)

        for _ in C:
            print(f"A={A.value}, B={B.value}")

.. code-block:: text

    A=0, B=0
    A=1, B=0
    A=0, B=1
    A=1, B=1
    A=0, B=2
    A=1, B=2

The first state (A) varies fastest, cycling through all its values before the
second state (B) advances.

You can combine more than two states:

.. code-block:: python

    with Manager():
        A = State(num_values=2, name="A")
        B = State(num_values=2, name="B")
        C = State(num_values=2, name="C")

        D = product([A, B, C], name="D")  # 8 values
        print(f"D has {D.num_values} values")

.. code-block:: text

    D has 8 values

.. note::

    ``product()`` does not allow duplicate states. Use ``ordered_product()`` if
    you need automatic deduplication.


``ordered_product()``
---------------------

The ``ordered_product()`` function automatically:

- Removes duplicate states
- Flattens nested products
- Orders states by ``iter_order`` and ``id``

.. code-block:: python

    from statetracker import ordered_product

    with Manager():
        A = State(num_values=2, name="A")
        B = State(num_values=3, name="B")

        # Duplicates are automatically removed
        C = ordered_product([A, B, A])  # Same as ordered_product([A, B])
        print(f"C has {C.num_values} values")

.. code-block:: text

    C has 6 values


``set_product_order_mode()``
----------------------------

Control the ordering with ``set_product_order_mode()``:

.. code-block:: python

    from statetracker import set_product_order_mode

    set_product_order_mode("first_state_fastest")  # Default behavior
    # set_product_order_mode("first_state_slowest")  # Reverse ordering
