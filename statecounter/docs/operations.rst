Operations Guide
================

This guide provides detailed explanations and examples for each counter operation
available in StateCounter.

.. contents:: On this page
   :local:
   :depth: 2

Product (Cartesian Product)
---------------------------

The **product** operation creates a counter whose states enumerate all
combinations of parent counter states—the Cartesian product.

Using the ``*`` Operator
~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to create a product is with the ``*`` operator:

.. code-block:: python

    from statecounter import Counter, Manager

    with Manager():
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        
        C = A * B  # 6 states (2 × 3)
        
        for _ in C:
            print(f"A={A.state}, B={B.state}")

Output::

    A=0, B=0
    A=1, B=0
    A=0, B=1
    A=1, B=1
    A=0, B=2
    A=1, B=2

The first counter (A) varies fastest, cycling through all its states before
the second counter (B) advances.

Using ``product()``
~~~~~~~~~~~~~~~~~~~

The :func:`~statecounter.product` function provides explicit control:

.. code-block:: python

    from statecounter import Counter, Manager, product

    with Manager():
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=2, name='B')
        C = Counter(num_states=2, name='C')
        
        D = product([A, B, C], name='D')  # 8 states

.. note::

    ``product()`` does not allow duplicate counters. Use ``ordered_product()``
    if you need automatic deduplication.

Using ``ordered_product()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~statecounter.ordered_product` function automatically:

- Removes duplicate counters
- Flattens nested products
- Orders counters by ``iter_order`` and ``id``

.. code-block:: python

    from statecounter import Counter, Manager, ordered_product

    with Manager():
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        
        # Duplicates are automatically removed
        C = ordered_product([A, B, A])  # Same as ordered_product([A, B])

Control the ordering with :func:`~statecounter.set_product_order_mode`:

.. code-block:: python

    from statecounter import set_product_order_mode

    set_product_order_mode('first_counter_fastest')  # Default behavior
    set_product_order_mode('first_counter_slowest')  # Reverse ordering

Stack (Disjoint Union)
----------------------

The **stack** operation creates a counter that iterates through each parent
counter sequentially—a disjoint union. Only one parent is active at a time.

.. code-block:: python

    from statecounter import Counter, Manager, stack

    with Manager():
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        
        C = stack([A, B])  # 5 states (2 + 3)
        
        for state in C:
            print(f"C={state}, A={A.state}, B={B.state}")

Output::

    C=0, A=0, B=None
    C=1, A=1, B=None
    C=2, A=None, B=0
    C=3, A=None, B=1
    C=4, A=None, B=2

Notice that when A is active, B is ``None`` (inactive), and vice versa.

Synchronize
-----------

The **sync** operation creates a counter that keeps multiple parent counters
in lockstep—they all have the same state value.

.. code-block:: python

    from statecounter import Counter, Manager, sync

    with Manager():
        A = Counter(num_states=4, name='A')
        B = Counter(num_states=4, name='B')
        
        C = sync([A, B])  # 4 states
        
        for _ in C:
            print(f"A={A.state}, B={B.state}")

Output::

    A=0, B=0
    A=1, B=1
    A=2, B=2
    A=3, B=3

.. warning::

    All counters passed to ``sync()`` must have the same ``num_states``.

Slice
-----

The **slice** operation selects a subset of states from a parent counter,
similar to Python list slicing.

Using Index Notation
~~~~~~~~~~~~~~~~~~~~

The most convenient way is Python's slice syntax:

.. code-block:: python

    from statecounter import Counter, Manager

    with Manager():
        A = Counter(num_states=10, name='A')
        
        B = A[2:5]   # States 2, 3, 4
        C = A[::2]   # Even states: 0, 2, 4, 6, 8
        D = A[::-1]  # Reversed: 9, 8, 7, ..., 0
        E = A[3]     # Single state: 3

Using ``slice()``
~~~~~~~~~~~~~~~~~

The :func:`~statecounter.slice` function provides the same functionality:

.. code-block:: python

    from statecounter import Counter, Manager, slice

    with Manager():
        A = Counter(num_states=10, name='A')
        
        B = slice(A, start=2, stop=5)  # States 2, 3, 4
        C = slice(A, step=2)           # Even states

Repeat
------

The **repeat** operation creates a counter that cycles through the parent's
states multiple times.

.. code-block:: python

    from statecounter import Counter, Manager, repeat

    with Manager():
        A = Counter(num_states=3, name='A')
        
        B = repeat(A, times=2)  # 6 states
        
        for state in B:
            print(f"B={state}, A={A.state}")

Output::

    B=0, A=0
    B=1, A=1
    B=2, A=2
    B=3, A=0
    B=4, A=1
    B=5, A=2

Shuffle
-------

The **shuffle** operation creates a counter with randomly permuted states.
The shuffle is deterministic when a seed is provided.

.. code-block:: python

    from statecounter import Counter, Manager, shuffle

    with Manager():
        A = Counter(num_states=5, name='A')
        
        # Random permutation with seed for reproducibility
        B = shuffle(A, seed=42)
        
        for state in B:
            print(f"B={state}, A={A.state}")

You can also provide an explicit permutation:

.. code-block:: python

    with Manager():
        A = Counter(num_states=4, name='A')
        
        # Explicit permutation: reverse order
        B = shuffle(A, permutation=[3, 2, 1, 0])

Sample
------

The **sample** operation creates a counter with sampled states from the parent.
This is useful for creating random subsets.

.. code-block:: python

    from statecounter import Counter, Manager, sample

    with Manager():
        A = Counter(num_states=100, name='A')
        
        # Sample 10 states with replacement
        B = sample(A, num_states=10, seed=42)
        
        # Sample 10 states without replacement
        C = sample(A, num_states=10, seed=42, with_replacement=False)

You can also provide explicit sampled states:

.. code-block:: python

    with Manager():
        A = Counter(num_states=10, name='A')
        
        # Explicit states to sample
        B = sample(A, sampled_states=[0, 2, 4, 6, 8])

Split
-----

The **split** operation divides a counter into multiple sub-counters.

Equal Split
~~~~~~~~~~~

Split into N roughly equal parts:

.. code-block:: python

    from statecounter import Counter, Manager, split

    with Manager():
        A = Counter(num_states=10, name='A')
        
        # Split into 3 parts: sizes 4, 3, 3
        B, C, D = split(A, 3)
        
        print(f"B: {B.num_states} states")  # 4
        print(f"C: {C.num_states} states")  # 3
        print(f"D: {D.num_states} states")  # 3

Proportional Split
~~~~~~~~~~~~~~~~~~

Split according to proportions:

.. code-block:: python

    with Manager():
        A = Counter(num_states=100, name='A')
        
        # Split 80/20
        train, test = split(A, [0.8, 0.2], names=['train', 'test'])
        
        print(f"train: {train.num_states} states")  # 80
        print(f"test: {test.num_states} states")    # 20

Interleave
----------

The **interleave** operation creates a counter that alternates between parent
counters' states in a round-robin fashion.

.. code-block:: python

    from statecounter import Counter, Manager, interleave

    with Manager():
        A = Counter(num_states=3, name='A')
        B = Counter(num_states=3, name='B')
        
        C = interleave([A, B])  # 6 states
        
        for state in C:
            print(f"C={state}, A={A.state}, B={B.state}")

Output::

    C=0, A=0, B=None
    C=1, A=None, B=0
    C=2, A=1, B=None
    C=3, A=None, B=1
    C=4, A=2, B=None
    C=5, A=None, B=2

The states alternate: A's state 0, B's state 0, A's state 1, B's state 1, etc.

.. warning::

    All counters passed to ``interleave()`` must have the same ``num_states``.

Passthrough
-----------

The **passthrough** operation creates a counter that mirrors its parent exactly.
This is useful for creating an alias or checkpoint in the counter DAG.

.. code-block:: python

    from statecounter import Counter, Manager, passthrough

    with Manager():
        A = Counter(num_states=5, name='A')
        
        B = passthrough(A, name='B')  # B mirrors A exactly
        
        for state in B:
            print(f"B={state}, A={A.state}")

Output::

    B=0, A=0
    B=1, A=1
    B=2, A=2
    B=3, A=3
    B=4, A=4

Combining Operations
--------------------

Operations can be freely combined to create complex iteration patterns:

.. code-block:: python

    from statecounter import Counter, Manager, stack, shuffle, split

    with Manager():
        # Create base counters for different conditions
        control = Counter(num_states=10, name='control')
        treatment = Counter(num_states=10, name='treatment')
        
        # Stack them (20 states total)
        all_samples = stack([control, treatment])
        
        # Shuffle for randomization
        randomized = shuffle(all_samples, seed=42)
        
        # Split into train/test
        train, test = split(randomized, [0.8, 0.2])
        
        print(f"Train set: {train.num_states} samples")
        print(f"Test set: {test.num_states} samples")
