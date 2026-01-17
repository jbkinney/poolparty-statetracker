Core Concepts
=============

This guide explains the fundamental concepts behind StateCounter's design.

.. contents:: On this page
   :local:
   :depth: 2

Counters and States
-------------------

A **Counter** is an object that can take on a finite number of discrete states,
numbered from 0 to ``num_states - 1``. The simplest counter is a "leaf" counter
created directly with a specified number of states:

.. code-block:: python

    from statecounter import Counter, Manager

    with Manager():
        A = Counter(num_states=5, name='A')
        print(list(A))  # [0, 1, 2, 3, 4]

Counters can be **iterated** to cycle through all their states, and their current
state can be read or set via the ``state`` property.

The Manager Context
-------------------

All counters must be created within a :class:`~statecounter.Manager` context.
The Manager tracks all counters and their relationships:

.. code-block:: python

    with Manager() as mgr:
        A = Counter(num_states=3, name='A')
        B = Counter(num_states=2, name='B')
        
        # Manager tracks all counters
        print(mgr.get_all_names())  # ['A', 'B']

Creating counters outside a Manager context raises a ``RuntimeError``.

Counter Composition
-------------------

The power of StateCounter comes from **composing** counters using operations.
When you combine counters, you create a new "derived" counter that depends on
its "parent" counters:

.. code-block:: python

    with Manager():
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        
        # C is derived from A and B via product
        C = A * B  # 6 states (2 × 3)
        C.name = 'C'

This creates a **directed acyclic graph (DAG)** of counter dependencies:

.. code-block:: text

    C (counter, io=0, n=6)
    └── [op=Product]
        ├── A (counter, io=0, n=2)
        └── B (counter, io=0, n=3)

You can visualize this structure using :meth:`~statecounter.Counter.print_dag`:

.. code-block:: python

    C.print_dag()

Unidirectional State Propagation
--------------------------------

StateCounter uses **unidirectional state propagation**: when you set the state
of a derived counter, the states of all its parent counters are automatically
computed and updated.

.. code-block:: python

    with Manager():
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        C = A * B
        
        C.state = 5  # Set the derived counter's state
        
        # Parent states are automatically computed
        print(f"A.state = {A.state}")  # 1
        print(f"B.state = {B.state}")  # 2

This is the key insight: you iterate over the **derived** counter, and the
**parent** counters automatically track along.

Active vs Inactive States
-------------------------

A counter's state can be either:

- **Active**: An integer from 0 to ``num_states - 1``
- **Inactive**: ``None``

Inactive states arise with operations like :func:`~statecounter.stack` where
only one parent is "active" at a time:

.. code-block:: python

    with Manager():
        A = Counter(num_states=2, name='A')
        B = Counter(num_states=3, name='B')
        C = stack([A, B])  # 5 states total
        
        for state in C:
            print(f"C={state}, A={A.state}, B={B.state}")

Output::

    C=0, A=0, B=None
    C=1, A=1, B=None
    C=2, A=None, B=0
    C=3, A=None, B=1
    C=4, A=None, B=2

Use :meth:`~statecounter.Counter.is_active` to check if a counter is active.

Conflict Detection
------------------

When a counter appears in multiple branches of the DAG, StateCounter detects
**conflicting state assignments**. This happens when two different paths would
assign different values to the same parent counter.

.. code-block:: python

    with Manager():
        A = Counter(num_states=3, name='A')
        B = A[0:2]  # States 0, 1 of A
        C = A[1:3]  # States 1, 2 of A
        D = stack([B, C])
        
        # State 0 of D → B state 0 → A state 0 ✓
        # State 2 of D → C state 0 → A state 1
        # But what if we try to set both B and C active?
        
        # This would conflict: A can't be both 0 and 1

A :class:`~statecounter.ConflictingStateAssignmentError` is raised when such
conflicts are detected during state propagation.

Iteration Order
---------------

Each counter has an ``iter_order`` property that influences how counters are
ordered in operations like :func:`~statecounter.ordered_product`. Lower values
iterate "faster" (change more frequently in the inner loop).

.. code-block:: python

    with Manager():
        A = Counter(num_states=2, name='A', iter_order=0)
        B = Counter(num_states=2, name='B', iter_order=1)
        
        C = ordered_product([A, B])
        
        for _ in C:
            print(f"A={A.state}, B={B.state}")

Output::

    A=0, B=0
    A=1, B=0
    A=0, B=1
    A=1, B=1

The default ``iter_order`` is 0 for leaf counters. Derived counters inherit
the minimum ``iter_order`` of their parents.

You can control the global ordering behavior with
:func:`~statecounter.set_product_order_mode`:

- ``'first_counter_slowest'`` (default): First counter in sorted order changes slowest
- ``'first_counter_fastest'``: First counter in sorted order changes fastest

Counter Identity
----------------

Each counter has a unique ``id`` assigned by the Manager. This is used for:

- Deduplication in :func:`~statecounter.ordered_product`
- Tie-breaking when counters have the same ``iter_order``

Counters are compared by identity (``is``), not value, so two counters with
the same ``num_states`` are still distinct objects.

Copying Counters
----------------

Counters support two types of copying:

- :meth:`~statecounter.Counter.copy`: Creates a new counter with the same parents (shallow copy)
- :meth:`~statecounter.Counter.deepcopy`: Creates a new counter with copied parents (deep copy)

.. code-block:: python

    with Manager():
        A = Counter(num_states=3, name='A')
        B = A[1:3]
        
        # Shallow copy: C shares parent A with B
        C = B.copy(name='C')
        
        # Deep copy: D has its own copy of the parent
        D = B.deepcopy(name='D')
