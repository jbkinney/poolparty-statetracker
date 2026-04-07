Alternative Approaches
======================

This page compares StateTracker to other methods for working with
combinatorial structures.

**The key insight**: StateTracker's utility comes from building a *computation
DAG* (directed acyclic graph) of states, where setting the value of a derived
state automatically propagates to all parent states.
this.


The Problem: Complex Combinatorial Structures
----------------------------------------------

Real-world combinatorial problems often involve more than simple Cartesian
products. Consider designing an experiment with:

- 2 control samples
- Plus a treatment arm with 3 treatments x 4 replicates

This is a **stack** (disjoint union) of a simple state and a **product**. The
total is 2 + 12 = 14 samples, but they have different structure depending on
which "arm" is active.

When you iterate through these 14 samples, you need to know:

1. Which arm is active (control or treatment)?
2. If treatment, what are the treatment and replicate indices?
3. If you shuffle, slice, or sample, how do you track all this?

Let's see how hard this is without StateTracker.


The Manual Approach
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Manual approach: 2 controls + (3 treatments x 4 replicates)
    def get_indices_manual(sample_idx):
        """Manually compute which arm is active and the component indices."""
        num_controls = 2
        num_treatments = 3

        if sample_idx < num_controls:
            return {"control": sample_idx, "treatment": None, "replicate": None}
        else:
            adjusted = sample_idx - num_controls
            treatment = adjusted % num_treatments
            replicate = adjusted // num_treatments
            return {"control": None, "treatment": treatment, "replicate": replicate}

    print("Manual enumeration of 14 samples:")
    for i in range(14):
        print(f"  Sample {i}: {get_indices_manual(i)}")

.. code-block:: text

    Manual enumeration of 14 samples:
      Sample 0: {'control': 0, 'treatment': None, 'replicate': None}
      Sample 1: {'control': 1, 'treatment': None, 'replicate': None}
      Sample 2: {'control': None, 'treatment': 0, 'replicate': 0}
      Sample 3: {'control': None, 'treatment': 1, 'replicate': 0}
      Sample 4: {'control': None, 'treatment': 2, 'replicate': 0}
      Sample 5: {'control': None, 'treatment': 0, 'replicate': 1}
      Sample 6: {'control': None, 'treatment': 1, 'replicate': 1}
      Sample 7: {'control': None, 'treatment': 2, 'replicate': 1}
      Sample 8: {'control': None, 'treatment': 0, 'replicate': 2}
      Sample 9: {'control': None, 'treatment': 1, 'replicate': 2}
      Sample 10: {'control': None, 'treatment': 2, 'replicate': 2}
      Sample 11: {'control': None, 'treatment': 0, 'replicate': 3}
      Sample 12: {'control': None, 'treatment': 1, 'replicate': 3}
      Sample 13: {'control': None, 'treatment': 2, 'replicate': 3}

This works, but now imagine you want to **shuffle** these samples. You need to:

1. Generate a random permutation
2. Apply it before looking up indices
3. Rewrite all your index math to account for the permutation

And what if you want to **slice** (take samples 5--10)? Or **split** into
train/test? Each operation requires rewriting the index logic. This quickly
becomes unmanageable.


StateTracker's Solution: The State DAG
--------------------------------------

StateTracker solves this by building a **directed acyclic graph (DAG)** of
states. You declare the structure once, and StateTracker handles all the index
math automatically, including for shuffles, slices, samples, and splits.

.. code-block:: python

    from statetracker import Manager, State, product, stack

    with Manager():
        control = State(num_values=2, name="control")
        treatment = State(num_values=3, name="treatment")
        replicate = State(num_values=4, name="replicate")

        treatment_arm = product([treatment, replicate])
        samples = stack([control, treatment_arm])

        print("StateTracker enumeration of 14 samples:")
        for value in samples:
            print(
                f"  Sample {value}: control={control.value}, "
                f"treatment={treatment.value}, replicate={replicate.value}"
            )

.. code-block:: text

    StateTracker enumeration of 14 samples:
      Sample 0: control=0, treatment=None, replicate=None
      Sample 1: control=1, treatment=None, replicate=None
      Sample 2: control=None, treatment=0, replicate=0
      Sample 3: control=None, treatment=1, replicate=0
      Sample 4: control=None, treatment=2, replicate=0
      Sample 5: control=None, treatment=0, replicate=1
      Sample 6: control=None, treatment=1, replicate=1
      Sample 7: control=None, treatment=2, replicate=1
      Sample 8: control=None, treatment=0, replicate=2
      Sample 9: control=None, treatment=1, replicate=2
      Sample 10: control=None, treatment=2, replicate=2
      Sample 11: control=None, treatment=0, replicate=3
      Sample 12: control=None, treatment=1, replicate=3
      Sample 13: control=None, treatment=2, replicate=3

The magic is **automatic value propagation**: when you set the value of the
derived state, all parent states (control, treatment, replicate) update
automatically. Inactive states show ``None``.

Now adding shuffle, slice, or split is trivial:

.. code-block:: python

    from statetracker import shuffle, split

    with Manager():
        control = State(num_values=2, name="control")
        treatment = State(num_values=3, name="treatment")
        replicate = State(num_values=4, name="replicate")

        treatment_arm = product([treatment, replicate])
        samples = stack([control, treatment_arm])

        shuffled = shuffle(samples, seed=42)
        train, test = split(shuffled, [0.8, 0.2])

        print(f"Total: {samples.num_values}, Train: {train.num_values}, Test: {test.num_values}")
        print("\nTest set (values still propagate correctly):")
        for value in test:
            print(
                f"  control={control.value}, treatment={treatment.value}, "
                f"replicate={replicate.value}"
            )

.. code-block:: text

    Total: 14, Train: 11, Test: 3

    Test set (values still propagate correctly):
      control=0, treatment=None, replicate=None
      control=1, treatment=None, replicate=None
      control=None, treatment=2, replicate=2

You can visualize the state DAG to understand the structure:

.. code-block:: python

    with Manager():
        control = State(num_values=2, name="control")
        treatment = State(num_values=3, name="treatment")
        replicate = State(num_values=4, name="replicate")

        treatment_arm = product([treatment, replicate], name="treatment_arm")
        samples = stack([control, treatment_arm], name="samples")
        shuffled = shuffle(samples, seed=42, name="shuffled")
        train, test = split(shuffled, [0.8, 0.2], names=["train", "test"])

        test.print_dag()

.. code-block:: text

    test (counter, io=0, n=3)
    +-- [op=Slice]
        +-- shuffled (counter, io=0, n=14)
            +-- [op=Shuffle]
                +-- samples (counter, io=0, n=14)
                    +-- [op=Stack]
                        +-- control (counter, io=0, n=2)
                        +-- treatment_arm (counter, io=0, n=12)
                            +-- [op=Product]
                                +-- treatment (counter, io=0, n=3)
                                +-- replicate (counter, io=0, n=4)


Alternative Approaches in Python
---------------------------------

Let's examine what other Python tools offer and why they fall short for
problems involving both products and stacks with additional operations.


itertools
~~~~~~~~~

Python's ``itertools`` module provides ``product`` for Cartesian products and
``chain`` for concatenation:

.. code-block:: python

    from itertools import chain, product

    # itertools.product for Cartesian products
    print("itertools.product (2 x 3):")
    for i, (t, r) in enumerate(product(range(2), range(3))):
        print(f"  {i}: ({t}, {r})")

    # itertools.chain for concatenation
    print("\nitertools.chain (2 controls + 3 treatments):")
    controls = [("control", i) for i in range(2)]
    treatments = [("treatment", i) for i in range(3)]
    for i, item in enumerate(chain(controls, treatments)):
        print(f"  {i}: {item}")

.. code-block:: text

    itertools.product (2 x 3):
      0: (0, 0)
      1: (0, 1)
      2: (0, 2)
      3: (1, 0)
      4: (1, 1)
      5: (1, 2)

    itertools.chain (2 controls + 3 treatments):
      0: ('control', 0)
      1: ('control', 1)
      2: ('treatment', 0)
      3: ('treatment', 1)
      4: ('treatment', 2)

**Limitations of itertools:**

1. **No value propagation**: You get tuples, not connected objects. There is no
   way to ask "given index 4, what are the component values?"
2. **No composition**: You cannot easily combine ``product`` and ``chain`` into
   a single enumerable structure with unified indexing.
3. **No tracking for chain**: With ``chain``, you lose information about which
   source contributed each element. You have to encode it manually.
4. **Operations do not compose**: To shuffle an ``itertools.product``, you must
   materialize it into a list first.


NumPy's unravel_index / ravel_multi_index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NumPy provides functions to convert between flat indices and multi-dimensional
indices for arrays:

.. code-block:: python

    import numpy as np

    shape = (2, 3)

    # Convert flat index 4 to multi-dimensional indices
    indices = np.unravel_index(4, shape)
    print(f"Flat index 4 -> treatment={indices[0]}, replicate={indices[1]}")

    # Convert back to flat index
    flat = np.ravel_multi_index(indices, shape)
    print(f"treatment={indices[0]}, replicate={indices[1]} -> flat index {flat}")

.. code-block:: text

    Flat index 4 -> treatment=1, replicate=1
    treatment=1, replicate=1 -> flat index 4

**Limitations of NumPy's index functions:**

1. **Only rectangular products**: NumPy assumes a regular multi-dimensional
   array shape. It cannot represent structures like "2 controls + (3 x 4
   treatments)" which are not rectangular.
2. **No stacks (disjoint unions)**: There is no NumPy equivalent for
   StateTracker's ``stack`` operation.
3. **No value propagation**: You get index tuples, not connected state objects.
4. **No composable operations**: Slicing, shuffling, or sampling requires
   manual reimplementation of all the index math.


Manual Index Math (divmod)
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can always compute indices manually using ``divmod``, as shown on the
:doc:`motivation` page:

.. code-block:: python

    def get_product_indices(flat_idx, sizes):
        """Convert flat index to component indices for a Cartesian product."""
        indices = []
        for size in sizes:
            flat_idx, idx = divmod(flat_idx, size)
            indices.append(idx)
        return tuple(indices)

    sizes = [2, 3]
    print("Manual product enumeration:")
    for flat in range(6):
        t, r = get_product_indices(flat, sizes)
        print(f"  {flat}: treatment={t}, replicate={r}")

.. code-block:: text

    Manual product enumeration:
      0: treatment=0, replicate=0
      1: treatment=1, replicate=0
      2: treatment=0, replicate=1
      3: treatment=1, replicate=1
      4: treatment=0, replicate=2
      5: treatment=1, replicate=2

**Limitations of manual index math:**

1. **Does not compose**: Every new operation (stack, slice, shuffle, sample)
   requires writing new index logic from scratch.
2. **Error-prone**: Off-by-one errors, wrong divisors, forgetting edge cases.
3. **No state tracking**: You get tuples, not connected objects.
4. **Tedious for complex structures**: Combining products and stacks manually
   is already complicated. Adding shuffle or slice makes it worse.


What Makes StateTracker Different
---------------------------------

The key difference is that StateTracker builds a **computation DAG** where:

1. **Leaf states** represent basic dimensions (control, treatment, replicate, etc.)
2. **Operations** (product, stack, slice, shuffle, etc.) create derived states
3. **Value propagation** flows automatically from any derived state to all its
   ancestors

This means you declare your structure once, and StateTracker handles all the
index math for every operation you compose on top.


Comparison Summary
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Feature
     - itertools
     - NumPy
     - Manual divmod
     - StateTracker
   * - Cartesian products
     - Yes
     - Yes
     - Yes
     - Yes
   * - Disjoint unions (stack)
     - chain (no tracking)
     - No
     - Manual
     - **Yes, with tracking**
   * - Value propagation
     - No
     - No
     - No
     - **Automatic**
   * - Composable operations
     - No
     - No
     - No
     - **Yes**
   * - Shuffle/slice/sample
     - Must materialize
     - Manual
     - Manual
     - **Built-in**
   * - Conflict detection
     - N/A
     - N/A
     - Manual
     - **Automatic**
   * - Synchronization
     - N/A
     - N/A
     - Manual
     - **Built-in**


When to Use Each Approach
--------------------------

**Use itertools when:**
    You just need to iterate through all combinations once, you do not need to
    track component indices, and your structure is simple.

**Use NumPy when:**
    You are working with actual array data, your structure is a regular
    rectangular grid, and you need fast vectorized operations.

**Use manual divmod when:**
    You have a one-off simple product and are comfortable with the math.

**Use StateTracker when:**
    You have complex structures combining products AND stacks, you need
    automatic tracking of which components are active, you want to compose
    operations without reimplementing index math, or you need synchronization
    and conflict detection.


Summary
-------

StateTracker's unique value is **automatic child-to-parent value propagation
through a computation DAG**. This is fundamentally different from:

- **itertools**: Provides iteration but no state tracking or composition
- **NumPy**: Provides index math for rectangular arrays but cannot handle
  stacks or complex compositions
- **Manual divmod**: Works for simple cases but does not compose and is
  error-prone

When you need to enumerate combinatorial structures that combine products,
stacks, slices, shuffles, samples, or splits while always knowing which
component values correspond to each position, StateTracker is the right tool.

For more details on StateTracker's capabilities, see:

- :doc:`quickstart` --- Get started with basic usage
- :doc:`concepts` --- Understand the state DAG and value propagation
- :doc:`operations/index` --- Complete reference for all operations
