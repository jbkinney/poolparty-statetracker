Why StateTracker?
=================

StateTracker was developed to support the design of complex DNA sequence
libraries (see `PoolParty <https://github.com/jkinney/poolparty>`_), but it
solves a general problem that arises whenever you need to enumerate a
combinatorial space.

This page explains the core problem StateTracker addresses and why existing
approaches fall short.


The Problem: Random Access to Combinatorial Spaces
---------------------------------------------------

Consider a common scenario: you're designing an experiment with multiple
conditions. Say you have 3 treatments and 4 replicates, giving you 12
experimental samples:

.. list-table::
   :header-rows: 1

   * - Sample
     - Treatment
     - Replicate
   * - 0
     - 0
     - 0
   * - 1
     - 1
     - 0
   * - 2
     - 2
     - 0
   * - 3
     - 0
     - 1
   * - ...
     - ...
     - ...
   * - 11
     - 2
     - 3

With nested loops, enumerating this space is trivial:

.. code-block:: python

    # Nested loops: easy to enumerate
    for replicate in range(4):
        for treatment in range(3):
            sample = replicate * 3 + treatment
            print(f"Sample {sample}: treatment={treatment}, replicate={replicate}")

.. code-block:: text

    Sample 0: treatment=0, replicate=0
    Sample 1: treatment=1, replicate=0
    Sample 2: treatment=2, replicate=0
    Sample 3: treatment=0, replicate=1
    Sample 4: treatment=1, replicate=1
    Sample 5: treatment=2, replicate=1
    Sample 6: treatment=0, replicate=2
    Sample 7: treatment=1, replicate=2
    Sample 8: treatment=2, replicate=2
    Sample 9: treatment=0, replicate=3
    Sample 10: treatment=1, replicate=3
    Sample 11: treatment=2, replicate=3

But what if you need to:

- **Random access**: Given sample #7, what are its treatment and replicate?
- **Shuffle**: Randomize the order of samples while still tracking which
  treatment/replicate each corresponds to?
- **Sample**: Select a random subset of 5 samples?
- **Split**: Divide into training (80%) and test (20%) sets?

Nested loops can't help here. You need a way to go from a **single index** to
the **component indices**.


The Naive Solution: Manual Index Math
-------------------------------------

You can compute component indices using ``divmod``:

.. code-block:: python

    # Manual index math: compute treatment and replicate from sample number
    def get_indices(sample, num_treatments=3):
        replicate, treatment = divmod(sample, num_treatments)
        return treatment, replicate

    # Random access to sample #7
    treatment, replicate = get_indices(7)
    print(f"Sample 7: treatment={treatment}, replicate={replicate}")

.. code-block:: text

    Sample 7: treatment=1, replicate=2

This works for simple products, but the approach has serious limitations:

**1. It doesn't compose.** What if you have a more complex structure?

.. code-block:: python

    # Complex scenario: 2 control samples + (3 treatments x 4 replicates)
    # This is a "stack" of a simple state and a product
    # Total: 2 + 12 = 14 samples

    def get_complex_indices(sample):
        """Manual index math for: stack(control[2], product(treatment[3], replicate[4]))"""
        if sample < 2:
            return {"type": "control", "control": sample, "treatment": None, "replicate": None}
        else:
            adjusted = sample - 2
            replicate, treatment = divmod(adjusted, 3)
            return {
                "type": "treatment",
                "control": None,
                "treatment": treatment,
                "replicate": replicate,
            }

    # This is already getting complicated...
    for i in [0, 1, 2, 7, 13]:
        print(f"Sample {i}: {get_complex_indices(i)}")

.. code-block:: text

    Sample 0: {'type': 'control', 'control': 0, 'treatment': None, 'replicate': None}
    Sample 1: {'type': 'control', 'control': 1, 'treatment': None, 'replicate': None}
    Sample 2: {'type': 'treatment', 'control': None, 'treatment': 0, 'replicate': 0}
    Sample 7: {'type': 'treatment', 'control': None, 'treatment': 2, 'replicate': 1}
    Sample 13: {'type': 'treatment', 'control': None, 'treatment': 2, 'replicate': 3}

**2. Every operation requires new math.** Want to shuffle? You need to track a
permutation and apply it before computing indices. Want to sample? You need to
track which original indices were sampled. Want to split? More bookkeeping.

**3. It's error-prone.** Off-by-one errors, wrong divisors, forgetting to
handle edge cases --- manual index math is a minefield.


StateTracker's Solution: Composable States
-------------------------------------------

StateTracker solves this with a simple but powerful idea: **build a state DAG
that mirrors your combinatorial structure, then let values propagate
automatically**.

Here's the same complex scenario with StateTracker:

.. code-block:: python

    from statetracker import Manager, State, product, stack

    with Manager():
        # Define the structure declaratively
        control = State(num_values=2, name="control")
        treatment = State(num_values=3, name="treatment")
        replicate = State(num_values=4, name="replicate")

        # Compose: stack control with (treatment x replicate)
        treatment_arm = product([treatment, replicate])
        samples = stack([control, treatment_arm])

        # Now iterate -- parent states update automatically!
        for value in samples:
            print(
                f"Sample {value}: control={control.value}, "
                f"treatment={treatment.value}, replicate={replicate.value}"
            )

.. code-block:: text

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

The key insight: **set one value, and all parent states propagate
automatically**. This gives you single-index random access to any point in the
combinatorial space:

.. code-block:: python

    with Manager():
        control = State(num_values=2, name="control")
        treatment = State(num_values=3, name="treatment")
        replicate = State(num_values=4, name="replicate")

        treatment_arm = product([treatment, replicate])
        samples = stack([control, treatment_arm])

        # Random access: what are the indices for sample #7?
        samples.value = 7
        print(
            f"Sample 7: control={control.value}, "
            f"treatment={treatment.value}, replicate={replicate.value}"
        )

.. code-block:: text

    Sample 7: control=None, treatment=2, replicate=1

And because StateTracker handles the index math internally, operations like
shuffle, sample, and split become trivial:

.. code-block:: python

    from statetracker import sample, shuffle, split

    with Manager():
        control = State(num_values=2, name="control")
        treatment = State(num_values=3, name="treatment")
        replicate = State(num_values=4, name="replicate")

        treatment_arm = product([treatment, replicate])
        samples = stack([control, treatment_arm], name="samples")

        # Shuffle: randomize sample order
        shuffled = shuffle(samples, seed=42)

        # Split: 80% train, 20% test
        train, test = split(shuffled, [0.8, 0.2])

        print(f"Total samples: {samples.num_values}")
        print(f"Train samples: {train.num_values}")
        print(f"Test samples: {test.num_values}")
        print()

        # Iterate through test set -- parent states still propagate correctly!
        print("Test set:")
        for value in test:
            print("  ", end="")
            test.print_states(include_inactive=False)

.. code-block:: text

    Total samples: 14
    Train samples: 11
    Test samples: 3

    Test set:
      samples=0, control=0
      samples=1, control=1
      samples=10, treatment=2, replicate=2


When to Use StateTracker
------------------------

StateTracker is useful whenever you need **single-index access to a
combinatorial space**. Common scenarios include:

**Experimental Design**
    Randomizing treatment/control order while tracking which condition each
    sample belongs to. Splitting experiments into batches while maintaining
    structured indices.

**Combinatorial Libraries**
    Generating DNA sequence variants with structured indices (the original
    motivation --- see `PoolParty <https://github.com/jkinney/poolparty>`_).
    Enumerating parameter combinations for hyperparameter search.

**Machine Learning**
    Creating train/validation/test splits on structured datasets.
    Stratified sampling from combinatorial data.

**General Enumeration**
    Any domain where you build complex iteration patterns from simpler ones.
    When you need to shuffle, sample, or slice a combinatorial space without
    reimplementing index math.


Summary
-------

**If you've ever written nested loops and wished you could shuffle the
iteration order, or needed random access to a point in a Cartesian product,
StateTracker is for you.**

The library lets you:

1. Define your combinatorial structure declaratively
2. Compose states using algebraic operations (product, stack, slice, etc.)
3. Set one value and have all parent states propagate automatically
4. Freely shuffle, sample, split, and slice without reimplementing index math

Continue to the :doc:`quickstart` to learn the basics, or dive into
:doc:`concepts` for a deeper understanding.
