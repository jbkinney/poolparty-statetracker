Operation Modes
===============

Each operation has a **mode** that controls how it produces its output
sequences — whether they are enumerated exhaustively, sampled randomly, or
uniquely determined by the input.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Number of possibilities
-----------------------

Each operation's type and parameters determine a **design space**: the set of
distinct outputs the operation could yield from a given input. The size of
this space is the **number of possibilities**:

- ``mutagenize(num_mutations=1)`` on ``ACG``: 3 positions × 3 alternative
  bases = **9** possible mutants
- ``deletion_scan(deletion_length=2)`` on an 8-mer: 8 − 2 + 1 = **7** window
  positions
- ``rc()``: always exactly **1** output; the reverse complement is fully
  determined by the input

A 3-mer has only 9 single-point mutants, but single-point mutagenesis of a
100-bp sequence yields 300, and multi-point mutagenesis grows much faster.
When the design space is too large to enumerate, random sampling becomes
necessary. Modes let the user choose between these strategies.

Internal state
--------------

The **internal state** is an index into those possibilities. It tells the
operation which one to use. Each operation exposes
``operation.num_states``, the count of internal states it has. The resulting
``pool.num_states`` is ``operation.num_states`` multiplied by the input
pool's ``num_states`` (see :doc:`library_size` for composition rules).

Each mode sets ``operation.num_states`` differently, and the ``num_states``
parameter can override the default. The following sections cover each mode
and the override.

----

Sequential mode
---------------

``mode="sequential"`` enumerates every possibility of the design space in a deterministic
order. The same input always produces the same library.
``operation.num_states`` equals the number of possibilities.

.. code-block:: python

    wt      = pp.from_seq("ACG")
    mutants = pp.mutagenize(wt, num_mutations=1, mode="sequential")
    mutants.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mutants: seq_length=3, num_states=9</em>
    CCG<br>
    GCG<br>
    TCG<br>
    AAG<br>
    AGG<br>
    ATG<br>
    ACA<br>
    ACC<br>
    ACT
    </div>

All 9 single-point mutants of a 3-mer are produced — 3 positions × 3
non-wild-type bases. ``operation.num_states`` is 9.

Random mode
-----------

``mode="random"`` draws a possibility from the design space each time the library is
generated. Each call to ``generate_library`` draws independently, so
results differ between runs unless a ``seed`` is set. By default,
``operation.num_states`` is 1 and the resulting ``pool.num_states`` is
unchanged.

.. code-block:: python

    wt      = pp.from_seq("ACG")
    mutants = pp.mutagenize(wt, num_mutations=1, mode="random")
    mutants.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mutants: seq_length=3, num_states=1</em>
    ATG
    </div>

A single random mutant is drawn. To generate multiple fixed random designs,
use the ``num_states`` parameter (see
`Overriding with the num_states parameter`_ below).

Fixed mode
----------

``mode="fixed"`` means the output is uniquely determined by the input; there
is no variation. These operations do not accept a ``mode`` parameter because
there is only one possible behaviour. ``operation.num_states`` is 1 and the
resulting ``pool.num_states`` is unchanged.

.. code-block:: python

    wt = pp.from_seq("ATCG")
    r  = pp.rc(wt)
    r.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">r: seq_length=4, num_states=1</em>
    CGAT
    </div>

The reverse complement of ``ATCG`` is always ``CGAT``.

----

Overriding with the ``num_states`` parameter
---------------------------------------------

The ``num_states`` parameter lets you control how many internal states the
operation uses, overriding the default.

**Sequential mode:**
    With ``num_states=N``:

    - **N < number of possibilities** — the enumeration is **truncated** to the
      first *N* outputs.
    - **N = number of possibilities** — identical to omitting ``num_states``.
    - **N > number of possibilities** — the enumeration **cycles**, wrapping back
      to the beginning after exhausting all possibilities.

    In all cases, ``operation.num_states`` is set to *N*.

    The examples below use single-point mutagenesis of ``ACG``, which has 9
    possibilities (3 positions × 3 alternative bases).

    *Truncation* (``num_states`` < number of possibilities):

    .. code-block:: python

        wt      = pp.from_seq("ACG")
        mutants = pp.mutagenize(wt, num_mutations=1, mode="sequential", num_states=3)
        mutants.print_library()

    .. raw:: html

        <div class="pp-pool">
        <em class="pp-header">mutants: seq_length=3, num_states=3</em>
        CCG<br>
        GCG<br>
        TCG
        </div>

    Only the first 3 of 9 possibilities are produced.

    *Exact match* (``num_states`` omitted): all 9 possibilities are produced, as
    shown in the `Sequential mode`_ example above.

    *Cycling* (``num_states`` > number of possibilities):

    .. code-block:: python

        wt      = pp.from_seq("ACG")
        mutants = pp.mutagenize(wt, num_mutations=1, mode="sequential", num_states=12)
        mutants.print_library()

    .. raw:: html

        <div class="pp-pool">
        <em class="pp-header">mutants: seq_length=3, num_states=12</em>
        <span class="pp-ellipsis"># cycle 1</span><br>
        CCG<br>
        GCG<br>
        TCG<br>
        AAG<br>
        AGG<br>
        ATG<br>
        ACA<br>
        ACC<br>
        ACT<br>
        <span class="pp-ellipsis"># cycle 2 (wraps around)</span><br>
        CCG<br>
        GCG<br>
        TCG
        </div>

    After exhausting all 9 possibilities, the enumeration wraps around and
    repeats from the beginning.

**Random mode:**
    With ``num_states=N``, the operation draws *N* random possibilities once.
    These same *N* designs are applied to every input sequence.
    ``operation.num_states`` is *N* and the resulting ``pool.num_states`` is
    the input pool's ``num_states`` × *N*.

    .. code-block:: python

        wt      = pp.from_seq("ACG")
        mutants = pp.mutagenize(wt, num_mutations=1, mode="random", num_states=5)
        mutants.print_library()

    .. raw:: html

        <div class="pp-pool">
        <em class="pp-header">mutants: seq_length=3, num_states=5</em>
        ATG<br>
        GCG<br>
        ACT<br>
        ACT<br>
        AAG
        </div>

    Five randomly chosen single-point mutants are generated. Duplicates are
    possible because each draw is independent.

**Fixed mode:**
    Does not accept the ``num_states`` parameter — the output is always
    uniquely determined by the input.

----

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Mode
     - ``num_states`` argument
     - ``operation.num_states``
   * - sequential
     - ``None`` (default)
     - all possibilities (e.g. 9)
   * - sequential
     - N
     - N (truncates or cycles)
   * - random
     - ``None`` (default)
     - 1 (fresh draw each time)
   * - random
     - N
     - N (N fixed random designs)
   * - fixed
     - not accepted
     - 1 (determined by input)

----

Inspecting pool and operation attributes
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Attribute
     - Description
   * - ``pool.num_states``
     - Total number of sequences the pool produces
   * - ``pool.seq_length``
     - Fixed sequence length (or ``None``)
   * - ``pool.operation.mode``
     - The mode of the operation that created this pool
   * - ``pool.operation.num_states``
     - The operation's internal state count
   * - ``pool.operation.natural_num_states``
     - The number of possibilities before any ``num_states`` override

Use ``print_dag()`` to inspect the full operation tree, which shows each
node's mode and internal state count:

.. code-block:: python

    mutants.print_dag()

.. code-block:: text

    pool[1] (pool, n=9)
    └── op[1]:mutagenize [mode=sequential, n=9]
        └── pool[0] (pool, n=1)
            └── op[0]:from_seq [mode=fixed, n=1]

See :doc:`library_size` for how internal states compose when operations are
chained.

----

Default modes
-------------

Each operation has its own default mode. Most operations default to
``random``; operations like ``rc`` are always ``fixed``. Check each
operation's parameter table for its default.
