join
====

Concatenate a list of pools (or plain strings) end-to-end into a single
composite pool. Every combination of one sequence from each input pool is
produced (Cartesian product). Pass a spacer string to insert a fixed linker
between adjacent segments.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 18 12 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pools``
     - ``list[Pool | str]``
     - *(required)*
     - List of :class:`~poolparty.Pool` objects or plain strings. Strings are
       automatically promoted to single-sequence pools.
   * - ``spacer_str``
     - ``str``
     - ``''``
     - Fixed string inserted between every adjacent pair of segments. Not
       itself a pool dimension.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering: the pool with the *smallest* value varies
       fastest (inner loop). Can be set on each individual pool instead.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to the joined result.

----

Examples
--------

Join two pools (all pairwise combinations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Joining two multi-sequence pools produces every combination of one sequence
from each input.

.. code-block:: python

    left  = pp.from_seqs(["AA", "CC"], mode="sequential")
    right = pp.from_seqs(["GG", "TT"], mode="sequential")
    pool  = pp.join([left, right])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (4 sequences &mdash; all combinations of left &times; right)</em>
    AAGG<br>AATT<br>CCGG<br>CCTT
    </div>

Join with a fixed spacer
~~~~~~~~~~~~~~~~~~~~~~~~~

``spacer_str`` inserts a constant linker between every adjacent pair.

.. code-block:: python

    left  = pp.from_seq("ACGT")
    right = pp.from_seqs(["AAAA", "CCCC", "GGGG"], mode="sequential")
    pool  = pp.join([left, right], spacer_str="TTT")

.. raw:: html

    <div class="pp-pool"> 
    <em class="pp-header">Pool (3 sequences &mdash; left + "TTT" spacer + right)</em>
    ACGTTTTAAAA<br>ACGTTTTCCCC<br>ACGTTTTGGGG
    </div>

Controlling iteration order with ``iter_order``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pool with the smallest ``iter_order`` value varies fastest (inner loop).
Here ``b`` is the inner loop and ``a`` is the outer loop.

.. code-block:: python

    a    = pp.from_seqs(["A", "C"], mode='sequential', iter_order=2)   # outer (slower)
    b    = pp.from_seqs(["G", "T"], mode='sequential', iter_order=1)   # inner (faster)
    pool = pp.join([a, b])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (4 sequences &mdash; b varies fastest: G,T,G,T; a varies slowest: A,A,C,C)</em>
    AG<br>AT<br>CG<br>CT
    </div>

Join three pools with a literal string segment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plain strings in the list are treated as fixed single-sequence pools,
eliminating the need to call :func:`~poolparty.from_seq` for constant flanks.

.. code-block:: python

    promoter = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
    insert   = pp.from_seqs(["GCGC", "TATA"], mode="sequential")
    pool     = pp.join([promoter, "NNNNN", insert])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (4 sequences &mdash; promoter + fixed "NNNNN" linker + insert)</em>
    AAAANNNNNGCGC<br>AAAANNNNNTATA<br>CCCCNNNNNGCGC<br>CCCCNNNNNTATA
    </div>

See :func:`~poolparty.join`.
