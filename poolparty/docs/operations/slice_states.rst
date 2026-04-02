slice\_states
=============

Retain a contiguous slice of a pool's state space using standard Python slice
syntax, without altering the sequences themselves.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Parameters
----------

.. list-table::
   :widths: 20 18 12 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pool``
     - ``Pool``
     - *(required)*
     - Input pool whose state space will be sliced.
   * - ``key``
     - ``int | slice``
     - *(required)*
     - Integer index (single state) or ``slice`` object specifying which
       states to retain. Supports negative indices and standard Python
       slice syntax.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Iteration priority for downstream multi-pool iteration.

----

Examples
--------

First 8 States of a 2-mer Pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Take the first eight states of the 16-state 2-mer enumeration to obtain the
eight 2-mers that begin with ``A`` or ``C``.

.. code-block:: python

    kmers  = pp.get_kmers(length=2, alphabet="ACGT")  # 16 states: AA…TT
    first8 = pp.state_slice(kmers, slice(0, 8))

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (8 sequences &mdash; states 0&ndash;7 of 16 2-mers)</em>
    AA<br>AC<br>AG<br>AT<br>CA<br>CC<br>CG<br>CT
    </div>

Middle Slice: States 10–15
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select a window in the middle of the state space to focus on a particular
sub-range of sequences without regenerating the entire pool.

.. code-block:: python

    kmers   = pp.get_kmers(length=2, alphabet="ACGT")  # 16 states
    mid     = pp.state_slice(kmers, slice(10, 15))

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (5 sequences &mdash; states 10&ndash;14 of 16 2-mers)</em>
    GG<br>GT<br>TA<br>TC<br>TG
    </div>

Last 4 States Using Negative Indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Negative indices count from the end of the state space, mirroring Python list
slicing conventions.

.. code-block:: python

    kmers = pp.get_kmers(length=2, alphabet="ACGT")  # 16 states
    last4 = pp.state_slice(kmers, slice(-4, None))

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (4 sequences &mdash; last 4 of 16 2-mers)</em>
    TA<br>TC<br>TG<br>TT
    </div>

See :func:`~poolparty.state_slice`.
