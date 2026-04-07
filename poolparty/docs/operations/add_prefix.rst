add_prefix
==========

Add a label prefix to sequence names without modifying the sequences
themselves. Useful for marking which branch of a pipeline produced each
sequence.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Parameters
----------

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pool``
     - ``Pool``
     - *(required)*
     - Input pool.
   * - ``prefix``
     - ``str``
     - *(required)*
     - Prefix string appended to each sequence name.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Enumeration order when combined with other pools.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.add_prefix` in the
   :doc:`API Reference </api>`.

Examples
--------

Label sequences from a pipeline branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operations like ``mutagenize`` can assign a ``prefix`` to each sequence
name. ``add_prefix`` appends an additional label, letting you distinguish
sequences that came from different branches of a design.

.. code-block:: python

    wt     = pp.from_seq("ATCG")
    muts   = pp.mutagenize(wt, num_mutations=1, mode="sequential",
                           prefix="mut")
    tagged = pp.add_prefix(muts, "batch1")
    df     = tagged.generate_library(num_seqs=5)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 5 rows × 2 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th></tr>
    <tr><td>mut_00.batch1</td><td>CTCG</td></tr>
    <tr><td>mut_01.batch1</td><td>GTCG</td></tr>
    <tr><td>mut_02.batch1</td><td>TTCG</td></tr>
    <tr><td>mut_03.batch1</td><td>AACG</td></tr>
    <tr><td>mut_04.batch1</td><td>ACCG</td></tr>
    </table>
    </div>

See :func:`~poolparty.add_prefix`.
