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
     - Iteration priority for downstream multi-pool iteration.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.add_prefix` in the
   :doc:`API Reference </api>`.

Examples
--------

Label sequences from different branches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    wt     = pp.from_seq("ACGTACGT")
    branch = pp.mutagenize(wt, num_mutations=1, prefix="mut", mode="random")
    tagged = pp.add_prefix(branch, "experiment1")
    tagged.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">tagged: seq_length=8, num_states=1</em>
    ACGTGCGT
    </div>

See :func:`~poolparty.add_prefix`.
