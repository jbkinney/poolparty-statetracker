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
   :widths: 20 18 12 50
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

Examples
--------

Label sequences from different branches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    wt     = pp.from_seq("ACGTACGT")
    branch = pp.mutagenize(wt, num_mutations=1, prefix="mut")
    tagged = pp.add_prefix(branch, "experiment1")
    df     = tagged.generate_library(num_seqs=3)
    # name column: mut.0001.experiment1, mut.0002.experiment1, ...

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; sequences unchanged, names prefixed)</em>
    A<span class="pp-mut">G</span>CGATCG &nbsp;<em style="color:#6b7280;">mut.0001.experiment1</em><br>
    ACGT<span class="pp-mut">T</span>CGT &nbsp;<em style="color:#6b7280;">mut.0002.experiment1</em><br>
    <span class="pp-mut">G</span>CGTACGT &nbsp;<em style="color:#6b7280;">mut.0003.experiment1</em>
    </div>

See :func:`~poolparty.add_prefix`.
