Mutagenesis Operations
======================

Mutagenesis operations introduce random variation into sequences through
point mutations, base shuffling, recombination, or orientation flipping.
These operations are stochastic by default and produce a different draw
each time, though most support ``mode='sequential'`` for exhaustive
enumeration of all possible variants.

.. code-block:: python

    import poolparty as pp
    pp.init()

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`mutagenize`
     - Introduce random point mutations at a specified rate or count.
   * - :doc:`shuffle_seq`
     - Randomly permute the bases of a sequence or a tagged region.
   * - :doc:`recombine`
     - Produce chimeric sequences by recombining with source sequences at random breakpoints.
   * - :doc:`flip`
     - Produce forward and reverse-complement variants as distinct pool states.

.. toctree::
   :hidden:

   mutagenize
   shuffle_seq
   recombine
   flip
