Sequence Utilities
==================

Sequence utilities apply deterministic, one-to-one transformations to
sequences. Each input sequence produces exactly one output sequence with
no internal state. These operations cover reverse complementation, case
conversion, display styling, cleanup, slicing, and name prefixing.

.. code-block:: python

    import poolparty as pp
    pp.init()

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`rc`
     - Take the reverse complement of a sequence or named region.
   * - :doc:`case_ops`
     - Convert sequence case: ``upper``, ``lower``, ``swapcase``.
   * - :doc:`stylize`
     - Apply a named display style to a sequence or region.
   * - :doc:`clear_gaps`
     - Remove gap characters (``-``) from sequences.
   * - :doc:`clear_annotation`
     - Strip region tags while keeping the underlying sequence.
   * - :doc:`slice_seq`
     - Extract a subsequence by index range, named region, or both.
   * - :doc:`add_prefix`
     - Add a label prefix to sequence names without modifying sequences.

.. toctree::
   :hidden:

   rc
   case_ops
   stylize
   clear_gaps
   clear_annotation
   slice_seq
   add_prefix
