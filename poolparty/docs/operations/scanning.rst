Scanning Operations
===================

Scanning operations systematically tile a sliding window across a sequence,
applying a transformation at each position. Every scan produces one output
state per window position (or per explicit position list), making them ideal
for saturation-style screens. Multiscan variants place multiple
non-overlapping windows simultaneously.

All scans share a common interface:

- **pool** — the input sequence (or Pool)
- **positions** — optional list of window start positions (default: all valid)
- **region** — restrict the scan to a tagged region
- **mode** — ``'sequential'`` (left-to-right) or ``'random'`` (shuffled)

.. code-block:: python

    import poolparty as pp
    pp.init()

.. toctree::
   :hidden:

   replacement_scan
   deletion_scan
   insertion_scan
   shuffle_scan
   mutagenize_scan
   subseq_scan
   replacement_multiscan
   deletion_multiscan
   insertion_multiscan

.. rubric:: Single-window scans

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`replacement_scan`
     - Replace a sliding window at each position with sequences drawn from an insertion pool.
   * - :doc:`deletion_scan`
     - Systematically delete a fixed-length window at each position.
   * - :doc:`insertion_scan`
     - Insert sequences from a pool at each position along the sequence.
   * - :doc:`shuffle_scan`
     - Shuffle bases within a sliding window at each position.
   * - :doc:`mutagenize_scan`
     - Apply random point mutations within a sliding window tiling across the sequence.
   * - :doc:`subseq_scan`
     - Extract the subsequence at each sliding-window position.

.. rubric:: Multi-window scans

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`replacement_multiscan`
     - Place multiple non-overlapping replacement windows simultaneously.
   * - :doc:`deletion_multiscan`
     - Apply multiple simultaneous deletion windows.
   * - :doc:`insertion_multiscan`
     - Insert sequences at multiple positions simultaneously.
