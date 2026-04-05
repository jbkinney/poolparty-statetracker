Region Operations
=================

Region operations tag, replace, and transform specific parts of sequences
identified by XML-style region markers. For an introduction to regions and
tag syntax, see :doc:`/regions`.

.. code-block:: python

    import poolparty as pp
    pp.init()

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :doc:`insert_tags`
     - Add region tags to a sequence by specifying start/stop indices.
   * - :doc:`remove_tags`
     - Remove region tags, optionally keeping or discarding their contents.
   * - :doc:`annotate_region`
     - Tag a region by position range with an optional display style.
   * - :doc:`replace_region`
     - Replace the content of a named region with sequences from a pool.
   * - :doc:`apply_at_region`
     - Apply a transformation function to the content of a named region.
   * - :doc:`extract_region`
     - Extract the content of a named region as a new pool.
   * - :doc:`region_scan`
     - Insert a named region tag at successive positions within a sequence.
   * - :doc:`region_multiscan`
     - Insert multiple region tags at non-overlapping positions simultaneously.

.. toctree::
   :hidden:

   insert_tags
   remove_tags
   annotate_region
   replace_region
   apply_at_region
   extract_region
   region_scan
   region_multiscan
