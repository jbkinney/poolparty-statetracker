apply_at_region
===============

Apply an arbitrary callable to the content of a named region, leaving the
flanking sequences unchanged. The function receives the region's content as
a string and must return a string. By default region tags are removed from
the output; set ``remove_tags=False`` to keep them.

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
   * - ``pool``
     - ``Pool | str``
     - *(required)*
     - The Pool to apply the transform to. Can also be a plain sequence string.
   * - ``region_name``
     - ``str``
     - *(required)*
     - Name of the region whose content to transform.
   * - ``transform_fn``
     - ``callable``
     - *(required)*
     - Callable ``(str) -> str`` applied to the region content string.
       Examples: ``str.upper``, ``lambda s: s[::-1]``,
       ``lambda s: s.replace("A", "T")``.
   * - ``rc``
     - ``bool``
     - ``False``
     - When ``True``, reverse-complement the region content before passing
       it to ``transform_fn``, then reverse-complement the result back.
   * - ``remove_tags``
     - ``bool``
     - ``True``
     - When ``True``, region tags are stripped from the output. When
       ``False``, tags are preserved around the transformed content.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.

----

Examples
--------

Uppercase a soft-masked region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``str.upper`` lifts a lowercase region to uppercase without touching the
flanking bases.

.. code-block:: python

    wt         = pp.from_seq("AAAA<cre>atcg</cre>TTTT")
    uppercased = pp.apply_at_region(wt, "cre", str.upper)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; lowercase <em>cre</em> content uppercased)</em>
    AAAA<span class="pp-region">ATCG</span>TTTT
    </div>

Reverse the region content
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A lambda can apply any string transformation. Here the region is reversed
in place (not reverse-complemented; just the character order).

.. code-block:: python

    wt         = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    reversed_r = pp.apply_at_region(wt, "cre", lambda s: s[::-1])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; <em>cre</em> content reversed: ATCG &rarr; GCTA)</em>
    AAAAGCTATTTT
    </div>

Keep region tags in output (remove_tags=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``remove_tags=False`` to preserve the tag markup so the transformed
region is still addressable by name downstream.

.. code-block:: python

    wt       = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    tagged   = pp.apply_at_region(wt, "cre", lambda s: s[::-1],
                                  remove_tags=False)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; reversed content, <em>cre</em> tags retained)</em>
    AAAA<span class="pp-region">GCTA</span>TTTT
    </div>

Substitute bases inside the region (custom lambda)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace all A with T within the region only.

.. code-block:: python

    wt           = pp.from_seq("AAAA<cre>AACGAAT</cre>TTTT")
    substituted  = pp.apply_at_region(
        wt, "cre", lambda s: s.replace("A", "T")
    )

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; A&rarr;T inside <em>cre</em> only; flanks unchanged)</em>
    AAAATCGTTTTTT
    </div>

Operate on reverse complement of region (rc=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rc=True`` reverse-complements the region content before passing it to
``transform_fn``, then reverse-complements the result back. Useful when
the region was defined relative to the minus strand.

.. code-block:: python

    wt          = pp.from_seq("AAAA<cre>atcg</cre>TTTT")
    transformed = pp.apply_at_region(wt, "cre", str.upper, rc=True)
    # RC of "atcg" = "cgat", upper = "CGAT", RC back = "ATCG"

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; uppercased via RC round-trip; content is unchanged)</em>
    AAAAATCGTTTT
    </div>

See :func:`~poolparty.apply_at_region`.
