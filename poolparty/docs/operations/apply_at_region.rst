apply_at_region
===============

Apply an arbitrary Pool-to-Pool transform to the content of a named region,
leaving the flanking sequences unchanged. The function receives the region's
content as a :class:`~poolparty.Pool` and must return a
:class:`~poolparty.Pool`. By default region tags are removed from the output;
set ``remove_tags=False`` to keep them.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: auto

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
     - Callable ``(Pool) -> Pool`` applied to the region content.
       Examples: ``lambda p: p.upper()``, ``lambda p: p.rc()``,
       ``lambda p: p.mutagenize(num_mutations=1)``.
   * - ``rc``
     - ``bool``
     - ``False``
     - When ``True``, reverse-complement the region content before passing
       it to ``transform_fn``, then reverse-complement the result back.
       Useful for minus-strand regions.
   * - ``remove_tags``
     - ``bool``
     - ``True``
     - When ``True``, region tags are stripped from the output. When
       ``False``, tags are preserved around the transformed content.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Enumeration order when combined with other pools.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.apply_at_region` in the
   :doc:`API Reference </api>`.

Examples
--------

Uppercase a soft-masked region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``p.upper()`` lifts a lowercase region to uppercase without touching the
flanking bases. Tags are removed by default.

.. code-block:: python

    wt         = pp.from_seq("AAAA<cre>atcg</cre>TTTT")
    uppercased = pp.apply_at_region(wt, "cre", lambda p: p.upper())
    uppercased.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">uppercased: seq_length=12, num_states=1</em>
    AAAAATCGTTTT
    </div>

Reverse-complement a region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``p.rc()`` reverse-complements only the region content (``ATCG`` becomes
``CGAT``) while the flanking bases remain untouched.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    flipped = pp.apply_at_region(wt, "cre", lambda p: p.rc())
    flipped.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">flipped: seq_length=12, num_states=1</em>
    AAAACGATTTTT
    </div>

Keep region tags in output (remove_tags=False)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``remove_tags=False`` to preserve the tag markup so the transformed
region is still addressable by name downstream.

.. code-block:: python

    wt     = pp.from_seq("AAAA<cre>atcg</cre>TTTT")
    tagged = pp.apply_at_region(wt, "cre", lambda p: p.upper(),
                                remove_tags=False)
    tagged.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">tagged: seq_length=12, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>

Mutagenize within a region
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any Pool operation can serve as the transform. Here ``mutagenize`` with
``mode='sequential'`` generates every single-base substitution inside the
region while the flanks stay wild-type.

.. code-block:: python

    wt     = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    mutant = pp.apply_at_region(wt, "cre",
                 lambda p: p.mutagenize(num_mutations=1,
                                        mode="sequential"))
    mutant.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">mutant: seq_length=12, num_states=12</em>
    AAAACTCGTTTT<br>
    AAAAGTCGTTTT<br>
    AAAATTCGTTTT<br>
    AAAAAACGTTTT<br>
    AAAAACCGTTTT
    <span class="pp-ellipsis">... (12 total)</span>
    </div>

See :func:`~poolparty.apply_at_region`.
