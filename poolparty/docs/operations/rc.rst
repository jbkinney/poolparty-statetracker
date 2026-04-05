rc
==

Return the reverse complement of a sequence or a named region. By default the
entire sequence is reverse-complemented and region tags are stripped; pass
``region=`` to restrict the operation to a single tagged segment.

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
     - Pool or sequence string to reverse complement.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Named region tag, ``[start, stop]`` interval, or ``None`` for the
       full sequence.
   * - ``remove_tags``
     - ``bool | None``
     - ``None``
     - If ``True`` when ``region`` is a marker name, strip region tags from
       the output; ``None`` uses the operation defaults.
   * - ``iter_order``
     - ``int | float | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for sequence names in the resulting pool.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style passed to :func:`~poolparty.stylize` on the result.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.rc` in the
   :doc:`API Reference </api>`.

Examples
--------

Reverse complement a simple sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Take the reverse complement of the 4-mer ``ATCG``; the result is ``CGAT``.

.. code-block:: python

    import poolparty as pp
    pp.init()

    wt = pp.from_seq("ATCG")
    r  = pp.rc(wt)
    r.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">r: seq_length=4, num_states=1</em>
    CGAT
    </div>

Reverse complement a longer tagged sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reverse-complement the full sequence ``AAAA<cre>ATCGATCG</cre>TTTT``
(tags are stripped before the operation and not carried into the output).

.. code-block:: python

    import poolparty as pp
    pp.init()

    wt = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    r  = pp.rc(wt)
    r.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">r: seq_length=16, num_states=1</em>
    AAAACGATCGATTTTT
    </div>

Reverse complement only a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``region=`` to reverse-complement only the content of the ``cre``
segment; flanking sequences are returned unchanged.

.. code-block:: python

    import poolparty as pp
    pp.init()

    wt = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    r  = pp.rc(wt, region="cre")
    r.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">r: seq_length=8, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>CGATCGAT<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>

See :func:`~poolparty.rc`.
