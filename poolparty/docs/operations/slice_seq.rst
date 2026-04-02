slice_seq
=========

Extract a subsequence using Python-style slicing, a named region, or both.
When ``keep_context=True``, the sliced content is reassembled into the
original flanking context.

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
     - ``Pool | str``
     - *(required)*
     - Input pool or sequence string.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Region to slice from. A string selects a named region; a
       ``[start, stop]`` pair selects by position. ``None`` uses the full
       sequence.
   * - ``start``
     - ``int | None``
     - ``None``
     - Start index (0-based, Python-style). Applied after region
       extraction when ``region`` is set.
   * - ``stop``
     - ``int | None``
     - ``None``
     - Stop index (exclusive, Python-style).
   * - ``step``
     - ``int | None``
     - ``None``
     - Step size (Python-style).
   * - ``keep_context``
     - ``bool``
     - ``False``
     - If ``True``, reassemble the sliced content back into the original
       flanking context (prefix + sliced + suffix). Requires ``region``.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Iteration priority for downstream multi-pool iteration.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for the operation node name in the pool graph.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to the sliced output.

----

Examples
--------

Slice by position
~~~~~~~~~~~~~~~~~

Extract bases 2–6 from an 8-mer.

.. code-block:: python

    pool   = pp.from_seq("ACGTACGT")
    sliced = pp.slice_seq(pool, start=2, stop=6)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; positions [2:6])</em>
    GTAC
    </div>

Extract a named region
~~~~~~~~~~~~~~~~~~~~~~

Pull the content of a tagged region into its own pool.

.. code-block:: python

    pool = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
    orf  = pp.slice_seq(pool, region="orf")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; content of <em>orf</em>)</em>
    ATGCCC
    </div>

Slice within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine ``region`` with ``start``/``stop`` to slice inside the region. Here,
take only the first codon.

.. code-block:: python

    pool   = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
    codon1 = pp.slice_seq(pool, region="orf", start=0, stop=3)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; first 3 bases of <em>orf</em>)</em>
    ATG
    </div>

Keep context — reassemble flanks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``keep_context=True`` the sliced region content is placed back into the
original flanking sequence.

.. code-block:: python

    pool   = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
    sliced = pp.slice_seq(pool, region="orf", start=0, stop=3, keep_context=True)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; first codon of <em>orf</em> with flanks)</em>
    AAAATGTTT
    </div>

See :func:`~poolparty.slice_seq`.
