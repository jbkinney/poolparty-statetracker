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
   :widths: auto
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

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.slice_seq` in the
   :doc:`API Reference </api>`.

Examples
--------

Slice by position
~~~~~~~~~~~~~~~~~

Extract bases 2–6 from an 8-mer.

.. code-block:: python

    pool   = pp.from_seq("ACGTACGT")
    sliced = pp.slice_seq(pool, start=2, stop=6)
    sliced.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">sliced: seq_length=4, num_states=1</em>
    GTAC
    </div>

Extract a named region
~~~~~~~~~~~~~~~~~~~~~~

Pull the content of a tagged region into its own pool.

.. code-block:: python

    pool = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
    orf  = pp.slice_seq(pool, region="orf")
    orf.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">orf: seq_length=6, num_states=1</em>
    ATGCCC
    </div>

Slice within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine ``region`` with ``start``/``stop`` to slice inside the region. Here,
take only the first codon.

.. code-block:: python

    pool   = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
    codon1 = pp.slice_seq(pool, region="orf", start=0, stop=3)
    codon1.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">codon1: seq_length=3, num_states=1</em>
    ATG
    </div>

Keep context — reassemble flanks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``keep_context=True`` the sliced region content is placed back into the
original flanking sequence.

.. code-block:: python

    pool   = pp.from_seq("AAA<orf>ATGCCC</orf>TTT")
    sliced = pp.slice_seq(pool, region="orf", start=0, stop=3, keep_context=True)
    sliced.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">sliced: seq_length=9, num_states=1</em>
    AAAATGTTT
    </div>

See :func:`~poolparty.slice_seq`.
