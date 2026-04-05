clear_gaps
==========

Remove all gap and non-molecular characters (``-``, ``.``, spaces, and any
other characters outside the DNA alphabet) from sequences. XML region tags are
preserved intact; only characters between tags are filtered. Because the output
length varies with the number of gaps removed, the resulting pool does not
carry a fixed ``seq_length``.

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
     - The Pool (or plain sequence string) to clear gaps from.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Restrict gap removal to a named region or ``[start, stop]`` pair.
   * - ``remove_tags``
     - ``bool | None``
     - ``None``
     - When ``True`` and ``region`` is a name, strip the constraint region
       tags from the output.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.clear_gaps` in the
   :doc:`API Reference </api>`.

Examples
--------

Remove gap markers from a deletion_scan result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``deletion_scan`` replaces deleted bases with ``-`` markers. Pipe the result
through ``clear_gaps`` to produce gapless sequences of varying length.

.. code-block:: python

    wt   = pp.from_seq("ATCGATCG")
    dels = pp.deletion_scan(wt, deletion_length=2, mode="sequential")
    clean = pp.clear_gaps(dels)
    clean.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">clean: seq_length=None, num_states=7</em>
    CGATCG<br>
    AGATCG<br>
    ATATCG<br>
    ATCTCG<br>
    ATCGCG<br>
    ATCGAG<br>
    ATCGAT
    </div>

Clear gaps from a manually gapped sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strip dash characters from a sequence that was constructed with explicit
alignment gaps.

.. code-block:: python

    wt    = pp.from_seq("AT--CG--AT")
    clean = pp.clear_gaps(wt)
    clean.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">clean: seq_length=None, num_states=1</em>
    ATCGAT
    </div>

Chain clear_gaps with another operation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove gaps first, then apply ``rc`` to produce gapless reverse-complement
sequences ready for downstream analysis.

.. code-block:: python

    wt    = pp.from_seq("AT--CG")
    clean = pp.clear_gaps(wt)
    rev   = pp.rc(clean)
    rev.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">rev: seq_length=None, num_states=1</em>
    CGAT
    </div>

See :func:`~poolparty.clear_gaps`.
