from_seq
========

Create a pool containing a single, fixed sequence. Inline region tags
(``<name>...</name>``) delimit named segments for later targeting by
mutagenesis, scan, or replacement operations.

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
   * - ``seq``
     - ``str``
     - *(required)*
     - The sequence string. Inline region tags (``<name>...</name>``) are
       parsed and registered automatically.
   * - ``pool``
     - ``Pool | str | None``
     - ``None``
     - Background pool or sequence string. When provided with ``region``,
       the content of that region is replaced by ``seq``. Requires
       ``region``.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Region to replace in ``pool``: a marker name or ``[start, stop]``
       interval. Required when ``pool`` is provided.
   * - ``remove_tags``
     - ``bool | None``
     - ``None``
     - If ``True``, strip region tags from the output.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to the sequence (affects rendering only).
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.from_seq` in the
   :doc:`API Reference </api>`.

Examples
--------

Basic single sequence
~~~~~~~~~~~~~~~~~~~~~

Create a pool from a plain sequence string.

.. code-block:: python

    pool = pp.from_seq("ACGTACGT")
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=8, num_states=1</em>
    ACGTACGT
    </div>

Sequence with embedded region tags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inline tags define named regions that downstream operations can target.

.. code-block:: python

    wt = pp.from_seq("AAAA<promoter>TATAAA</promoter><core>ATCGATCG</core>TTTT")
    wt.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">wt: seq_length=22, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;promoter&gt;</span>TATAAA<span class="pp-xtag-cre">&lt;/promoter&gt;</span><span class="pp-xtag-cre">&lt;core&gt;</span>ATCGATCG<span class="pp-xtag-cre">&lt;/core&gt;</span>TTTT
    </div>

Applying a display style
~~~~~~~~~~~~~~~~~~~~~~~~

``style`` controls how the sequence is rendered without changing its value.

.. code-block:: python

    pool = pp.from_seq("ACGTACGT", style="blue bold")
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=8, num_states=1</em>
    <span class="pp-codon-a">ACGTACGT</span>
    </div>

Replacing a named region in a background pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide ``pool`` and ``region`` to substitute the content of a tagged region.

.. code-block:: python

    import poolparty as pp
    pp.init()
    wt     = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    mutant = pp.from_seq("GGGG", pool=wt, region="cre")
    wt.print_library()
    mutant.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">wt: seq_length=12, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>
    <div class="pp-pool">
    <em class="pp-header">mutant: seq_length=4, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>GGGG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>

See :func:`~poolparty.from_seq`.
