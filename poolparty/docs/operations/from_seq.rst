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
   :widths: 20 18 12 50

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
     - ``Pool | None``
     - ``None``
     - Background pool. When provided with ``region``, the content of that
       region is replaced by ``seq``. Requires ``region``.
   * - ``region``
     - ``str | None``
     - ``None``
     - Region to replace in ``pool``. Required when ``pool`` is provided.
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

Examples
--------

Basic single sequence
~~~~~~~~~~~~~~~~~~~~~

Create a pool from a plain sequence string.

.. code-block:: python

    pool = pp.from_seq("ACGTACGT")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence)</em>
    ACGTACGT
    </div>

Sequence with embedded region tags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inline tags define named regions that downstream operations can target.

.. code-block:: python

    wt = pp.from_seq("AAAA<promoter>TATAAA</promoter><core>ATCGATCG</core>TTTT")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; regions: <em>promoter</em>, <em>core</em>)</em>
    AAAA<span class="pp-region">TATAAA</span><span class="pp-region">ATCGATCG</span>TTTT
    </div>

Applying a display style
~~~~~~~~~~~~~~~~~~~~~~~~

``style`` controls how the sequence is rendered without changing its value.

.. code-block:: python

    pool = pp.from_seq("ACGTACGT", style="bold_blue")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; bold blue style)</em>
    <strong style="color:#3b82f6;">ACGTACGT</strong>
    </div>

Replacing a named region in a background pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide ``pool`` and ``region`` to substitute the content of a tagged region.

.. code-block:: python

    wt     = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    mutant = pp.from_seq("GGGG", pool=wt, region="cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">wt</em>
    AAAA<span class="pp-region">ATCG</span>TTTT
    </div>
    <div class="pp-pool">
    <em class="pp-header">mutant &mdash; <em>cre</em> replaced with GGGG</em>
    AAAA<span class="pp-region">GGGG</span>TTTT
    </div>

See :func:`~poolparty.from_seq`.
