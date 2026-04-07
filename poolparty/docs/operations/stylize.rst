stylize
=======

Attach an inline display style to sequences (or to a named region) without
modifying the underlying bases. Styles control the colour and weight used when
printing pools to the terminal or rendering them in HTML documentation. Use
``which=`` to target only uppercase characters, lowercase characters, gap
characters, or the full contents of a region; ``regex=`` allows arbitrary
pattern-based targeting.

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
     - Parent pool or sequence string to stylize.
   * - ``region``
     - ``str | Sequence[int] | None``
     - ``None``
     - Region to restrict styling (marker name or ``[start, stop]`` nontag
       coordinates). ``None`` styles the entire sequence.
   * - ``style``
     - ``str``
     - *(required)*
     - Style specification (e.g. ``"red bold"``, ``"blue"``, ``"cyan"``).
   * - ``which``
     - ``str``
     - ``'contents'``
     - Pattern selector when ``regex`` is ``None``: ``'all'``, ``'upper'``,
       ``'lower'``, ``'gap'``, ``'tags'``, or ``'contents'``.
   * - ``regex``
     - ``str | None``
     - ``None``
     - Custom regex; when set, overrides ``which``.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Enumeration order when combined with other pools.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for sequence names in the resulting pool.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.stylize` in the
   :doc:`API Reference </api>`.

Examples
--------

Stylize a named region with a built-in style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply ``red bold`` to the ``cre`` region so it stands out when the pool is
displayed.

.. code-block:: python

    wt     = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    styled = pp.stylize(wt, region="cre", style="red bold")
    styled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">styled: seq_length=12, num_states=1</em>
    AAAA<span class="pp-xtag-light">&lt;cre&gt;</span><strong class="pp-mut">ATCG</strong><span class="pp-xtag-light">&lt;/cre&gt;</span>TTTT
    </div>

Stylize a different region with a different style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply ``blue bold`` to a ``promoter`` region embedded in a longer construct,
leaving the rest of the sequence unstyled.

.. code-block:: python

    wt     = pp.from_seq("GCGCGC<promoter>TATAAT</promoter>ATGAAATTT")
    styled = pp.stylize(wt, region="promoter", style="blue bold")
    styled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">styled: seq_length=21, num_states=1</em>
    GCGCGC<span class="pp-xtag-light">&lt;promoter&gt;</span><strong class="pp-codon-a">TATAAT</strong><span class="pp-xtag-light">&lt;/promoter&gt;</span>ATGAAATTT
    </div>

Stylize the full sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~

Omit ``region=`` to apply a style to every base in the sequence.

.. code-block:: python

    wt     = pp.from_seq("ATCGATCG")
    styled = pp.stylize(wt, style="cyan")
    styled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">styled: seq_length=8, num_states=1</em>
    <span style="color:#0891b2;">ATCGATCG</span>
    </div>

See :func:`~poolparty.stylize`.
