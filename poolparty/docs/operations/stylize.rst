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

Examples
--------

Stylize a named region with a built-in style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply ``bold_red`` to the ``cre`` region so it stands out when the pool is
displayed.

.. code-block:: python

    wt     = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    styled = pp.stylize(wt, region="cre", style="bold_red")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; <em>cre</em> region styled bold red)</em>
    AAAA<span style="color:#dc2626;font-weight:bold;">ATCG</span>TTTT
    </div>

Stylize a different region with a different style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply ``bold_blue`` to a ``promoter`` region embedded in a longer construct,
leaving the rest of the sequence unstyled.

.. code-block:: python

    wt     = pp.from_seq("GCGCGC<promoter>TATAAT</promoter>ATGAAATTT")
    styled = pp.stylize(wt, region="promoter", style="bold_blue")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; <em>promoter</em> region styled bold blue)</em>
    GCGCGC<span style="color:#1d4ed8;font-weight:bold;">TATAAT</span>ATGAAATTT
    </div>

Stylize the full sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~

Omit ``region=`` to apply a style to every base in the sequence.

.. code-block:: python

    wt     = pp.from_seq("ATCGATCG")
    styled = pp.stylize(wt, style="cyan")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (1 sequence &mdash; entire sequence styled cyan)</em>
    <span style="color:#0891b2;">ATCGATCG</span>
    </div>

See :func:`~poolparty.stylize`.
