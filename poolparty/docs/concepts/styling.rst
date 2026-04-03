Styling
=======

PoolParty can apply **inline styles** to sequences so that mutations,
regions, and other features are visually highlighted when printed. Styles
are tracked per-character through the entire DAG and rendered as ANSI
escape codes in terminal output.

All examples assume::

    import poolparty as pp
    pp.init()

----

Style Specifications
--------------------

A style is a space-separated string of one or more tokens. Tokens can be
combined freely (e.g. ``"blue bold"``).

**Named ANSI colours and modifiers:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Category
     - Values
   * - Foreground
     - ``red``, ``green``, ``yellow``, ``blue``, ``magenta``, ``cyan``,
       ``white``, ``black``
   * - Background
     - ``on_red``, ``on_green``, ``on_yellow``, ``on_blue``,
       ``on_magenta``, ``on_cyan``, ``on_white``, ``on_black``
   * - Modifiers
     - ``bold``, ``underline``, ``blink`` (or ``blinking``),
       ``invert`` (or ``reverse``)

**CSS named colours** — all 140+ standard names are supported:
``coral``, ``salmon``, ``dodgerblue``, ``rebeccapurple``, etc.

**Hex codes** — ``#RRGGBB`` format (e.g. ``"#ff7f50"``).

**Combined examples:**

.. code-block:: python

    style="red"              # red text
    style="blue bold"        # blue + bold
    style="cyan underline"   # cyan + underline
    style="#ff7f50 bold"     # coral (hex) + bold
    style="on_yellow black"  # black text on yellow background

To see every available colour printed in its own colour, call:

.. code-block:: python

    pp.print_named_colors()

----

Applying Styles
---------------

The ``pp.stylize()`` operation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply a style to an entire sequence, a named region, or characters
matching a pattern.

.. code-block:: python

    wt     = pp.from_seq("ATCGATCG")
    styled = pp.stylize(wt, style="red")
    styled.print_library()

The ``which`` parameter selects which characters are styled:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Value
     - Characters styled
   * - ``"contents"`` (default)
     - All non-tag molecular characters.
   * - ``"upper"``
     - Only uppercase A–Z.
   * - ``"lower"``
     - Only lowercase a–z.
   * - ``"gap"``
     - Gap characters (``-``, ``.``, space).
   * - ``"all"``
     - Every character including tag markup.
   * - ``"tags"``
     - Only XML tag characters.

.. code-block:: python

    # Style only uppercase characters
    mixed  = pp.from_seq("AcGtAcGt")
    styled = pp.stylize(mixed, style="bold", which="upper")

    # Style a named region
    wt     = pp.from_seq("GCGCGC<promoter>TATAAT</promoter>ATGAAATTT")
    styled = pp.stylize(wt, region="promoter", style="blue bold")

    # Style characters matching a regex
    styled = pp.stylize(wt, style="cyan", regex="[GC]")

Styles on other operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many operations accept a ``style`` parameter that automatically highlights
the modified positions:

.. code-block:: python

    # Mutations shown in red
    muts = pp.mutagenize(wt, num_mutations=1, style="red")

    # Deletion markers shown in grey
    dels = pp.deletion_scan(wt, deletion_length=2, style="grey")

    # Inserted content shown in green
    scan = pp.insertion_scan(wt, ins_pool, style="green")

The ``style`` parameter on ``pp.from_seq`` colours the entire sequence:

.. code-block:: python

    pool = pp.from_seq("ACGTACGT", style="blue bold")

----

How Styles Render
-----------------

**Terminal / ``print_library()``**
    Styles are rendered as ANSI escape codes. Each styled character gets
    its own escape sequence, and overlapping styles are merged (modifiers
    combine; the last foreground colour wins).

**Jupyter notebooks**
    ANSI codes are embedded in ``print()`` output. Jupyter renders bold
    and some colours, but support varies by notebook frontend. Named ANSI
    colours (``red``, ``blue``) work best; CSS colour names and hex codes
    may fall back to default text colour.

**HTML docs**
    The documentation uses CSS classes (``pp-mut``, ``pp-region``, etc.)
    for colour, not inline ANSI. These are separate from the runtime
    styling system.

----

Suppressing Styles
------------------

To disable style tracking for better performance (e.g. large libraries
where you only need sequence strings):

.. code-block:: python

    pp.toggle_styles(on=False)

When styles are suppressed, ``Seq.style`` is ``None`` and no ANSI codes
are generated. Re-enable with ``pp.toggle_styles(on=True)``.
