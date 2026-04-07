Styling
=======

PoolParty can apply **inline styles** to sequences so that mutations,
regions, and other features are visually highlighted when printed. Styles
are tracked per-character through the entire DAG and rendered as ANSI
escape codes in terminal output.

All examples assume:

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Style specifications
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

**CSS named colours**: all 140+ standard names are supported.
``coral``, ``salmon``, ``dodgerblue``, ``rebeccapurple``, etc.

**Hex codes**: ``#RRGGBB`` format (e.g. ``"#ff7f50"``).

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

Applying styles
---------------

.. rubric:: The ``stylize`` operation

Apply a style to an entire sequence, a named
:doc:`region </regions>`, or characters matching a pattern.

.. code-block:: python

    wt     = pp.from_seq("ATCGATCG")
    styled = wt.stylize(style="red")
    styled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">styled: seq_length=8, num_states=1</em>
    <span class="pp-mut">ATCGATCG</span>
    </div>

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

    # Style a named region
    wt     = pp.from_seq("GCGC<cre>TATAAT</cre>ATGAAATTT")
    styled = wt.stylize(region="cre", style="blue bold")
    styled.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">styled: seq_length=19, num_states=1</em>
    GCGC<span class="pp-xtag-light">&lt;cre&gt;</span><span class="pp-region">TATAAT</span><span class="pp-xtag-light">&lt;/cre&gt;</span>ATGAAATTT
    </div>

.. code-block:: python

    # Style only uppercase characters
    mixed  = pp.from_seq("AcGtAcGt")
    styled = mixed.stylize(style="bold", which="upper")

    # Style characters matching a regex
    styled = wt.stylize(style="cyan", regex="[GC]")

.. rubric:: Styles on other operations

Many operations accept a ``style`` parameter that automatically highlights
the modified positions:

.. code-block:: python

    wt   = pp.from_seq("GCGC<cre>TATAAT</cre>ATGAAATTT")
    muts = wt.mutagenize(num_mutations=1, style="red")
    muts.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">muts: seq_length=19, num_states=1</em>
    <span class="pp-ellipsis"># mutated positions shown in red</span><br>
    <span class="pp-mut">A</span>CGC<span class="pp-xtag-light">&lt;cre&gt;</span>TATAAT<span class="pp-xtag-light">&lt;/cre&gt;</span>ATGAAATTT<br>
    G<span class="pp-mut">A</span>GC<span class="pp-xtag-light">&lt;cre&gt;</span>TATAAT<span class="pp-xtag-light">&lt;/cre&gt;</span>ATGAAATTT<br>
    GC<span class="pp-mut">A</span>C<span class="pp-xtag-light">&lt;cre&gt;</span>TATAAT<span class="pp-xtag-light">&lt;/cre&gt;</span>ATGAAATTT<br>
    <span class="pp-ellipsis">...</span>
    </div>

.. code-block:: python

    # Deletion markers shown in grey
    dels = wt.deletion_scan(deletion_length=2, style="grey")

    # Inserted content shown in green
    ins  = pp.from_seqs(["AA", "CC", "GG"], mode="sequential")
    scan = wt.insertion_scan(ins, style="green")

The ``style`` parameter on ``pp.from_seq`` colours the entire sequence:

.. code-block:: python

    pool = pp.from_seq("ACGTACGT", style="blue bold")

For codon-aware styling that highlights reading frames, see
:doc:`/operations/stylize_orf` and the ``style_codons`` parameter on
:doc:`/operations/annotate_orf`.

----

How styles render
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

Suppressing styles
------------------

To disable style tracking for better performance (e.g. large libraries
where you only need sequence strings):

.. code-block:: python

    pp.toggle_styles(on=False)

When styles are suppressed, ``Seq.style`` is ``None`` and no ANSI codes
are generated. Re-enable with ``pp.toggle_styles(on=True)``.
