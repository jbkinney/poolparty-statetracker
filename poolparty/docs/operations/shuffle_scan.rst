shuffle_scan
============

Slide a window of fixed length across the sequence (or a named region) and,
at each position, shuffle the bases within that window. Bases outside the window
are unchanged. Use ``mode='sequential'`` to enumerate every window start as its
own state; ``mode='random'`` (the default) samples window positions according to
``num_states`` (default ``1`` for a single draw). Pair with
``shuffles_per_position`` to list several independent shuffles per draw.

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
     - The Pool to scan. Can also be a plain sequence string.
   * - ``shuffle_length``
     - ``int``
     - *(required)*
     - Width of the shuffle window in bases. A sequence of length *L*
       produces *L* - ``shuffle_length`` + 1 window positions.
   * - ``positions``
     - ``list[int] | None``
     - ``None``
     - Explicit list of window start positions. ``None`` = all valid positions.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Restrict the scan to a named region or ``[start, stop]`` interval.
       Flanks are never modified.
   * - ``shuffle_type``
     - ``str``
     - ``"mono"``
     - ``"mono"`` shuffles individual bases; ``"dinuc"`` preserves
       dinucleotide frequencies.
   * - ``shuffles_per_position``
     - ``int``
     - ``1``
     - Number of independent shuffles generated per window position.
       Values > 1 multiply the library size by that factor.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` iterates positions left-to-right; ``'random'``
       shuffles.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Number of output states. ``None`` auto-computes in sequential mode
       or defaults to 1 in random mode.
   * - ``style``
     - ``str | None``
     - ``None``
     - Named display style applied to the shuffled window.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Iteration priority for downstream multi-pool composition.

.. note::

   With ``shuffle_type="dinuc"``, the **first and last bases of each window
   are always fixed** — this is a mathematical constraint of the Euler-path
   algorithm used to preserve dinucleotide frequencies.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.shuffle_scan` in the
   :doc:`API Reference </api>`.

Examples
--------

3-base shuffle window across an 8-mer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Six starts are valid for a length-3 window on an 8-mer. With ``mode='random'``
and default ``num_states``, the pool has a single state, so ``print_library()``
shows one shuffled draw (reproducible after ``pp.init()`` with default library
generation seeding).

.. code-block:: python

    import poolparty as pp
    pp.init()
    wt   = pp.from_seq("ACGTACGT")
    scan = wt.shuffle_scan(shuffle_length=3, mode="random", style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=8, num_states=1</em>
    ACG<span class="pp-mut">CAT</span>GT
    </div>

Multiple shuffles per position (shuffles_per_position)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``shuffles_per_position=3`` attaches three independent shuffles to the drawn
window; the preview lists each of the three pool states.

.. code-block:: python

    import poolparty as pp
    pp.init()
    wt   = pp.from_seq("ACGTACGT")
    scan = wt.shuffle_scan(shuffle_length=3, shuffles_per_position=3, mode="random",
                           style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=8, num_states=3</em>
    ACG<span class="pp-mut">CAT</span>GT<br>
    ACGT<span class="pp-mut">ACG</span>T<br>
    ACG<span class="pp-mut">TCA</span>GT
    </div>

Explicit position list
~~~~~~~~~~~~~~~~~~~~~~~

Pass ``positions=[0, 3, 6]`` so the shuffle window may start only at those
indices. With ``mode='random'`` and default ``num_states``, one of those starts
is drawn per state (here the preview is a single row).

.. code-block:: python

    import poolparty as pp
    pp.init()
    wt   = pp.from_seq("ACGTACGT")
    scan = wt.shuffle_scan(shuffle_length=2, positions=[0, 3, 6], mode="random",
                           style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=8, num_states=1</em>
    ACG<span class="pp-mut">TA</span>CGT
    </div>

Shuffle scan within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict the scan to the ``cre`` region; flanking sequences are never
shuffled. Literal tags appear in the printed sequence; below they are escaped
for HTML.

.. code-block:: python

    import poolparty as pp
    pp.init()
    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    scan = wt.shuffle_scan(shuffle_length=3, region="cre", mode="random",
                           style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=16, num_states=1</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATC<span class="pp-mut">TAG</span>CG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>

Sequential mode — all window positions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode='sequential'`` enumerates every window start position left-to-right,
producing one shuffled state per position. With a 3-base window on an 8-mer,
there are 6 positions.

.. code-block:: python

    import poolparty as pp
    pp.init()
    wt   = pp.from_seq("ACGTACGT")
    scan = wt.shuffle_scan(shuffle_length=3, mode="sequential", style="red")
    scan.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">scan: seq_length=8, num_states=6</em>
    <span class="pp-mut">GCA</span>TACGT<br>
    A<span class="pp-mut">CGT</span>ACGT<br>
    AC<span class="pp-mut">GAT</span>CGT<br>
    ACG<span class="pp-mut">CAT</span>GT<br>
    ACGT<span class="pp-mut">ACG</span>T<br>
    ACGTA<span class="pp-mut">TGC</span>
    </div>

See :func:`~poolparty.shuffle_scan`.
