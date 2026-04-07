flip
====

Produce forward and reverse-complement variants of a pool as distinct
states. Unlike :func:`~poolparty.rc` (which always reverses), ``flip``
tracks orientation as part of the pool's state dimension: sequential
mode enumerates both orientations deterministically; random mode
coin-flips each draw with a configurable probability.

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
     - The background Pool or sequence string. Must be a DNA pool.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Region to flip. A named tag (str), ``[start, stop]`` interval, or
       ``None`` to flip the full sequence.
   * - ``rc_prob``
     - ``float``
     - ``0.5``
     - Probability of reverse complement in random mode. Ignored in
       sequential mode.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated sequence names.
   * - ``mode``
     - ``str``
     - ``'sequential'``
     - ``'sequential'`` enumerates forward then RC as two states;
       ``'random'`` coin-flips per draw using ``rc_prob``.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Override the natural state count. In sequential mode the natural
       count is 2 (cycling if greater). In random mode, ``None`` means
       one draw per call.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to the reverse-complemented variant only.
   * - ``cards``
     - ``list | dict | None``
     - ``None``
     - Design card keys to include. Available key: ``'flip'``
       (value: ``'forward'`` or ``'rc'``).

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.flip` in the
   :doc:`API Reference </api>`.

Examples
--------

Sequential mode — both orientations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode='sequential'`` produces two states: forward then RC.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    flipped = wt.flip(mode="sequential", style="red")
    flipped.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">flipped: seq_length=8, num_states=2</em>
    ATCGATCG<br>
    <span class="pp-mut">CGATCGAT</span>
    </div>

Random mode with custom rc_prob
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode='random'`` samples orientation per draw. Here ``rc_prob=0.3``
means 30% chance of RC on each draw.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    flipped = wt.flip(mode="random", rc_prob=0.3, num_states=5, style="red")
    flipped.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">flipped: seq_length=8, num_states=5</em>
    ATCGATCG<br>
    ATCGATCG<br>
    ATCGATCG<br>
    ATCGATCG<br>
    <span class="pp-mut">CGATCGAT</span>
    </div>

Flip only a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~

``region`` restricts the RC operation to a tagged segment; flanks are
returned unchanged.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    flipped = wt.flip(region="cre", style="red")
    flipped.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">flipped: seq_length=16, num_states=2</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ATCGATCG<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span><span class="pp-mut">CGATCGAT</span><span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>

Use with iter_order in a join
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``iter_order`` so that ``flip`` is the inner loop when joined with
another pool — every insert appears in both orientations together.

.. code-block:: python

    inserts = pp.from_seqs(["ACGT", "GGCC"], mode="sequential", iter_order=2)
    wt      = pp.from_seq("AAAA<ins/>TTTT")
    both    = wt.replace_region(inserts, "ins", sync=False, keep_tags=False).flip(iter_order=1, style="red")
    both.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">both: seq_length=12, num_states=4</em>
    AAAAACGTTTTT<br>
    AAAAGGCCTTTT<br>
    <span class="pp-mut">AAAAACGTTTTT</span><br>
    <span class="pp-mut">AAAAGGCCTTTT</span>
    </div>

See :func:`~poolparty.flip`.
