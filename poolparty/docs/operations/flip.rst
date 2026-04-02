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
   :widths: 20 18 12 50

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

Examples
--------

Sequential mode — both orientations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Default ``mode='sequential'`` produces two states: forward then RC.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    flipped = pp.flip(wt)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (2 sequences &mdash; state 0: forward, state 1: RC)</em>
    ATCGATCG<br>
    CGATCGAT
    </div>

Random mode with custom rc_prob
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mode='random'`` samples orientation per draw. Here ``rc_prob=0.3``
means 30% chance of RC on each draw.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    flipped = pp.flip(wt, mode="random", rc_prob=0.3, num_states=5)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (5 stochastic draws &mdash; RC probability 30%)</em>
    ATCGATCG<br>
    ATCGATCG<br>
    CGATCGAT<br>
    ATCGATCG<br>
    ATCGATCG
    </div>

Flip only a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~

``region`` restricts the RC operation to a tagged segment; flanks are
returned unchanged.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    flipped = pp.flip(wt, region="cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (2 sequences &mdash; only <em>cre</em> region flipped; flanks fixed)</em>
    AAAA<span class="pp-region">ATCGATCG</span>TTTT<br>
    AAAA<span class="pp-region">CGATCGAT</span>TTTT
    </div>

Using the ``flip`` design card
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``cards={"flip": "orientation"}`` to add a column recording whether
each sequence is forward or reverse-complemented.

.. code-block:: python

    wt      = pp.from_seq("ATCGATCG")
    flipped = pp.flip(wt, cards={"flip": "orientation"})
    df      = flipped.generate_library()
    # df["orientation"] contains "forward" or "rc" for each row

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (2 sequences &mdash; "orientation" column in output)</em>
    ATCGATCG &nbsp;<em style="color:#6b7280;">orientation=forward</em><br>
    CGATCGAT &nbsp;<em style="color:#6b7280;">orientation=rc</em>
    </div>

Use with iter_order in a join
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``iter_order`` so that ``flip`` is the inner loop when joined with
another pool — every insert appears in both orientations together.

.. code-block:: python

    inserts = pp.from_seqs(["ACGT", "GGCC"], mode="sequential", iter_order=2)
    wt      = pp.from_seq("AAAA<ins/>TTTT")
    both    = pp.flip(pp.replace_region(wt, inserts, "ins"), iter_order=1)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (4 sequences &mdash; each insert in forward then RC orientation)</em>
    AAAACGTTTTT<br>
    AAAACGTTTTT<br>
    <span class="pp-ellipsis">... (forward + RC for each insert)</span>
    </div>

See :func:`~poolparty.flip`.
