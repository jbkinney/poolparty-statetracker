shuffle_scan
============

Slide a window of fixed length across the sequence (or a named region) and,
at each position, randomly shuffle the bases within that window. Bases outside
the window are returned unchanged, making this useful for dinucleotide-controlled
scramble controls.

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
     - ``str | None``
     - ``None``
     - Name of a tagged region to restrict the scan to. Flanks are never
       modified.
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
     - Fix the total number of output states.
   * - ``style``
     - ``str | None``
     - ``None``
     - Named display style applied to the shuffled window.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Controls which axis varies fastest when ``shuffles_per_position > 1``.

.. note::

   With ``shuffle_type="dinuc"``, the **first and last bases of each window
   are always fixed** — this is a mathematical constraint of the Euler-path
   algorithm used to preserve dinucleotide frequencies.

----

Examples
--------

3-base shuffle window across an 8-mer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A length-3 window slides across 6 positions; one stochastic shuffle per
position.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    scan = wt.shuffle_scan(shuffle_length=3)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 6 window positions, each window independently shuffled)</em>
    <span class="pp-mut">CAG</span>TACGT<br>
    A<span class="pp-mut">GTC</span>ACGT<br>
    AC<span class="pp-mut">TGC</span>CGT<br>
    ACG<span class="pp-mut">ATG</span>GT<br>
    <span class="pp-ellipsis">... (6 positions, each stochastic)</span>
    </div>

Multiple shuffles per position (shuffles_per_position)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``shuffles_per_position=3`` generates three independent shuffles at each
position, tripling the library size to 18.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    scan = wt.shuffle_scan(shuffle_length=3, shuffles_per_position=3)

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 6 positions &times; 3 shuffles = 18 variants)</em>
    <span class="pp-mut">CAG</span>TACGT<br>
    <span class="pp-mut">GAC</span>TACGT<br>
    <span class="pp-mut">GCA</span>TACGT<br>
    A<span class="pp-mut">GTC</span>ACGT<br>
    A<span class="pp-mut">CTG</span>ACGT<br>
    <span class="pp-ellipsis">... (18 total)</span>
    </div>

Explicit position list
~~~~~~~~~~~~~~~~~~~~~~~

Pass ``positions=[0, 3, 6]`` to shuffle only three specific windows.

.. code-block:: python

    wt   = pp.from_seq("ACGTACGT")
    scan = wt.shuffle_scan(shuffle_length=2, positions=[0, 3, 6])

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 3 explicit positions, 2-base window shuffled at each)</em>
    <span class="pp-mut">CA</span>GTACGT<br>
    ACG<span class="pp-mut">TA</span>CGT<br>
    ACGTAC<span class="pp-mut">TG</span>
    </div>

Shuffle scan within a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrict the scan to the ``cre`` region; flanking sequences are always
unchanged.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCGATCG</cre>TTTT")
    scan = wt.shuffle_scan(shuffle_length=3, region="cre")

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">Pool (stochastic &mdash; 6 window positions within <em>cre</em>; flanks unchanged)</em>
    AAAA<span class="pp-region"><span class="pp-mut">CAT</span>GATCG</span>TTTT<br>
    AAAA<span class="pp-region">A<span class="pp-mut">GTC</span>ATCG</span>TTTT<br>
    AAAA<span class="pp-region">AT<span class="pp-mut">ACG</span>TCG</span>TTTT<br>
    <span class="pp-ellipsis">... (6 positions; AAAA and TTTT always unchanged)</span>
    </div>

See :func:`~poolparty.shuffle_scan`.
