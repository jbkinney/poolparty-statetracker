from_seqs
=========

Create a pool that draws from an explicit list of sequences. By default the
pool samples uniformly at random; pass ``mode='sequential'`` to iterate
through the list in order.

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
   * - ``seqs``
     - ``list[str]``
     - *(required)*
     - List of DNA strings. Must not be empty.
   * - ``pool``
     - ``Pool | None``
     - ``None``
     - Background pool. When provided with ``region``, each sequence
       replaces the content of that region. Requires ``region``.
   * - ``region``
     - ``str | None``
     - ``None``
     - Region to replace in ``pool``. Required when ``pool`` is provided.
   * - ``seq_names``
     - ``list[str] | None``
     - ``None``
     - Explicit label for each sequence; populates the ``seq_name`` column
       in :func:`~poolparty.generate_library` output. Cannot be used with
       ``prefix``.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for auto-generated names (e.g. ``'seq_'`` produces
       ``'seq_0'``, ``'seq_1'``, …). Cannot be used with ``seq_names``.
   * - ``mode``
     - ``str``
     - ``'random'``
     - ``'sequential'`` iterates through ``seqs`` in order (cycling if
       ``num_states`` exceeds the list length); ``'random'`` samples
       uniformly at random.
   * - ``num_states``
     - ``int | None``
     - ``None``
     - Number of states to generate. In ``'sequential'`` mode, cycles if
       greater than ``len(seqs)`` or clips if less.
   * - ``iter_order``
     - ``int | None``
     - ``None``
     - Dimension-name ordering for downstream multi-pool iteration.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to every generated sequence.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card columns to include in library output.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.from_seqs` in the
   :doc:`API Reference </api>`.

Examples
--------

Basic list, random mode
~~~~~~~~~~~~~~~~~~~~~~~

``mode='random'`` (the default) — each draw picks one sequence uniformly at
random. This example sets it explicitly and shows one representative draw.

.. code-block:: python

    pool = pp.from_seqs(
        ["AAAA", "CCCC", "GGGG", "TTTT"],
        mode="random",
    )
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=4, num_states=1</em>
    TTTT
    </div>

Sequential mode
~~~~~~~~~~~~~~~

``mode='sequential'`` iterates through the list in order — useful for
reproducible, ordered enumeration of a fixed variant set.

.. code-block:: python

    pool = pp.from_seqs(
        ["ATCG", "CGTA", "GCAT"],
        mode="sequential",
    )
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=4, num_states=3</em>
    ATCG<br>
    CGTA<br>
    GCAT
    </div>

Custom sequence names
~~~~~~~~~~~~~~~~~~~~~

``seq_names`` assigns meaningful labels that appear in library output.

.. code-block:: python

    pool = pp.from_seqs(
        ["ATCG", "ATAG", "AACG"],
        seq_names=["wt", "mut_A", "mut_B"],
        mode="sequential",
    )
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=4, num_states=3</em>
    wt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ATCG<br>
    mut_A&nbsp;&nbsp;ATAG<br>
    mut_B&nbsp;&nbsp;AACG
    </div>

Cycling with ``num_states``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In sequential mode, ``num_states`` greater than ``len(seqs)`` cycles through
the list to produce the requested number of output rows.

.. code-block:: python

    pool = pp.from_seqs(
        ["AAAA", "CCCC"],
        mode="sequential",
        num_states=6,
    )
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=4, num_states=6</em>
    AAAA<br>
    CCCC<br>
    AAAA<br>
    CCCC<br>
    AAAA<br>
    CCCC
    </div>

Inserting into a named region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide ``pool`` and ``region`` to insert each sequence into a fixed context.

.. code-block:: python

    bg   = pp.from_seq("AAAA<cre>XXXX</cre>TTTT")
    pool = pp.from_seqs(
        ["ACGT", "TGCA", "GGCC"],
        pool=bg,
        region="cre",
        mode="sequential",
    )
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool: seq_length=4, num_states=3</em>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>ACGT<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>TGCA<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT<br>
    AAAA<span class="pp-xtag-cre">&lt;cre&gt;</span>GGCC<span class="pp-xtag-cre">&lt;/cre&gt;</span>TTTT
    </div>

See :func:`~poolparty.from_seqs`.
