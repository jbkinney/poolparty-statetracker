clear_annotation
================

Strip all XML region tags and non-molecular characters from sequences, then
uppercase the result. This produces clean, undecorated sequences suitable for
export (e.g., writing to FASTA) or for operations that do not accept tagged
input. When a ``region=`` is specified, only the content inside that tagged
segment is cleaned; the outer tags themselves are removed from the output
according to the ``remove_tags`` setting.

.. code-block:: python

    import poolparty as pp
    pp.init()

----

Parameters
----------

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``pool``
     - ``Pool | str``
     - *(required)*
     - Parent pool or sequence to transform.
   * - ``region``
     - ``str | list | None``
     - ``None``
     - Region to transform: marker name, ``[start, stop]``, or ``None`` for
       the full sequence.
   * - ``remove_tags``
     - ``bool | None``
     - ``None``
     - If ``True`` and ``region`` is a marker name, strip marker tags from
       the output.
   * - ``iter_order``
     - ``float | None``
     - ``None``
     - Iteration order priority for the operation.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for sequence names in the resulting pool.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.clear_annotation` in the
   :doc:`API Reference </api>`.

Examples
--------

Strip tags from a region-tagged sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove the ``cre`` open/close tags and uppercase every base, producing a
plain sequence with no markup.

.. code-block:: python

    wt    = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    plain = pp.clear_annotation(wt)
    plain.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">plain: seq_length=None, num_states=1</em>
    AAAAATCGTTTT
    </div>

Strip tags from the result of a replacement_scan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``replacement_scan`` within a tagged region carries nested region tags.
Pipe through ``clear_annotation`` to yield bare sequences ready for
counting or export.

.. code-block:: python

    wt   = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    alt  = pp.from_seqs(["A", "C", "G", "T"], mode="sequential")
    scan = wt.replacement_scan(replacement_pool=alt, region="cre",
                               mode="sequential")
    bare = pp.clear_annotation(scan)
    bare.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">bare: seq_length=None, num_states=16</em>
    AAAAATCGTTTT<br>
    AAAAACGTTTT<br>
    AAAAATAGTTTT<br>
    AAAAATCATTTT<br>
    AAAACTCGTTTT<br>
    <span class="pp-ellipsis">... (16 total)</span>
    </div>

Clear annotation before saving to FASTA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chain ``clear_annotation`` so downstream steps see plain uppercase
sequences with no markup characters.

.. code-block:: python

    wt      = pp.from_seq("AAAA<cre>ATCG</cre>TTTT")
    mutants = pp.mutagenize(wt, num_mutations=1, region="cre", mode="random")
    clean   = pp.clear_annotation(mutants)
    clean.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">clean: seq_length=None, num_states=1</em>
    AAAAATGGTTTT
    </div>

See :func:`~poolparty.clear_annotation`.
